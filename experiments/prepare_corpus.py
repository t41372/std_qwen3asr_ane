"""Prepare fixed, duration-bounded, metadata-balanced English/Chinese test subsets.

Requires temporary pyarrow (uv run --with 'pyarrow>=20,<22'). No model is loaded.
One verified parquet per corpus is cached in the workspace; selection never
reads model predictions. FLEURS lacks speaker IDs, so only gender is balanced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pyarrow.parquet as pq
import soundfile as sf
from prepare_eval import DATASETS, write_audio, write_json

SOURCES = {
    "librispeech": {
        "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        "path": "clean/test/0000.parquet",
        "sha256": "7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0",
        "bytes": 350452636,
    },
    "fleurs": {
        "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "path": "parquet-data/cmn_hans_cn/test-00000-of-00001.parquet",
        "sha256": "87c0aebbe183f3a36ac87b5c3421b6ab57036824744ff695029a3f858e7622fd",
        "bytes": 695674033,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_source(args: argparse.Namespace, provenance: dict[str, Any]) -> Path:
    spec, source = DATASETS[args.dataset], SOURCES[args.dataset]
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{source['revision']}/{source['path']}"
    path = (args.source_dir / f"{args.dataset}-{source['revision']}.parquet").resolve()
    provenance["source_parquet"] = {**source, "local_path": str(path), "url": url}
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".parquet.partial")
        started, received = time.monotonic(), 0
        print(f"Downloading {source['bytes']} bytes from {url}", flush=True)
        # A single attempt is bounded by byte count, socket timeout and elapsed time.
        with urlopen(url, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if received > source["bytes"] or time.monotonic() - started > 600:
                    raise ValueError(
                        "Download exceeded its declared size or 10-minute budget"
                    )
                if received % (64 * 1024 * 1024) < 1024 * 1024:
                    print(f"Downloaded {received / 1024**2:.0f} MiB", flush=True)
        if received != source["bytes"] or file_sha256(partial) != source["sha256"]:
            raise ValueError(
                "Downloaded parquet differs from pinned Hugging Face LFS size/SHA256"
            )
        partial.replace(path)
    if path.stat().st_size != source["bytes"] or file_sha256(path) != source["sha256"]:
        raise ValueError(
            "Cached parquet differs from pinned Hugging Face LFS size/SHA256"
        )
    return path


def stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def scan_candidates(
    source: pq.ParquetFile, args: argparse.Namespace, provenance: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    spec = DATASETS[args.dataset]
    schema_names = set(source.schema_arrow.names)
    grouping = "speaker_id" if "speaker_id" in schema_names else "gender"
    if grouping not in schema_names:
        raise ValueError(
            "Corpus has neither speaker_id nor gender for metadata-balanced sampling"
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: dict[str, int] = defaultdict(int)
    invalid_rows = []
    index = 0
    for batch in source.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            current_index, index = index, index + 1
            try:
                audio = row["audio"]["bytes"]
                info = sf.info(io.BytesIO(audio))
                reference = row[spec["text"]]
                if not isinstance(reference, str) or not reference.strip():
                    exclusions["empty_reference"] += 1
                    continue
                if not 0 < info.duration <= args.max_duration:
                    exclusions["duration_outside_limit"] += 1
                    continue
                if len(audio) > 16 * 1024 * 1024:
                    exclusions["audio_bytes_over_limit"] += 1
                    continue
                group = str(row[grouping])
                key = stable_order(args.seed, f"{current_index}:{row.get('id')}")
                groups[group].append({"row": current_index, "key": key, "group": group})
            except Exception as exc:  # noqa: BLE001 — preserve every invalid source row.
                invalid_rows.append(
                    {"row": current_index, "error": f"{type(exc).__name__}: {exc}"}
                )
    provenance["selection"] = {
        "method": "stable-hash within groups; round-robin across stable-hash ordered groups",
        "grouping_field": grouping,
        "seed": args.seed,
        "count": args.count,
        "max_duration_seconds": args.max_duration,
        "source_rows": index,
        "eligible_rows": sum(len(group) for group in groups.values()),
        "eligible_groups": len(groups),
        "exclusions": dict(exclusions),
        "invalid_rows": invalid_rows,
        "prefixes_are_nested": True,
        "selection_bias": "Duration-filtered metadata-balanced engineering subset; not a random corpus WER estimate.",
        "speaker_diversity_verified": grouping == "speaker_id",
    }
    if invalid_rows:
        raise ValueError(
            f"{len(invalid_rows)} invalid source rows; inspect provenance before selecting"
        )
    return groups, grouping


def select_rows(
    groups: dict[str, list[dict[str, Any]]], count: int, seed: int
) -> list[dict[str, Any]]:
    for members in groups.values():
        members.sort(key=lambda row: row["key"])
    ordered_groups = sorted(groups, key=lambda group: stable_order(seed, group))
    selected, depth = [], 0
    while len(selected) < count:
        previous = len(selected)
        for group in ordered_groups:
            if depth < len(groups[group]) and len(selected) < count:
                selected.append(groups[group][depth])
        if len(selected) == previous:
            raise ValueError(
                f"Only {len(selected)} eligible rows for requested {count}"
            )
        depth += 1
    return selected


def prepare(args: argparse.Namespace, provenance: dict[str, Any]) -> None:
    spec = DATASETS[args.dataset]
    source = pq.ParquetFile(acquire_source(args, provenance))
    groups, grouping = scan_candidates(source, args, provenance)
    selected = select_rows(groups, args.count, args.seed)
    wanted = {row["row"] for row in selected}
    prepared = {}
    index = 0
    for batch in source.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            current_index, index = index, index + 1
            if current_index not in wanted:
                continue
            audio = row["audio"]["bytes"]
            extension = (
                ".flac" if sf.info(io.BytesIO(audio)).format == "FLAC" else ".wav"
            )
            name = f"audio/{current_index:06d}{extension}"
            details = write_audio(audio, args.output / name)
            prepared[current_index] = {
                "id": f"{args.dataset}-{spec['config']}-test-{current_index}",
                "audio_path": name,
                "reference": row[spec["text"]],
                "language": spec["language"],
                "split": "test",
                "source_row": current_index,
                "source_id": row.get("id"),
                "speaker_id": row.get("speaker_id"),
                "gender": row.get("gender"),
                "sampling_group": str(row[grouping]),
                "dataset_revision": SOURCES[args.dataset]["revision"],
                **details,
            }
        if len(prepared) == len(wanted):
            break
    if len(prepared) != len(wanted):
        raise ValueError("Not every selected parquet row could be materialized")
    counts: dict[str, int] = defaultdict(int)
    with (args.output / "manifest.jsonl").open("x") as manifest:
        for choice in selected:
            row = prepared[choice["row"]]
            counts[row["sampling_group"]] += 1
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            provenance["samples"].append(row)
    provenance["selection"]["selected_group_counts"] = dict(counts)
    provenance["selection"]["selected_source_rows"] = [row["row"] for row in selected]
    print(
        f"Prepared {len(selected)} samples from {len(counts)} {grouping} groups",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("artifacts/evaluation/sources")
    )
    args = parser.parse_args()
    if args.count < 1 or not 0 < args.max_duration <= 30:
        parser.error("count must be positive; max-duration must be in (0, 30]")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[args.dataset]
    provenance = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "dataset_repository": spec["repo"],
        "dataset_revision": SOURCES[args.dataset]["revision"],
        "dataset_config": spec["config"],
        "split": "test",
        "license": spec["license"],
        "dataset_url": f"https://huggingface.co/datasets/{spec['repo']}",
        "reference_status": "dataset-provided transcription",
        "reference_field": spec["text"],
        "complete": False,
        "error": None,
        "samples": [],
    }
    try:
        prepare(args, provenance)
        provenance["complete"] = True
    except Exception as exc:  # noqa: BLE001 — keep exact preparation failures with partial artifacts.
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        print(provenance["error"], file=sys.stderr)
    finally:
        write_json(args.output / "provenance.json", provenance)
    return 0 if provenance["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
