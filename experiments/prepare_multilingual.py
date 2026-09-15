"""Materialize disjoint calibration/regression slices from pinned FLEURS files.

Each language contributes the first 100 unique, nonempty-reference clips no
longer than 12 seconds in source order: 50 calibration, then 50 regression.
Selection reads metadata and audio only, never model predictions.
"""

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
import soundfile as sf


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def acquire(record, cache):
    path = cache / f"fleurs-{record['config']}-{record['revision']}.parquet"
    if path.exists():
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Cached source differs from its pinned hash")
        return path
    temporary = path.with_suffix(".partial")
    received, started, checksum = 0, time.monotonic(), hashlib.sha256()
    with (
        urlopen(record["url"], timeout=60) as source,
        temporary.open("wb") as destination,
    ):
        while chunk := source.read(1024 * 1024):
            received += len(chunk)
            if received > record["bytes"] or time.monotonic() - started > 1200:
                raise ValueError("Download exceeded its size or time budget")
            destination.write(chunk)
            checksum.update(chunk)
    if received != record["bytes"] or checksum.hexdigest() != record["sha256"]:
        raise ValueError("Downloaded source differs from its pinned hash")
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("artifacts/evaluation/sources")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    args.output.mkdir(parents=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    report = {
        "complete": False,
        "sources_sha256": digest(args.sources),
        "languages": [],
        "license": "cc-by-4.0",
        "reference_status": "dataset-provided transcription",
        "selection": "first 100 unique eligible rows per language, <=12s; first 50 calibration, next 50 regression",
    }
    groups = {"calibration": [], "regression": []}
    seen = set()
    try:
        for record in sources:
            if not record["url"].startswith(
                f"https://huggingface.co/datasets/google/fleurs/resolve/{record['revision']}/parquet-data/"
            ):
                raise ValueError("Unexpected corpus source URL")
            path = acquire(record, args.cache)
            selected, ordinal = [], 0
            source = pq.ParquetFile(path)
            try:
                for batch in source.iter_batches(batch_size=16):
                    for row in batch.to_pylist():
                        row_index, ordinal = ordinal, ordinal + 1
                        blob = row["audio"]["bytes"]
                        info = sf.info(io.BytesIO(blob))
                        reference = row["raw_transcription"]
                        if (
                            not 0 < info.duration <= 12
                            or not isinstance(reference, str)
                            or not reference.strip()
                        ):
                            continue
                        audio_hash = hashlib.sha256(blob).hexdigest()
                        if audio_hash in seen:
                            continue
                        seen.add(audio_hash)
                        suffix = (
                            ".wav"
                            if info.format == "WAV"
                            else ".flac"
                            if info.format == "FLAC"
                            else ".audio"
                        )
                        audio = (
                            args.output
                            / "audio"
                            / f"{record['config']}-{row_index:05d}{suffix}"
                        )
                        audio.parent.mkdir(parents=True, exist_ok=True)
                        audio.write_bytes(blob)
                        selected.append(
                            {
                                "id": f"fleurs-{record['config']}-test-{row_index}",
                                "audio_path": str(audio.resolve()),
                                "audio_sha256": audio_hash,
                                "reference": reference,
                                "language": record["language"],
                                "duration_seconds": info.duration,
                                "source_row": row_index,
                                "source_id": row["id"],
                                "gender": row["gender"],
                                "dataset_revision": record["revision"],
                                "dataset_config": record["config"],
                            }
                        )
                        if len(selected) == 100:
                            break
                    if len(selected) == 100:
                        break
            finally:
                source.close()
            if len(selected) != 100:
                raise ValueError(f"Too few eligible samples for {record['language']}")
            for role, rows in (
                ("calibration", selected[:50]),
                ("regression", selected[50:]),
            ):
                for row in rows:
                    row["evaluation_role"] = f"round2_multilingual_{role}"
                groups[role].extend(rows)
            report["languages"].append(
                {**record, "eligible_selected": 100, "source_rows_scanned": ordinal}
            )
            print(
                json.dumps({"language": record["language"], "selected": 100}),
                flush=True,
            )
        for role, rows in groups.items():
            path = args.output / f"{role}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            report[f"{role}_sha256"] = digest(path)
        report["complete"] = True
    finally:
        (args.output / "provenance.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
