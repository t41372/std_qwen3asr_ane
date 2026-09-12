"""Download two official Qwen fixtures, or materialize a bounded dataset subset.

Smoke uses only stdlib + soundfile. Larger corpora lazily require datasets;
neither mode acquires a model. References always come from the named source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import soundfile as sf

QWEN_REVISION = "7c6daf77a2421100f5fb066495372c00129d39ff"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/{QWEN_REVISION}/"
    "examples/example_qwen3_forced_aligner.py"
)
DATASETS = {
    "librispeech": {
        "repo": "openslr/librispeech_asr",
        "config": "clean",
        "language": "en",
        "license": "cc-by-4.0",
        "text": "text",
    },
    "fleurs": {
        "repo": "google/fleurs",
        "config": "cmn_hans_cn",
        "language": "zh",
        "license": "cc-by-4.0",
        "text": "raw_transcription",
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    with urlopen(url, timeout=60) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Download exceeds {max_bytes} bytes: {url}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_audio(data: bytes, path: Path) -> dict[str, Any]:
    # Decode before writing, so an HTML error response never becomes a fixture.
    info = sf.info(io.BytesIO(data))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Empty or invalid audio: {path}")
    if path.exists():
        raise ValueError(f"Refusing to overwrite audio: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "audio_sha256": sha256(data),
        "audio_bytes": len(data),
        "source_sample_rate": info.samplerate,
        "source_channels": info.channels,
        "duration_seconds": info.duration,
    }


def prepare_smoke(root: Path, manifest: Any, provenance: dict[str, Any]) -> None:
    source = download(REFERENCE_URL, max_bytes=1024 * 1024)
    constants = {}
    # Parse literals without importing or executing upstream Python.
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in {"URL_EN", "URL_ZH", "TEXT_EN", "TEXT_ZH"}:
                constants[name] = ast.literal_eval(node.value)
    provenance.update(
        source_repository="https://github.com/QwenLM/Qwen3-ASR",
        source_revision=QWEN_REVISION,
        reference_source_url=REFERENCE_URL,
        reference_source_sha256=sha256(source),
        reference_status="author-provided forced-aligner example text; not independently human-verified",
        source_code_license="Apache-2.0",
        audio_license="No separate audio license found in the example; retain original URL and avoid redistribution claims.",
    )
    for language in ("en", "zh"):
        url = constants[f"URL_{language.upper()}"]
        text = constants[f"TEXT_{language.upper()}"]
        if not isinstance(url, str) or not isinstance(text, str) or not text:
            raise ValueError(f"Invalid upstream constants for {language}")
        name = f"qwen_official_{language}.wav"
        details = write_audio(download(url), root / name)
        row = {
            "id": f"qwen-official-{language}",
            "audio_path": name,
            "reference": text,
            "language": language,
            "split": "smoke",
            "source_url": url,
            "reference_source_url": REFERENCE_URL,
            "reference_status": provenance["reference_status"],
            **details,
        }
        manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest.flush()
        provenance["samples"].append(row)
        print(
            f"Prepared {row['id']}: {details['duration_seconds']:.2f}s, {details['audio_bytes']} bytes"
        )


def prepare_corpus(
    args: argparse.Namespace, manifest: Any, provenance: dict[str, Any]
) -> None:
    # Optional only: uv run --with datasets. decode=False avoids torchcodec/ffmpeg.
    from datasets import Audio, load_dataset

    spec = DATASETS[args.dataset]
    config = args.config or spec["config"]
    language = args.language or spec["language"]
    if args.dataset == "fleurs" and config != spec["config"] and not args.language:
        raise ValueError("Set --language when selecting a non-default FLEURS config")
    provenance.update(
        dataset_repository=spec["repo"],
        dataset_revision=args.revision,
        dataset_config=config,
        split=args.split,
        license=spec["license"],
        dataset_url=f"https://huggingface.co/datasets/{spec['repo']}",
        selection={
            "method": "contiguous rows in pinned streaming order",
            "offset": args.offset,
            "count": args.count,
        },
        reference_status="dataset-provided transcription",
        reference_field=spec["text"],
    )
    dataset = load_dataset(
        spec["repo"],
        config,
        split=args.split,
        revision=args.revision,
        streaming=True,
        cache_dir=str(args.output / ".datasets-cache"),
    )
    dataset = (
        dataset.cast_column("audio", Audio(decode=False))
        .skip(args.offset)
        .take(args.count)
    )
    for index, example in enumerate(dataset, start=args.offset):
        audio = example["audio"]
        data = audio.get("bytes")
        if data is None:
            source_path = Path(audio["path"])
            if not source_path.is_file():
                raise ValueError(
                    f"Dataset row {index} has no materialized audio bytes: {audio.get('path')}"
                )
            data = source_path.read_bytes()
        if len(data) > 16 * 1024 * 1024:
            raise ValueError(f"Dataset row {index} exceeds 16 MiB fixture limit")
        reference = example[spec["text"]]
        if not isinstance(reference, str):
            raise TypeError(f"Dataset row {index} reference is not text")
        # Container format is detected by libsndfile, not guessed from upstream path.
        extension = ".flac" if sf.info(io.BytesIO(data)).format == "FLAC" else ".wav"
        name = f"audio/{index:06d}{extension}"
        details = write_audio(data, args.output / name)
        row = {
            "id": f"{args.dataset}-{config}-{args.split}-{index}",
            "audio_path": name,
            "reference": reference,
            "language": language,
            "split": args.split,
            "source_row": index,
            "source_id": example.get("id"),
            "speaker_id": example.get("speaker_id"),
            "dataset_revision": args.revision,
            **details,
        }
        manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest.flush()
        provenance["samples"].append(row)
        print(f"Prepared {row['id']}", flush=True)
    if len(provenance["samples"]) != args.count:
        raise ValueError(
            f"Requested {args.count} rows; received {len(provenance['samples'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("smoke", *DATASETS), default="smoke")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/smoke")
    )
    parser.add_argument(
        "--revision",
        help="Required pinned 40-character dataset commit for non-smoke runs",
    )
    parser.add_argument("--config")
    parser.add_argument("--language")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.count < 1 or args.offset < 0:
        parser.error("count must be positive; offset nonnegative")
    if args.dataset != "smoke" and (
        not args.revision
        or len(args.revision) != 40
        or any(c not in "0123456789abcdef" for c in args.revision)
    ):
        parser.error("Corpus preparation requires --revision <40-character commit SHA>")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    provenance_path = args.output / "provenance.json"
    if manifest_path.exists() or provenance_path.exists():
        parser.error(
            f"Output already contains a manifest/provenance; select a new directory: {args.output}"
        )
    provenance = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "complete": False,
        "error": None,
        "samples": [],
    }
    try:
        with manifest_path.open("x") as manifest:
            if args.dataset == "smoke":
                prepare_smoke(args.output, manifest, provenance)
            else:
                prepare_corpus(args, manifest, provenance)
        provenance["complete"] = True
    except Exception as exc:  # noqa: BLE001 — preserve partial preparation and failure provenance.
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        print(provenance["error"], file=sys.stderr)
    finally:
        write_json(provenance_path, provenance)
    return 0 if provenance["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
