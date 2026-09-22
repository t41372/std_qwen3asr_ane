"""Private offline CPU worker for the optional Qwen forced aligner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .alignment import (
    ALIGNMENT_MODEL_ID,
    ALIGNMENT_MODEL_REVISION,
    ALIGNMENT_REQUIRED_FILES,
    ALIGNMENT_SAMPLE_RATE,
    ALIGNMENT_WEIGHTS_SHA256,
    ALIGNMENT_WEIGHTS_SIZE,
    MAX_ALIGNMENT_SECONDS,
    SUPPORTED_ALIGNMENT_LANGUAGES,
)

_MODEL_RECEIPT = "alignment-model.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_model(path: Path) -> None:
    missing = [name for name in ALIGNMENT_REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded model is missing files: {', '.join(missing)}")
    weights = path / "model.safetensors"
    if weights.stat().st_size != ALIGNMENT_WEIGHTS_SIZE:
        raise RuntimeError("Downloaded alignment weights have the wrong size")
    if _sha256(weights) != ALIGNMENT_WEIGHTS_SHA256:
        raise RuntimeError("Downloaded alignment weights do not match the pinned SHA-256")


def _receipt(*, source: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": ALIGNMENT_MODEL_ID,
        "revision": ALIGNMENT_MODEL_REVISION,
        "weights_size": ALIGNMENT_WEIGHTS_SIZE,
        "weights_sha256": ALIGNMENT_WEIGHTS_SHA256,
        "required_files": list(ALIGNMENT_REQUIRED_FILES),
        "license": "Apache-2.0",
        "source": source,
    }


def _write_receipt(destination: Path, *, source: str) -> None:
    temporary = destination / f".{_MODEL_RECEIPT}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(_receipt(source=source), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination / _MODEL_RECEIPT)
    finally:
        temporary.unlink(missing_ok=True)


def _has_pinned_receipt(destination: Path) -> bool:
    try:
        receipt = json.loads((destination / _MODEL_RECEIPT).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected = _receipt(source=str(receipt.get("source", "")))
    return receipt == expected


def acquire_model(destination: Path, *, allow_downloads: bool) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            _validate_model(destination)
            if not _has_pinned_receipt(destination):
                _write_receipt(destination, source="verified_existing_directory")
            return
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=ALIGNMENT_MODEL_ID,
                revision=ALIGNMENT_MODEL_REVISION,
                allow_patterns=list(ALIGNMENT_REQUIRED_FILES),
                local_dir=temporary,
                local_files_only=not allow_downloads,
            )
            _validate_model(temporary)
            _write_receipt(temporary, source="huggingface_hub.snapshot_download")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    output = bytearray(size)
    view = memoryview(output)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError("Unexpected end of alignment audio frame")
        offset += count
    return bytes(output)


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.buffer.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    sys.stdout.buffer.flush()


def serve(model_path: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    )
    _write_response({"status": "ready"})
    while line := sys.stdin.buffer.readline():
        try:
            request = json.loads(line)
            if request.get("operation") != "align":
                raise ValueError("Unknown worker operation")
            sample_rate = int(request["sample_rate"])
            sample_count = int(request["sample_count"])
            audio_bytes = int(request["audio_bytes"])
            if sample_rate != ALIGNMENT_SAMPLE_RATE:
                raise ValueError("Alignment worker accepts only 16 kHz audio")
            maximum_samples = int(MAX_ALIGNMENT_SECONDS * ALIGNMENT_SAMPLE_RATE)
            if (
                sample_count < 1
                or sample_count > maximum_samples
                or audio_bytes != sample_count * 4
            ):
                raise ValueError("Invalid alignment audio frame size")
            language = str(request["language"])
            if language not in SUPPORTED_ALIGNMENT_LANGUAGES.values():
                raise ValueError("Unsupported alignment language")
            text = str(request["text"])
            if not text:
                raise ValueError("Alignment text must be non-empty")
            frame = _read_exact(sys.stdin.buffer, audio_bytes)
            audio = np.frombuffer(frame, dtype=np.float32)
            results = aligner.align(
                audio=(audio, sample_rate),
                text=text,
                language=language,
            )
            if len(results) != 1:
                raise RuntimeError("Alignment model returned an unexpected batch size")
            items = [
                {
                    "text": item.text,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                }
                for item in results[0]
            ]
            _write_response({"status": "ok", "items": items})
        except Exception as error:  # noqa: BLE001 - preserve the worker protocol boundary
            _write_response({"status": "error", "message": f"{type(error).__name__}: {error}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--serve", type=Path, metavar="MODEL_DIR")
    operation.add_argument("--acquire", type=Path, metavar="MODEL_DIR")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-downloads", action="store_true")
    mode.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.serve is not None:
        if args.allow_downloads:
            parser.error("--serve never allows downloads")
        serve(args.serve)
    else:
        acquire_model(args.acquire, allow_downloads=args.allow_downloads)


if __name__ == "__main__":
    main()
