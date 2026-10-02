"""Verify real packed decoding parity without making a performance claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from std_qwen3asr_ane.batching import OfflineRecognitionRequest
from std_qwen3asr_ane.runtime import CoreMLRuntime

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/release-readiness/standalone-batch-head-2026-10-02-5f6eef25/model"
TARGET = ARTIFACT_ROOT / "qwen3-asr-1.7b"
HEAD = ARTIFACT_ROOT / "qwen3-asr-1.7b-batch-head"
EVIDENCE = ROOT / "research/release-readiness/packed-parity-2026-10-02.json"
FIXTURES = (
    ("english", ROOT / "artifacts/evaluation/smoke/qwen_official_en.wav", "en"),
    ("chinese", ROOT / "artifacts/evaluation/smoke/qwen_official_zh.wav", "zh"),
    (
        "heldout_english",
        ROOT / "artifacts/evaluation/librispeech-balanced-100/audio/000128.flac",
        "en",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono audio: {path}")
    if sample_rate != 16_000:
        divisor = gcd(sample_rate, 16_000)
        samples = resample_poly(samples, 16_000 // divisor, sample_rate // divisor).astype(
            np.float32, copy=False
        )
    return np.ascontiguousarray(samples)


def result_evidence(result: Any) -> dict[str, Any]:
    return {
        "text": result.text,
        "raw_text": result.raw_text,
        "language": result.language,
        "token_ids": list(result.token_ids),
        "eos_token_id": result.timings["eos_token_id"],
        "generated_tokens": result.timings["generated_tokens"],
        "timings": result.timings,
    }


def write_evidence(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def verify(max_new_tokens: int) -> dict[str, Any]:
    inputs = [(name, path, language, load_audio(path)) for name, path, language in FIXTURES]
    runtime = CoreMLRuntime(TARGET)
    head = runtime.load_batch_head(HEAD)
    try:
        serial = [
            runtime.transcribe(samples, language=language, max_new_tokens=max_new_tokens)
            for _, _, language, samples in inputs
        ]
        outcomes = runtime.transcribe_many(
            tuple(
                OfflineRecognitionRequest(samples, language, max_new_tokens)
                for _, _, language, samples in inputs
            ),
            batch_head=head,
        )
        if any(outcome.error is not None for outcome in outcomes):
            raise RuntimeError(
                f"Packed recognition failed: {[repr(outcome.error) for outcome in outcomes]}"
            )
        if any(outcome.execution != "packed" for outcome in outcomes):
            raise RuntimeError(
                f"Expected packed execution: {[outcome.execution for outcome in outcomes]}"
            )
        packed = [outcome.result for outcome in outcomes]
        if any(result is None for result in packed):
            raise RuntimeError("Packed recognition returned a missing result")

        rows = []
        for (name, path, language, samples), expected, actual in zip(
            inputs, serial, packed, strict=True
        ):
            assert actual is not None
            equality = {
                "text_equal": actual.text == expected.text,
                "raw_text_equal": actual.raw_text == expected.raw_text,
                "language_equal": actual.language == expected.language,
                "token_ids_equal": actual.token_ids == expected.token_ids,
                "eos_equal": actual.timings["eos_token_id"] == expected.timings["eos_token_id"],
            }
            packed_keys = set(actual.timings)
            telemetry_shape = {
                "has_packed_item_preparation_seconds": "packed_item_preparation_seconds"
                in packed_keys,
                "has_packed_group_elapsed_seconds": "packed_group_elapsed_seconds" in packed_keys,
                "has_packed_group_decoder_calls": "packed_group_decoder_calls" in packed_keys,
                "has_packed_group_head_calls": "packed_group_head_calls" in packed_keys,
                "has_packed_group_lane_count": "packed_group_lane_count" in packed_keys,
                "has_packed_group_reserved_cache_slots": "packed_group_reserved_cache_slots"
                in packed_keys,
                "omits_unverifiable_prefill_calls": "prefill_calls" not in packed_keys,
                "omits_unverifiable_generation_seconds": "generation_seconds" not in packed_keys,
                "omits_unverifiable_total_seconds": "total_seconds" not in packed_keys,
            }
            rows.append(
                {
                    "name": name,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "language": language,
                    "samples": int(samples.size),
                    "serial": result_evidence(expected),
                    "packed": result_evidence(actual),
                    "exact_parity": equality,
                    "telemetry_shape": telemetry_shape,
                    "passed": all(equality.values()) and all(telemetry_shape.values()),
                }
            )

        stats = outcomes[0].stats
        if stats is None or any(outcome.stats != stats for outcome in outcomes):
            raise RuntimeError("Packed outcomes do not share one truthful group measurement")
        passed = all(row["passed"] for row in rows)
        return {
            "schema_version": 1,
            "date": "2026-10-02",
            "status": "passed" if passed else "failed",
            "purpose": "Exact serial-versus-packed native parity and telemetry-shape validation.",
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "standard_asr_commit": "5f6eef25e35e5e66e9010474e6dee531021e61f1",
                "pythonpath_override": False,
            },
            "artifacts": {
                "target": str(TARGET.relative_to(ROOT)),
                "target_manifest_sha256": sha256(TARGET / "manifest.json"),
                "batch_head": str(HEAD.relative_to(ROOT)),
                "batch_head_manifest_sha256": sha256(HEAD / "manifest.json"),
            },
            "source_sha256": {
                "runtime": sha256(ROOT / "std_qwen3asr_ane/src/std_qwen3asr_ane/runtime.py"),
                "batching": sha256(ROOT / "std_qwen3asr_ane/src/std_qwen3asr_ane/batching.py"),
                "verifier": sha256(Path(__file__)),
            },
            "max_new_tokens": max_new_tokens,
            "rows": rows,
            "packed_group": asdict(stats),
            "performance_claim": False,
            "timing_interpretation": (
                "Elapsed fields are retained as honest runtime telemetry only; concurrent host "
                "work was uncontrolled, so no latency, throughput, or speedup claim is made."
            ),
            "failures": [] if passed else ["one or more parity or telemetry gates failed"],
        }
    finally:
        head.close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVIDENCE)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        document = verify(args.max_new_tokens)
    except Exception as error:
        document = {
            "schema_version": 1,
            "date": "2026-10-02",
            "status": "failed",
            "source_sha256": {"verifier": sha256(Path(__file__))},
            "performance_claim": False,
            "failures": [{"type": type(error).__name__, "message": str(error)}],
        }
        write_evidence(document, args.output)
        raise
    write_evidence(document, args.output)
    print(json.dumps({"status": document["status"], "output": str(args.output)}))
    if document["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
