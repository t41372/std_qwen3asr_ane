"""Measure actual Core ML independent-request packing against serial decoding.

Run from the repository root with the inference environment available.  The
script uses the target-bound compact verifier inside the existing draft bundle,
but does not import or load MLX or the draft language model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import gcd
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from std_qwen3asr_ane.batching import OfflineRecognitionRequest
from std_qwen3asr_ane.runtime import CoreMLRuntime


def load_audio(path: Path) -> np.ndarray:
    """Load a mono fixture and deterministically prepare the native 16 kHz input."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim != 1:
        raise ValueError(f"{path} is not mono")
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor).astype(
            np.float32, copy=False
        )
    return np.ascontiguousarray(samples)


def runtime_result(result) -> dict:
    decoder_seconds = result.timings["generation_seconds"]
    if "prefill_seconds" in result.timings:
        decoder_seconds += result.timings["prefill_seconds"]
    return {
        "text": result.text,
        "language": result.language,
        "raw_text": result.raw_text,
        "token_ids": list(result.token_ids),
        "audio_tokens": result.audio_tokens,
        "generated_tokens": result.timings["generated_tokens"],
        "eos_token_id": result.timings["eos_token_id"],
        "decoder_and_head_seconds": decoder_seconds,
        "total_seconds": result.timings["total_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/qwen3-asr-1.7b"))
    parser.add_argument("--draft", type=Path, default=Path("artifacts/qwen3-asr-1.7b-draft"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/release-readiness/batching-native.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    fixtures = (
        ("english", Path("artifacts/evaluation/smoke/qwen_official_en.wav"), "en"),
        ("chinese", Path("artifacts/evaluation/smoke/qwen_official_zh.wav"), "zh"),
        (
            "heldout_english",
            Path("artifacts/evaluation/librispeech-balanced-100/audio/000128.flac"),
            "en",
        ),
    )
    samples = [(name, load_audio(path), language) for name, path, language in fixtures]
    runtime = CoreMLRuntime(args.model)
    head = runtime.load_batch_head_from_draft(args.draft)
    try:
        # A serial run owns the ordinary full-logits head.  It is the oracle;
        # compact-head results must match every token and EOS-derived endpoint.
        serial_started = perf_counter()
        serial = [
            runtime.transcribe(audio, language=language, max_new_tokens=args.max_new_tokens)
            for _, audio, language in samples
        ]
        serial_elapsed = perf_counter() - serial_started
        packed_started = perf_counter()
        outcomes = runtime.transcribe_many(
            tuple(
                OfflineRecognitionRequest(audio, language, args.max_new_tokens)
                for _, audio, language in samples
            ),
            batch_head=head,
        )
        packed_elapsed = perf_counter() - packed_started
        if any(outcome.error is not None for outcome in outcomes):
            raise RuntimeError(f"Packed recognition failed: {[outcome.error for outcome in outcomes]}")
        if any(outcome.execution != "packed" for outcome in outcomes):
            raise RuntimeError(f"Expected packed outcomes, got {[outcome.execution for outcome in outcomes]}")
        packed = [outcome.result for outcome in outcomes]
        for (name, _, _), expected, actual in zip(samples, serial, packed, strict=True):
            if (
                actual.token_ids != expected.token_ids
                or actual.raw_text != expected.raw_text
                or actual.text != expected.text
                or actual.language != expected.language
            ):
                raise AssertionError(f"{name} packed transcript/token/EOS parity failed")
        stats = outcomes[0].stats
        assert stats is not None
        serial_decoder_seconds = sum(
            result.timings["prefill_seconds"] + result.timings["generation_seconds"]
            for result in serial
        )
        record = {
            "target": {
                "path": str(runtime.model_dir),
                "model_id": runtime.manifest["model_id"],
                "token_batch_size": runtime.token_batch_size,
                "decoder_partitions": runtime.manifest["decoder_partitions"],
                "compact_head_source": str(args.draft.resolve() / "verify_head.mlmodelc"),
            },
            "method": {
                "decoder": "One T16 Core ML decoder state per group with disjoint cache ranges, per-row RoPE positions, block-diagonal masks and distinct update slots.",
                "head": "The existing target-bound draft verify head scores compact rows; no draft model or ThreadPool is used.",
                "audio": "Audio frontend and encoder remain per-request serial work and are excluded from decoder speedup claims.",
            },
            "fixtures": [
                {
                    "name": name,
                    "language": language,
                    "samples": int(audio.size),
                    "serial": runtime_result(expected),
                    "packed": runtime_result(actual),
                    "token_and_eos_parity": True,
                }
                for (name, audio, language), expected, actual in zip(samples, serial, packed, strict=True)
            ],
            "packed_group": asdict(stats),
            "measurements": {
                "serial_end_to_end_seconds": serial_elapsed,
                "packed_end_to_end_seconds": packed_elapsed,
                "serial_decoder_and_head_seconds": serial_decoder_seconds,
                "packed_decoder_and_head_seconds": stats.elapsed_seconds,
                "decoder_and_head_speedup": serial_decoder_seconds / stats.elapsed_seconds,
                "end_to_end_speedup": serial_elapsed / packed_elapsed,
            },
            "claim": "Measured decoder/head call consolidation for this three-input group only. It does not claim frontend or encoder batching.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        head.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
