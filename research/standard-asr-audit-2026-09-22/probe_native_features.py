"""Run direct, bounded Core ML checks for native runtime release changes.

This script reads only existing model and fixture artifacts. It emits one JSON
document to stdout; the caller records that output as release evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from std_qwen3asr_ane.errors import InferenceCancelled
from std_qwen3asr_ane.runtime import CoreMLRuntime

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/qwen3-asr-1.7b"
BASELINE = ROOT / "artifacts/evaluation/smoke/coreml-first.jsonl"
FIXTURES = {
    "qwen-official-en": ROOT / "artifacts/evaluation/smoke/qwen_official_en.wav",
    "qwen-official-zh": ROOT / "artifacts/evaluation/smoke/qwen_official_zh.wav",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"Expected mono fixture: {path}")
    mono = samples[:, 0]
    if sample_rate != 16000:
        factor = math.gcd(sample_rate, 16000)
        mono = resample_poly(mono, 16000 // factor, sample_rate // factor).astype(np.float32)
    return mono, {
        "path": str(path),
        "sha256": sha256(path),
        "source_sample_rate": sample_rate,
        "native_samples": len(mono),
        "native_seconds": len(mono) / 16000,
    }


def stored_baselines() -> dict[str, str]:
    result = {}
    for line in BASELINE.read_text().splitlines():
        row = json.loads(line)
        result[row["id"]] = row["hypothesis"]
    return result


def summary(result) -> dict[str, object]:
    return {
        "text": result.text,
        "language": result.language,
        "raw_model_language": result.raw_model_language,
        "raw_text": result.raw_text,
        "token_ids": list(result.token_ids),
        "timings": result.timings,
    }


def main() -> None:
    audio, inputs = zip(*(read_audio(path) for path in FIXTURES.values()), strict=True)
    english, chinese = audio
    recorded_inputs = dict(zip(FIXTURES, inputs, strict=True))
    previous = stored_baselines()
    runtime = CoreMLRuntime(BUNDLE)
    try:
        baseline_en = runtime.transcribe(english, language=None, max_new_tokens=256)
        baseline_zh = runtime.transcribe(chinese, language=None, max_new_tokens=256)
        refined_en = runtime.transcribe(english, language="en-US", max_new_tokens=256)
        candidate_en = runtime.transcribe(
            english,
            language=None,
            max_new_tokens=256,
            candidate_language_names=["English"],
        )
        hinted_en = runtime.transcribe(
            english,
            language=None,
            max_new_tokens=256,
            phrase_hints=["music"],
        )

        cancelled = Event()
        decoder_context = runtime.new_decoder_context()
        audio_context = runtime.new_audio_context()
        original_predict = runtime.lm_head.predict
        head_calls = 0

        def cancel_after_head(data, *, state=None):
            nonlocal head_calls
            output = original_predict(data, state=state) if state is not None else original_predict(data)
            head_calls += 1
            cancelled.set()
            return output

        runtime.lm_head.predict = cancel_after_head
        try:
            runtime.transcribe(
                english,
                language=None,
                max_new_tokens=256,
                decoder_context=decoder_context,
                audio_context=audio_context,
                cancel=cancelled,
            )
        except InferenceCancelled:
            cancellation = {
                "outcome": "InferenceCancelled",
                "head_predictions_before_stop": head_calls,
                "decoder_context_requires_fresh_states": decoder_context._needs_fresh_states,
                "audio_context_timings_after_stop": audio_context.timings,
            }
        else:
            raise AssertionError("Cancellation token did not stop after the active LM-head prediction")
        finally:
            runtime.lm_head.predict = original_predict

        retry_en = runtime.transcribe(
            english,
            language=None,
            max_new_tokens=256,
            decoder_context=decoder_context,
            audio_context=audio_context,
        )
        cancellation["retry_matches_unguided_baseline"] = retry_en.text == baseline_en.text
        cancellation["retry"] = summary(retry_en)

        print(
            json.dumps(
                {
                    "command": "./.venv/bin/python research/standard-asr-audit-2026-09-22/probe_native_features.py",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "scope": "direct CoreMLRuntime smoke only; not a corpus-quality result",
                    "bundle": {
                        "path": str(BUNDLE),
                        "manifest_sha256": sha256(BUNDLE / "manifest.json"),
                        "manifest": runtime.manifest,
                        "compute_units": runtime.compute_units,
                    },
                    "inputs": recorded_inputs,
                    "historical_baseline_text_match": {
                        "qwen-official-en": baseline_en.text == previous["qwen-official-en"],
                        "qwen-official-zh": baseline_zh.text == previous["qwen-official-zh"],
                    },
                    "runs": {
                        "unguided_auto_en": summary(baseline_en),
                        "unguided_auto_zh": summary(baseline_zh),
                        "forced_refinement_en_US": summary(refined_en),
                        "candidate_English_auto_en": summary(candidate_en),
                        "phrase_music_auto_en": summary(hinted_en),
                    },
                    "cancellation": cancellation,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
