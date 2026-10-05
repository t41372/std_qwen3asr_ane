"""Cheap fake-runtime probes for the 2026-09-22 audio/result audit."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from standard_asr import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioFormat,
    AudioPath,
    AudioStorageUri,
    AudioUrl,
    to_srt,
)
from standard_asr.contract.exceptions import StandardASRError

from std_qwen3asr_ane.errors import ModelLimitError
from std_qwen3asr_ane.plugin import create_engine, detected_language
from std_qwen3asr_ane.runtime import parse_output


class FakeRuntime:
    """Record adapter inputs without loading artifacts or Core ML."""

    max_audio_seconds = 180.0
    token_batch_size = 16

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, samples, *, language, max_new_tokens, context="", **kwargs):
        copied = np.array(samples, copy=True)
        self.calls.append(
            {
                "shape": copied.shape,
                "dtype": str(copied.dtype),
                "sample0": float(copied.flat[0]) if copied.size else None,
                "language": language,
                "max_new_tokens": max_new_tokens,
                "context": context,
            }
        )
        if copied.ndim != 1 or not copied.size or not np.isfinite(copied).all():
            raise ValueError("Expected nonempty, finite, mono 16 kHz audio")
        return SimpleNamespace(text="ok", language="English", raw_text="language English<asr_text>ok")

    def new_decoder_context(self):
        return SimpleNamespace(reset=lambda: None)

    def new_audio_context(self):
        return SimpleNamespace(reset=lambda: None)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return buffer.getvalue()


def outcome(call):
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - probe records the public boundary result
        return {
            "status": "error",
            "type": type(exc).__name__,
            "standard_error": isinstance(exc, StandardASRError),
            "message": str(exc),
            "cause": type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
        }
    return {
        "status": "ok",
        "text": result.text,
        "detected_language": result.detected_language,
        "duration": result.duration,
        "segments": result.segments,
        "words": result.words,
        "extra": result.extra,
        "diagnostics": [item.model_dump(mode="json") for item in result.diagnostics],
    }


def render_outcome(result, **kwargs):
    try:
        return {"status": "ok", "srt": to_srt(result, **kwargs)}
    except Exception as exc:  # noqa: BLE001 - renderer failure type is probe output
        return {"status": "error", "type": type(exc).__name__, "message": str(exc)}


async def stream_probe(engine) -> dict[str, object]:
    session = engine.start_transcription(
        audio_format=AudioFormat(sample_rate=16000, encoding="pcm_f32le")
    )
    async with session:
        await session.send_audio(np.zeros(8000, dtype="<f4").tobytes())
        await session.end_audio()
        events = [event async for event in session]
    result = session.result()
    return {
        "events": [
            {
                "type": event.type,
                "audio_processed_until": event.audio_processed_until,
                "extra": event.extra,
                "code": event.code,
            }
            for event in events
        ],
        "result_duration": result.duration,
        "result_has_error_field": hasattr(result, "error"),
        "result_text": result.text,
        "default_srt": render_outcome(result),
        "collapsed_srt": render_outcome(result, on_unrenderable="collapse"),
    }


def main() -> None:
    engine = create_engine(stream_chunk_seconds=0.5, stream_unfixed_chunks=99)
    runtime = FakeRuntime()
    engine._runtime = runtime
    mono = np.linspace(-0.25, 0.25, 8000, dtype=np.float32)
    encoded = wav_bytes(mono, 8000)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.wav"
        path.write_bytes(encoded)
        cases = {
            "tuple_16k": lambda: engine.transcribe((mono, 16000)),
            "bare_ndarray_strict": lambda: engine.transcribe(mono),
            "audio_array_8k_resample": lambda: engine.transcribe(AudioArray(mono, 8000)),
            "audio_bytes_decode": lambda: engine.transcribe(AudioBytes(encoded, "wav")),
            "audio_base64_decode": lambda: engine.transcribe(
                AudioBase64(base64.b64encode(encoded).decode("ascii"))
            ),
            "audio_path_decode": lambda: engine.transcribe(AudioPath(path)),
            "bare_path_decode": lambda: engine.transcribe(str(path)),
            "stereo_array": lambda: engine.transcribe(
                AudioArray(np.column_stack((mono, -mono)), 16000)
            ),
            "nonfinite_array": lambda: engine.transcribe(
                AudioArray(np.array([0.0, np.nan], dtype=np.float32), 16000)
            ),
            "empty_array": lambda: engine.transcribe(AudioArray(np.empty(0, np.float32), 16000)),
            "out_of_range_array": lambda: engine.transcribe(
                AudioArray(np.array([2.0, -2.0], dtype=np.float32), 16000)
            ),
            "url": lambda: engine.transcribe(AudioUrl("https://example.com/a.wav")),
            "storage_uri": lambda: engine.transcribe(AudioStorageUri("s3://bucket/a.wav")),
            "over_static_duration": lambda: engine.transcribe(
                AudioArray(np.zeros(30 * 16000 + 1, np.float32), 16000)
            ),
        }
        results = {name: outcome(call) for name, call in cases.items()}

    lenient = create_engine(strict=False)
    lenient_runtime = FakeRuntime()
    lenient._runtime = lenient_runtime
    results["bare_ndarray_best_effort"] = outcome(lambda: lenient.transcribe(mono))

    long_stream = create_engine(stream_max_audio_seconds=180)
    long_stream._runtime = FakeRuntime()
    results["whole_input_stream_31s"] = outcome(
        lambda: long_stream.start_transcription(
            audio=AudioArray(np.zeros(31 * 16000, np.float32), 16000)
        )
    )

    raw_text, normalized = parse_output("language Klingon<asr_text>hello", None)
    mapped, diagnostics = detected_language(normalized, None)
    results["native_unknown_language_pipeline"] = {
        "parsed_text": raw_text,
        "parser_language": normalized,
        "adapter_language": mapped,
        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
    }
    batch_result = engine.transcribe((mono, 16000))
    results["batch_render"] = render_outcome(batch_result)
    results["stream_success"] = asyncio.run(stream_probe(engine))
    for name, failure in (
        ("stream_native_failure", RuntimeError("native detail")),
        ("stream_capacity_failure", ModelLimitError("decoder KV cache exhausted")),
    ):
        failed_engine = create_engine(stream_chunk_seconds=0.5, stream_unfixed_chunks=99)
        failed_runtime = FakeRuntime()

        def fail(*args, _failure=failure, **kwargs):
            raise _failure

        failed_runtime.transcribe = fail
        failed_engine._runtime = failed_runtime
        results[name] = asyncio.run(stream_probe(failed_engine))
    results["runtime_calls"] = runtime.calls
    print(json.dumps(results, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
