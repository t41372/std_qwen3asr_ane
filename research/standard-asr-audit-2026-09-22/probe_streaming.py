"""Small public-API reproductions for the streaming audit.

Run from the repository root:
    .venv/bin/python research/standard-asr-audit-2026-09-22/probe_streaming.py

The fake runtime replaces only native inference.  Standard ASR's real public
EngineBase/session/reducer paths and the plugin's Qwen3ASRSession run unchanged.
No model artifact, Core ML load, or network access is required.
"""

from __future__ import annotations

import asyncio
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
from standard_asr.contract.exceptions import TranscriptionError
from standard_asr.engine import AudioFormat, RuntimeParams

from std_qwen3asr_ane.plugin import create_engine

FORMAT = AudioFormat(sample_rate=16000, encoding="pcm_f32le")
WIRE = np.zeros(8000, dtype="<f4").tobytes()


class _Context:
    def reset(self) -> None:
        pass


class RecordingRuntime:
    """The narrow native interface Qwen3ASRSession exercises for one decode."""

    max_audio_seconds = 30.0

    def __init__(
        self,
        *,
        language: str = "English",
        failure: Exception | None = None,
        require_base_language: bool = False,
    ) -> None:
        self.language = language
        self.failure = failure
        self.require_base_language = require_base_language
        self.calls: list[dict[str, object]] = []

    def new_decoder_context(self) -> _Context:
        return _Context()

    def new_audio_context(self) -> _Context:
        return _Context()

    def transcribe(self, samples: np.ndarray, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"samples": len(samples), **kwargs})
        if self.failure is not None:
            raise self.failure
        if self.require_base_language and kwargs["language"] not in {None, "en", "zh"}:
            raise ValueError("native language control requires a base Qwen language key")
        return SimpleNamespace(text="hello", raw_text="hello", language=self.language)


class BlockingRuntime(RecordingRuntime):
    """Shows that a cancelled session's active native work still owns the engine lock."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.entries = 0

    def transcribe(self, samples: np.ndarray, **kwargs: object) -> SimpleNamespace:
        self.entries += 1
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError("probe did not release native work")
        return super().transcribe(samples, **kwargs)


async def _record(engine, params: RuntimeParams) -> tuple[list[dict[str, object]], object]:
    session = engine.start_transcription(audio_format=FORMAT, params=params)
    async with session:
        session.feed([WIRE])
        events = [event async for event in session]
    return [event.model_dump(mode="json") for event in events], session


async def _main() -> dict[str, object]:
    refinement = create_engine(stream_chunk_seconds=0.5)
    refinement_runtime = RecordingRuntime(require_base_language=True)
    refinement._runtime = refinement_runtime
    refinement_events, refinement_session = await _record(
        refinement, RuntimeParams(language="en-US")
    )

    batch_refinement = create_engine()
    batch_runtime = RecordingRuntime(require_base_language=True)
    batch_refinement._runtime = batch_runtime
    try:
        batch_refinement.transcribe(
            (np.zeros(8000, dtype=np.float32), 16000), RuntimeParams(language="en-US")
        )
    except TranscriptionError as exc:  # Expected: the public batch boundary wraps native failure.
        batch_error = {
            "type": type(exc).__name__,
            "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        }
    else:  # pragma: no cover - documents the expected current behavior.
        batch_error = None

    unmapped = create_engine(stream_chunk_seconds=0.5)
    unmapped._runtime = RecordingRuntime(language="Klingon")
    unmapped_events, unmapped_session = await _record(unmapped, RuntimeParams(language="auto"))

    native_value_error = create_engine(stream_chunk_seconds=0.5)
    native_value_error._runtime = RecordingRuntime(failure=ValueError("native configuration failure"))
    value_events, value_session = await _record(native_value_error, RuntimeParams())

    cancellation = create_engine(stream_chunk_seconds=0.5)
    cancellation_runtime = BlockingRuntime()
    cancellation._runtime = cancellation_runtime
    first = cancellation.start_transcription(audio_format=FORMAT)
    async with first:
        consumer = asyncio.create_task(_collect(first))
        await first.send_audio(WIRE)
        while not cancellation_runtime.entered.is_set():
            await asyncio.sleep(0.001)
        await first.cancel()
        cancelled_events = [event.model_dump(mode="json") for event in await consumer]

    second = cancellation.start_transcription(audio_format=FORMAT)
    async with second:
        second_consumer = asyncio.create_task(_collect(second))
        await second.send_audio(WIRE)
        await asyncio.sleep(0.03)
        second_entered_before_release = cancellation_runtime.entries > 1
        cancellation_runtime.release.set()
        await second.end_audio()
        second_events = [event.model_dump(mode="json") for event in await second_consumer]

    return {
        "refinement": {
            "runtime_language": refinement_runtime.calls[0]["language"],
            "terminal": refinement_events[-1],
            "session_diagnostics": [
                diagnostic.model_dump(mode="json")
                for diagnostic in refinement_session.diagnostics()
            ],
            "reduced_result": refinement_session.result().model_dump(mode="json"),
        },
        "unmapped_auto_language": {
            "events": unmapped_events,
            "session_diagnostics": [
                diagnostic.model_dump(mode="json")
                for diagnostic in unmapped_session.diagnostics()
            ],
            "reduced_result": unmapped_session.result().model_dump(mode="json"),
        },
        "batch_refinement": {
            "runtime_language": batch_runtime.calls[0]["language"],
            "exception": batch_error,
        },
        "native_value_error": {
            "terminal": value_events[-1],
            "session_diagnostics": [
                diagnostic.model_dump(mode="json")
                for diagnostic in value_session.diagnostics()
            ],
            "reduced_result": value_session.result().model_dump(mode="json"),
        },
        "cancelled_native_work": {
            "first_terminal": cancelled_events[-1],
            "second_native_entry_before_release": second_entered_before_release,
            "second_terminal": second_events[-1],
        },
    }


async def _collect(session: object) -> list[object]:
    return [event async for event in session]  # type: ignore[attr-defined]


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_main()), indent=2, ensure_ascii=False))
