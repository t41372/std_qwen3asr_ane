"""Bounded Qwen prefix-rollback streaming with exact decoder-prefix state reuse.

This follows the upstream streaming strategy, not a causal audio encoder cache.
Every partial is revisable; only an end-of-input closed event is immutable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from threading import Event
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from standard_asr.engine import (
    AudioFormat,
    PreparedAudio,
    RuntimeParams,
    TranscriptionEvent,
    TranscriptionSession,
)

from .audio import SAMPLE_RATE
from .errors import ModelLimitError
from .languages import classify_model_language
from .runtime import rollback_prefix

if TYPE_CHECKING:
    from .plugin import Qwen3ASREngine

_T = TypeVar("_T")


class _Cancelled(Exception):
    pass


class _AudioLimit(Exception):
    def __init__(self, seconds: float, source: str):
        self.seconds = seconds
        self.source = source


class Qwen3ASRSession(TranscriptionSession):
    """One bounded utterance; end_audio()/finish() flush, cancel() terminates.

    Callers may use the inherited async context or Standard ASR's SyncSession.
    Native Core ML calls cannot be interrupted mid-prediction. Cancellation stops
    delivery promptly; the shared engine lock remains held until native work ends.
    """

    def __init__(
        self,
        engine: Qwen3ASREngine,
        params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ):
        super().__init__(
            audio_queue_maxsize=engine.config.stream_audio_queue_size, strict_lifecycle=True
        )
        self.engine = engine
        self.params = params
        self.audio_format = audio_format or AudioFormat(
            sample_rate=SAMPLE_RATE, encoding="pcm_s16le"
        )
        self.prepared_audio = prepared_audio
        self._cancel_requested = asyncio.Event()
        self._native_cancelled = Event()
        self._sample_limit = int(engine.config.stream_max_audio_seconds * SAMPLE_RATE)
        self._chunk_samples = max(1, round(engine.config.stream_chunk_seconds * SAMPLE_RATE))
        self._received_samples = 0
        self._processed_samples = 0
        self._decode_count = 0
        self._last_raw = ""
        self._last_result = None
        self._pcm_tail = b""
        self._decoder_context = None

    async def finish(self) -> None:
        """Flush the final partial audio chunk, then close the utterance."""
        await self.end_audio()

    async def cancel(self) -> None:
        """Request a terminal cancellation without waiting for native inference."""
        self._native_cancelled.set()
        self._cancel_requested.set()

    async def _close(self) -> None:
        self._native_cancelled.set()
        self._cancel_requested.set()
        # An in-flight native worker keeps its own context reference until it
        # finishes. Do not reset state from the async thread while it is used.
        self._decoder_context = None

    async def _until_cancelled(self, operation: Awaitable[_T]) -> _T:
        task = asyncio.ensure_future(operation)
        cancellation = asyncio.create_task(self._cancel_requested.wait())
        try:
            await asyncio.wait((task, cancellation), return_when=asyncio.FIRST_COMPLETED)
            if self._cancel_requested.is_set():
                raise _Cancelled
            return await task
        finally:
            for pending in (task, cancellation):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, cancellation, return_exceptions=True)

    def _recognize(self, samples: np.ndarray):
        with self.engine._inference_lock:
            if self._native_cancelled.is_set():
                raise _Cancelled
            runtime = self.engine._ensure_model_loaded()
            limit = min(self.engine.config.stream_max_audio_seconds, runtime.max_audio_seconds)
            if samples.size > int(limit * SAMPLE_RATE):
                source = (
                    "bundle_audio_limit"
                    if runtime.max_audio_seconds < self.engine.config.stream_max_audio_seconds
                    else "configured_session_limit"
                )
                raise _AudioLimit(limit, source)
            if self._decoder_context is None:
                self._decoder_context = runtime.new_decoder_context()
            decoder_context = self._decoder_context
            prefix = ""
            if self._decode_count >= self.engine.config.stream_unfixed_chunks:
                prefix = rollback_prefix(
                    runtime.tokenizer, self._last_raw, self.engine.config.stream_unfixed_tokens
                )
            try:
                return runtime.transcribe(
                    samples,
                    language=None if self.params.language == "auto" else self.params.language,
                    max_new_tokens=self.engine.config.max_new_tokens,
                    context=self.params.prompt or "",
                    prefix_text=prefix,
                    decoder_context=decoder_context,
                )
            except Exception:
                decoder_context.reset()
                raise

    async def _decode(self, samples: np.ndarray) -> TranscriptionEvent:
        result = await self._until_cancelled(asyncio.to_thread(self._recognize, samples.copy()))
        self._last_result = result
        self._last_raw = result.raw_text
        self._processed_samples = len(samples)
        self._decode_count += 1
        detected, extra = self._language_fields(result.language)
        return TranscriptionEvent.partial(
            "utterance-0",
            result.text,
            stable_until=0,
            detected_language=detected,
            extra={"audio_prefix_seconds": self._processed_samples / SAMPLE_RATE, **extra},
        )

    def _language_fields(self, model_language: str | None) -> tuple[str | None, dict]:
        """Map the model's language line; disclose names outside the published list."""
        requested = None if self.params.language == "auto" else self.params.language
        detected, unmapped = classify_model_language(model_language, requested)
        extra = {} if unmapped is None else {"unmapped_model_language": unmapped}
        return detected, extra

    async def _input_samples(self) -> AsyncIterator[np.ndarray]:
        if self.prepared_audio is not None:
            if self.prepared_audio.array is None:
                raise ValueError("Whole-input streaming requires prepared array audio")
            yield self.prepared_audio.array
            return
        dtype = "<i2" if self.audio_format.encoding == "pcm_s16le" else "<f4"
        sample_bytes = np.dtype(dtype).itemsize
        chunks = self.audio_chunks().__aiter__()
        while True:
            try:
                chunk = await self._until_cancelled(anext(chunks))
            except StopAsyncIteration:
                break
            wire = self._pcm_tail + chunk
            complete_bytes = len(wire) // sample_bytes * sample_bytes
            self._pcm_tail = wire[complete_bytes:]
            samples = np.frombuffer(wire[:complete_bytes], dtype=dtype).astype(np.float32)
            if dtype == "<i2":
                samples /= 32768.0
            yield samples
        if self._pcm_tail:
            raise ValueError("PCM input ended in an incomplete sample")

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        accumulated = np.empty(min(self._sample_limit, self._chunk_samples), dtype=np.float32)
        next_decode = self._chunk_samples
        try:
            async for samples in self._input_samples():
                if self._cancel_requested.is_set():
                    raise _Cancelled
                if samples.ndim != 1 or not np.isfinite(samples).all():
                    raise ValueError("Streaming audio must contain finite mono samples")
                received = self._received_samples + len(samples)
                if received > self._sample_limit:
                    self._received_samples = received
                    raise _AudioLimit(self._sample_limit / SAMPLE_RATE, "configured_session_limit")
                if received > accumulated.size:
                    capacity = min(self._sample_limit, max(received, accumulated.size * 2))
                    grown = np.empty(capacity, dtype=np.float32)
                    grown[: self._received_samples] = accumulated[: self._received_samples]
                    accumulated = grown
                accumulated[self._received_samples : received] = samples
                self._received_samples = received
                while next_decode <= received:
                    yield await self._decode(accumulated[:next_decode])
                    next_decode += self._chunk_samples
            if self._received_samples > self._processed_samples:
                yield await self._decode(accumulated[: self._received_samples])
            if self._last_result is not None:
                detected, extra = self._language_fields(self._last_result.language)
                yield TranscriptionEvent.closed(
                    "utterance-0",
                    self._last_result.text,
                    extra={"audio_prefix_seconds": self._processed_samples / SAMPLE_RATE, **extra},
                    detected_language=detected,
                )
            # The base emits done and reduces the recorded events.
        except _Cancelled:
            yield TranscriptionEvent.make_error(
                "cancelled", extra={"message": "Recognition cancelled."}
            )
        except _AudioLimit as exc:
            yield TranscriptionEvent.make_error(
                "audio_limit_exceeded",
                extra={
                    "max_audio_seconds": exc.seconds,
                    "limit_source": exc.source,
                    "received_audio_seconds": self._received_samples / SAMPLE_RATE,
                    "processed_audio_seconds": self._processed_samples / SAMPLE_RATE,
                    "message": (
                        "Audio exceeds the configured session duration."
                        if exc.source == "configured_session_limit"
                        else "Audio exceeds the loaded bundle's declared duration; a larger-context bundle is required."
                    ),
                },
            )
        except ModelLimitError as exc:
            # A fixed capacity of the loaded bundle, not a fault in the caller's audio.
            yield TranscriptionEvent.make_error(
                "bundle_capacity_exceeded",
                extra={
                    "message": str(exc),
                    "received_audio_seconds": self._received_samples / SAMPLE_RATE,
                    "processed_audio_seconds": self._processed_samples / SAMPLE_RATE,
                },
            )
        except ValueError as exc:
            yield TranscriptionEvent.make_error(
                "invalid_audio_or_context", extra={"message": str(exc)}
            )
