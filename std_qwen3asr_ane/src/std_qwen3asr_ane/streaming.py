"""Long-recording Qwen streaming with bounded native windows and exact prefix reuse.

This follows the upstream streaming strategy, with exact stateless audio-graph
reuse rather than a causal audio encoder state.
Every partial is revisable. Each bounded native window receives one closed
segment before the next segment id begins, without fabricating speech spans.
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
from .decoding_guidance import GuidanceRequestError
from .errors import InferenceCancelled, ModelLimitError
from .languages import classify_model_language
from .longform import AudioSpan, LongFormAudioLimit, LongFormCoordinator
from .runtime import rollback_prefix

if TYPE_CHECKING:
    from .plugin import Qwen3ASREngine

_T = TypeVar("_T")


class _Cancelled(Exception):
    pass


class _InvalidPCM(ValueError):
    """Caller-owned incremental PCM framing or value failure."""


class Qwen3ASRSession(TranscriptionSession):
    """One long recording composed of bounded, non-overlapping native windows.

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
        configured_limits = [
            engine.config.max_recording_seconds,
            engine.config.stream_max_audio_seconds,
        ]
        finite_limits = [value for value in configured_limits if value is not None]
        configured_guard = min(finite_limits) if finite_limits else None
        self._sample_limit = (
            None if configured_guard is None else int(configured_guard * SAMPLE_RATE)
        )
        self._chunk_samples = max(1, round(engine.config.stream_chunk_seconds * SAMPLE_RATE))
        self._coordinator: LongFormCoordinator | None = None
        self._received_samples = 0
        self._processed_samples = 0
        self._decode_count = 0
        self._last_raw = ""
        self._last_result = None
        self._active_start_sample: int | None = None
        self._active_processed_end_sample: int | None = None
        self._next_partial_end_sample = self._chunk_samples
        self._segment_index = 0
        self._unmapped_language_noted = False
        self._finalizer_diagnostic_codes: set[str] = set()
        self._speaker_tracker = None
        self._request_artifacts_checked = False
        self._audio_progress_supported = engine.supports("streaming.audio_progress")
        self._pcm_tail = b""
        self._decoder_context = None
        self._audio_context = None

    async def finish(self) -> None:
        """Flush the final partial audio chunk, then close the recording."""
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
        self._audio_context = None

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

    def _recognize(self, span: AudioSpan):
        if self._native_cancelled.is_set():
            raise _Cancelled
        return self._recognize_via_engine(span)

    def _recognize_via_engine(self, span: AudioSpan):
        """Run a bounded native window through the engine's serialized hook."""
        try:
            if self._decoder_context is None or self._audio_context is None:
                decoder_context, audio_context = self.engine._new_stream_contexts(
                    cancel=self._native_cancelled
                )
                if self._decoder_context is None:
                    self._decoder_context = decoder_context
                if self._audio_context is None:
                    self._audio_context = audio_context
            prefix = ""
            if self._decode_count >= self.engine.config.stream_unfixed_chunks:
                runtime = self.engine._runtime
                assert runtime is not None
                prefix = rollback_prefix(
                    runtime.tokenizer, self._last_raw, self.engine.config.stream_unfixed_tokens
                )
            return self.engine._recognize_chunk(
                span.samples,
                self.params,
                prefix_text=prefix,
                decoder_context=self._decoder_context,
                audio_context=self._audio_context,
                cancel=self._native_cancelled,
            )
        except InferenceCancelled as exc:
            self._reset_active_context()
            raise _Cancelled from exc
        except Exception:
            self._reset_active_context()
            raise

    async def _decode(self, span: AudioSpan) -> TranscriptionEvent:
        result = await self._until_cancelled(asyncio.to_thread(self._recognize, span))
        self._last_result = result
        self._last_raw = result.raw_text
        self._active_start_sample = span.start_sample
        self._active_processed_end_sample = span.end_sample
        # A final low-energy boundary can precede a provisional decode. The
        # processing frontier tracks audio already examined, not the final span.
        self._processed_samples = max(self._processed_samples, span.end_sample)
        self._decode_count += 1
        detected, extra = self._language_fields(result)
        return TranscriptionEvent.partial(
            self._segment_id,
            result.text,
            stable_until=0,
            detected_language=detected,
            audio_processed_until=(
                self._processed_samples / SAMPLE_RATE if self._audio_progress_supported else None
            ),
            extra=self._event_extra(span, extra),
        )

    @property
    def _segment_id(self) -> str:
        return f"utterance-{self._segment_index}"

    def _language_fields(self, result) -> tuple[str | None, dict]:
        """Map the model's language line; disclose names outside the published list."""
        requested = None if self.params.language == "auto" else self.params.language
        raw = result.raw_model_language
        model_language = raw if raw is not None else result.language
        if isinstance(raw, str) and raw.casefold() == "none":
            model_language = None
        detected, unmapped = classify_model_language(model_language, requested)
        extra = {} if unmapped is None else {"unmapped_model_language": unmapped}
        if unmapped is not None and not self._unmapped_language_noted:
            self.emit_diagnostic(
                code="detected_language_unmapped",
                message="The model reported a language name outside its published language list.",
                param="language",
                provided=unmapped,
                effective=None,
            )
            self._unmapped_language_noted = True
        return detected, extra

    def _event_extra(self, span: AudioSpan, language_extra: dict) -> dict:
        extra = {
            "audio_prefix_seconds": span.end_sample / SAMPLE_RATE,
            "input_start_seconds": span.start_sample / SAMPLE_RATE,
            "input_end_seconds": span.end_sample / SAMPLE_RATE,
            **language_extra,
        }
        if span.boundary_energy is not None:
            extra["window_boundary_energy"] = span.boundary_energy
        return extra

    async def _input_samples(self) -> AsyncIterator[np.ndarray]:
        if self.prepared_audio is not None:
            if self.prepared_audio.array is None:
                raise _InvalidPCM("Whole-input streaming requires prepared array audio")
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
            raise _InvalidPCM("PCM input ended in an incomplete sample")

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        try:
            coordinator = await self._initialize_longform()
            async for samples in self._input_samples():
                if self._cancel_requested.is_set():
                    raise _Cancelled
                if samples.ndim != 1 or not np.isfinite(samples).all():
                    raise _InvalidPCM("Streaming audio must contain finite mono samples")
                for span in coordinator.append(samples):
                    self._received_samples = coordinator.received_samples
                    async for event in self._emit_partials_through(span):
                        yield event
                    yield await self._close_span(span)
                self._received_samples = coordinator.received_samples
                active = coordinator.active_span()
                if active is not None:
                    async for event in self._emit_partials_through(active):
                        yield event
            tail = coordinator.finish()
            self._received_samples = coordinator.received_samples
            if tail is not None:
                async for event in self._emit_partials_through(tail):
                    yield event
                yield await self._close_span(tail)
            self._set_input_duration(self._received_samples / SAMPLE_RATE)
            # The base emits done and reduces the recorded events.
        except _Cancelled:
            yield TranscriptionEvent.make_error(
                "cancelled", extra={"message": "Recognition cancelled."}
            )
        except LongFormAudioLimit as exc:
            self._received_samples = exc.received_samples
            yield TranscriptionEvent.make_error(
                "audio_limit_exceeded",
                extra={
                    "max_audio_seconds": exc.max_samples / SAMPLE_RATE,
                    "limit_source": "configured_recording_guard",
                    "received_audio_seconds": exc.received_samples / SAMPLE_RATE,
                    "processed_audio_seconds": self._processed_samples / SAMPLE_RATE,
                    "message": "Audio exceeds the configured recording duration guard.",
                },
            )
        except GuidanceRequestError as exc:
            yield TranscriptionEvent.make_error("invalid_guidance", extra={"message": str(exc)})
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
        except _InvalidPCM as exc:
            yield TranscriptionEvent.make_error(
                "invalid_audio_or_context", extra={"message": str(exc)}
            )

    async def _initialize_longform(self) -> LongFormCoordinator:
        """Resolve the validated native window before retaining any stream PCM."""
        if self._coordinator is None:
            await self._until_cancelled(asyncio.to_thread(self._prepare_request_artifacts))
            seconds = await self._until_cancelled(
                asyncio.to_thread(self.engine._stream_window_seconds, cancel=self._native_cancelled)
            )
            self._coordinator = LongFormCoordinator(
                sample_rate=SAMPLE_RATE,
                native_window_samples=int(seconds * SAMPLE_RATE),
                max_total_samples=self._sample_limit,
            )
        return self._coordinator

    def _prepare_request_artifacts(self) -> None:
        """Check request-specific auxiliary artifacts before loading native inference."""
        if self._request_artifacts_checked:
            return
        self.engine._require_request_artifacts(self.params, mode="streaming")
        self._speaker_tracker = self.engine._new_speaker_tracker(self.params)
        self._request_artifacts_checked = True

    async def _emit_partials_through(self, span: AudioSpan) -> AsyncIterator[TranscriptionEvent]:
        """Emit revisable snapshots at the configured cadence within one segment."""
        if self._active_start_sample != span.start_sample:
            self._reset_active_context()
            self._active_start_sample = span.start_sample
            self._next_partial_end_sample = span.start_sample + self._chunk_samples
        while self._next_partial_end_sample <= span.end_sample:
            yield await self._decode(self._span_prefix(span, self._next_partial_end_sample))
            self._next_partial_end_sample += self._chunk_samples

    async def _close_span(self, span: AudioSpan) -> TranscriptionEvent:
        """Close one bounded native utterance and advance to a fresh segment id."""
        raw_result = await self._rescore_closed_span(span)
        result = await self._finalize_span(raw_result, span)
        self._emit_finalizer_diagnostics(result.diagnostics)
        detected, language_extra = self._language_fields(raw_result)
        start, end = self._aggregate_measured_span(result)
        words = result.words
        speaker = self._uniform_word_speaker(words)
        extra = self._event_extra(span, language_extra)
        speaker_turns = result.extra.get("std_qwen3asr_ane_diarization_turns")
        if speaker_turns is not None:
            extra["speaker_turns"] = speaker_turns
        measurement = result.extra.get("std_qwen3asr_ane_diarization_measurement")
        if measurement is not None:
            extra["diarization_measurement"] = measurement
        event = TranscriptionEvent.closed(
            self._segment_id,
            result.text,
            detected_language=detected,
            audio_processed_until=(
                self._processed_samples / SAMPLE_RATE if self._audio_progress_supported else None
            ),
            start=start,
            end=end,
            words=words,
            speaker=speaker,
            extra=extra,
        )
        self._processed_samples = max(self._processed_samples, span.end_sample)
        self._reset_active_context()
        self._segment_index += 1
        return event

    async def _rescore_closed_span(self, span: AudioSpan):
        """Decode a closed window without provisional text conditioning.

        The live partial path may retain a rollback prefix. A committed closed
        event must instead use the same full-window conditional decode as an
        offline request. Audio context reuse remains exact; decoder cache and
        prior text are discarded before this recognition pass.
        """
        self._reset_decoder_for_final_rescore()
        self._active_start_sample = span.start_sample
        await self._decode(span)
        assert self._last_result is not None
        return self._last_result

    async def _finalize_span(self, raw_result, span: AudioSpan):
        """Run optional CPU alignment/diarization after native work completed."""
        try:
            return await self._until_cancelled(
                asyncio.to_thread(
                    self.engine._finalize_chunk,
                    raw_result,
                    span.samples,
                    self.params,
                    span.start_sample / SAMPLE_RATE,
                    cancel=self._native_cancelled,
                    speaker_tracker=self._speaker_tracker,
                )
            )
        except InferenceCancelled as exc:
            self._reset_active_context()
            raise _Cancelled from exc

    def _emit_finalizer_diagnostics(self, diagnostics) -> None:
        """Expose one copy of each window-finalization note through the session channel."""
        for diagnostic in diagnostics:
            if diagnostic.code in self._finalizer_diagnostic_codes:
                continue
            if diagnostic.code == "detected_language_unmapped" and self._unmapped_language_noted:
                continue
            self.emit_diagnostic(
                code=diagnostic.code,
                message=diagnostic.message,
                level=diagnostic.level,
                param=diagnostic.param,
                provided=diagnostic.provided,
                effective=diagnostic.effective,
            )
            self._finalizer_diagnostic_codes.add(diagnostic.code)

    @staticmethod
    def _aggregate_measured_span(result) -> tuple[float | None, float | None]:
        segments = result.segments or []
        if not segments or any(
            segment.start is None or segment.end is None for segment in segments
        ):
            return None, None
        return min(segment.start for segment in segments), max(segment.end for segment in segments)

    @staticmethod
    def _uniform_word_speaker(words) -> str | None:
        if not words:
            return None
        speakers = {word.speaker for word in words}
        return next(iter(speakers)) if len(speakers) == 1 else None

    @staticmethod
    def _span_prefix(span: AudioSpan, end_sample: int) -> AudioSpan:
        if not span.start_sample < end_sample <= span.end_sample:
            raise RuntimeError("Partial decode must end inside its active audio span")
        count = end_sample - span.start_sample
        return AudioSpan(
            samples=np.array(span.samples[:count], dtype=np.float32, copy=True, order="C"),
            start_sample=span.start_sample,
            end_sample=end_sample,
        )

    def _reset_active_context(self) -> None:
        """Discard session-local prefix state before moving to another window."""
        for context in (self._decoder_context, self._audio_context):
            if context is not None:
                context.reset()
        self._decoder_context = None
        self._audio_context = None
        self._active_start_sample = None
        self._active_processed_end_sample = None
        self._decode_count = 0
        self._last_raw = ""
        self._last_result = None

    def _reset_decoder_for_final_rescore(self) -> None:
        """Drop provisional text state while retaining exact audio graph reuse."""
        if self._decoder_context is not None:
            self._decoder_context.reset()
        self._decoder_context = None
        self._active_start_sample = None
        self._active_processed_end_sample = None
        self._decode_count = 0
        self._last_raw = ""
        self._last_result = None

    def _set_input_duration(self, seconds: float) -> None:
        """Record the exact complete input length for the stream result."""
        self.set_input_duration(seconds)
