"""Bounded public bulk recognition using Standard ASR request preparation."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event
from typing import TYPE_CHECKING, Literal

import numpy as np
from standard_asr.contract.exceptions import (
    StructuredError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.engine import (
    Diagnostic,
    EngineBase,
    PreparedAudio,
    RuntimeParams,
    TranscriptionResult,
)

from .batching import OfflineRecognitionRequest
from .decoding_guidance import GuidanceRequestError
from .errors import ModelLimitError
from .languages import LANGUAGE_NAMES, qwen_control_language

if TYPE_CHECKING:
    from .plugin import Qwen3ASREngine


@dataclass(frozen=True)
class BulkTranscriptionOutcome:
    """One input's complete result or explicit failure, preserving input order."""

    request_index: int
    result: TranscriptionResult | None
    error: Exception | None
    execution: Literal["packed", "serial", "not_run", "unknown"]
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("An outcome must contain exactly one result or error")

    def result_or_raise(self) -> TranscriptionResult:
        """Return this input's result, raising its original typed failure otherwise."""
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _public_error(error: Exception, *, params: RuntimeParams | None = None) -> Exception:
    """Map a native failure without changing already-public error types."""
    if isinstance(error, StructuredError):
        return error
    if isinstance(error, GuidanceRequestError):
        result = UnsupportedFeatureError(
            str(error),
            param="phrase_hints"
            if params is not None and params.phrase_hints
            else "candidate_languages",
        )
    elif isinstance(error, ModelLimitError):
        result = TranscriptionError(f"Request exceeds the loaded bundle's capacity: {error}")
    else:
        result = TranscriptionError("Qwen3-ASR recognition failed for this input.")
    result.__cause__ = error
    return result


@dataclass
class _PendingRequest:
    """A standard pipeline paused at its documented engine execution hook."""

    audio: PreparedAudio
    params: RuntimeParams
    reply: Future[TranscriptionResult] = field(default_factory=Future)
    execution: Literal["packed", "serial", "not_run", "unknown"] = "not_run"
    fallback_reason: str | None = None


class _BulkPipeline(EngineBase):
    """Run the core's public pipeline while the coordinator owns inference.

    Each input has its own adapter and language-validation cache. Only audio
    preparation runs concurrently; native inference and auxiliary models stay
    on the calling thread. This avoids copying core gating or result semantics.
    """

    def __init__(
        self, engine: Qwen3ASREngine, ready: Queue[int], interrupted: Event, index: int
    ) -> None:
        self._engine = engine
        self.config = engine.config
        self.properties = engine.properties
        self.declared_capabilities = engine.declared_capabilities
        self.provider_params_type = engine.provider_params_type
        self._ready = ready
        self._interrupted = interrupted
        self._index = index
        self.pending: _PendingRequest | None = None

    @property
    def effective_capabilities(self):
        return self._engine.effective_capabilities

    def _prepare_audio(self, audio) -> PreparedAudio:
        return self._engine._prepare_audio(audio)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        self.pending = _PendingRequest(prepared, params)
        self._ready.put(self._index)
        if self._interrupted.is_set() and not self.pending.reply.done():
            raise TranscriptionError("Bulk transcription was interrupted.")
        return self.pending.reply.result()

    def run(self, audio, params: RuntimeParams | None) -> BulkTranscriptionOutcome:
        try:
            result = self.transcribe(audio, params)
            error = None
        except Exception as exc:  # noqa: BLE001 - errors belong to their input
            result, error = None, exc
        finally:
            # A rejected input never enters _transcribe. It must still release
            # the coordinator's rendezvous so successful peers can execute.
            if self.pending is None:
                self._ready.put(self._index)
        pending = self.pending
        return BulkTranscriptionOutcome(
            self._index,
            result,
            error,
            pending.execution if pending is not None else "not_run",
            pending.fallback_reason if pending is not None else None,
        )


def transcribe_many(
    engine: Qwen3ASREngine,
    recordings: Sequence,
    params: RuntimeParams | Sequence[RuntimeParams | None] | None = None,
    *,
    batch_size: int = 4,
) -> tuple[BulkTranscriptionOutcome, ...]:
    """Run bounded groups through Standard ASR's public transcription pipeline.

    At most ``batch_size`` workers decode/prepare independent inputs, then wait
    at the engine hook for a single coordinator to dispatch actual packed
    inference. A group finishes before another starts, bounding both workers
    and prepared recordings; this is not concurrent access to native models.
    """
    if not isinstance(recordings, Sequence) or isinstance(
        recordings, (str, bytes, Path, np.ndarray)
    ):
        raise TypeError("Pass a sequence of independent audio inputs")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer from 1 to 16")
    if params is None or isinstance(params, RuntimeParams):
        parameters = [params] * len(recordings)
    else:
        parameters = list(params)
        if len(parameters) != len(recordings):
            raise ValueError("Per-input params must match the number of recordings")
    outcomes = []
    with engine._operation_lock, ThreadPoolExecutor(max_workers=batch_size) as pool:
        for first in range(0, len(recordings), batch_size):
            ready: Queue[int] = Queue()
            interrupted = Event()
            adapters = [
                _BulkPipeline(engine, ready, interrupted, index)
                for index in range(first, min(first + batch_size, len(recordings)))
            ]
            completed = False
            try:
                futures = [
                    pool.submit(adapter.run, recordings[adapter._index], parameters[adapter._index])
                    for adapter in adapters
                ]
                for _ in adapters:
                    ready.get()
                _dispatch_group(engine, [a.pending for a in adapters if a.pending is not None])
                completed = True
            finally:
                # Also handles interruption while waiting for preparation. The
                # worker may enter its hook after this loop, so it also checks
                # the interruption flag before waiting for its reply.
                if not completed:
                    interrupted.set()
                for adapter in adapters:
                    if adapter.pending is not None and not adapter.pending.reply.done():
                        adapter.pending.reply.set_exception(
                            TranscriptionError("Bulk transcription was interrupted.")
                        )
            outcomes.extend(future.result() for future in futures)
    return tuple(outcomes)


def _dispatch_group(engine: Qwen3ASREngine, requests: list[_PendingRequest]) -> None:
    """Execute ready requests in input order, resolving every hook reply."""
    from .plugin import Qwen3ASRParams

    prepared = []
    for request in requests:
        try:
            provider = request.params.provider_params or Qwen3ASRParams()
            request.params = request.params.model_copy(
                update={"provider_params": provider.model_copy(update={"disable_draft": True})}
            )
            engine._require_request_artifacts(request.params, mode="batch")
            window = engine._stream_window_seconds()
            if request.audio.array.size > int(window * request.audio.sample_rate):
                request.execution = "serial"
                request.fallback_reason = "long recording uses bounded windows"
                request.reply.set_result(
                    engine._transcribe_recording(request.audio, request.params)
                )
            else:
                prepared.append(request)
        except Exception as error:  # noqa: BLE001 - keep successful peers
            request.reply.set_exception(_public_error(error, params=request.params))
    if not prepared:
        return
    native_requests = []
    for request in prepared:
        effective = request.params
        native_requests.append(
            OfflineRecognitionRequest(
                samples=request.audio.array,
                language=qwen_control_language(
                    None if effective.language == "auto" else effective.language
                ),
                max_new_tokens=engine._generation_budget(effective),
                context=effective.prompt or "",
                candidate_language_names=[
                    LANGUAGE_NAMES[qwen_control_language(tag)]
                    for tag in effective.candidate_languages
                ]
                if effective.candidate_languages
                else None,
                phrase_hints=effective.phrase_hints,
            )
        )
    head_error = None
    dispatched = False
    try:
        with engine._inference_lock:
            runtime = engine._ensure_model_loaded()
            if engine._batch_head is None and engine.config.batch_head_dir is not None:
                path = engine.config.batch_head_dir.expanduser()
                if path.exists():
                    try:
                        engine._batch_head = runtime.load_batch_head(path)
                    except Exception as error:  # noqa: BLE001 - an optional optimization
                        # A broken optional head does not make the target unusable.
                        # Keep the native serial path and disclose the failed load.
                        head_error = error
            dispatched = True
            decoded = runtime.transcribe_many(native_requests, batch_head=engine._batch_head)
        decoded = tuple(decoded)
        expected_indices = set(range(len(prepared)))
        received_indices = [getattr(item, "request_index", None) for item in decoded]
        if (
            len(decoded) != len(prepared)
            or any(type(index) is not int for index in received_indices)
            or set(received_indices) != expected_indices
            or any(
                getattr(item, "execution", None) not in {"packed", "serial"}
                or (getattr(item, "result", None) is None) == (getattr(item, "error", None) is None)
                or (
                    getattr(item, "error", None) is not None
                    and not isinstance(item.error, Exception)
                )
                for item in decoded
            )
        ):
            raise RuntimeError("Native batch did not return exactly one outcome per input")
    except Exception as error:  # noqa: BLE001 - a failed native group has per-input errors
        for request in prepared:
            request.execution = "unknown" if dispatched else "not_run"
            request.reply.set_exception(_public_error(error))
        return
    for native in decoded:
        request = prepared[native.request_index]
        fallback_reason = native.fallback_reason
        if head_error is not None:
            fallback_reason = (
                f"configured batch head could not be loaded ({type(head_error).__name__})"
                + (f"; {fallback_reason}" if fallback_reason else "")
            )
        request.execution = native.execution
        request.fallback_reason = fallback_reason
        if native.error is not None:
            request.reply.set_exception(_public_error(native.error, params=request.params))
            continue
        try:
            result = engine._finalize_chunk(
                native.result,
                request.audio.array,
                request.params,
                speaker_tracker=engine._new_speaker_tracker(request.params),
            )
            if fallback_reason is not None:
                result.diagnostics.append(
                    Diagnostic(
                        code="native_batch_serial_fallback",
                        message="This input used the serial target decoder.",
                        provided=fallback_reason,
                        effective="serial",
                    )
                )
            request.reply.set_result(result)
        except Exception as error:  # noqa: BLE001 - preserve successful peers on projection failure
            request.reply.set_exception(_public_error(error, params=request.params))
