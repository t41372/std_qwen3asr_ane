"""Bounded public bulk recognition using Standard ASR request preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from standard_asr.contract.exceptions import (
    StructuredError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.engine import Diagnostic, RuntimeParams, TranscriptionResult

from .batching import OfflineRecognitionRequest
from .decoding_guidance import GuidanceRequestError
from .errors import ModelLimitError
from .languages import LANGUAGE_NAMES, qwen_language_key

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


def transcribe_many(
    engine: Qwen3ASREngine,
    recordings: Sequence,
    params: RuntimeParams | Sequence[RuntimeParams | None] | None = None,
    *,
    batch_size: int = 4,
) -> tuple[BulkTranscriptionOutcome, ...]:
    """Prepare and execute bounded groups without duplicating standard negotiation."""
    from .plugin import Qwen3ASRParams

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
    outcomes: dict[int, BulkTranscriptionOutcome] = {}
    with engine._operation_lock:
        for first in range(0, len(recordings), batch_size):
            prepared = []
            for index in range(first, min(first + batch_size, len(recordings))):
                execution = "not_run"
                fallback_reason = None
                try:
                    request = engine._prepare_transcription_request(
                        recordings[index], parameters[index]
                    )
                    provider = request.params.provider_params or Qwen3ASRParams()
                    request = replace(
                        request,
                        params=request.params.model_copy(
                            update={
                                "provider_params": provider.model_copy(
                                    update={"disable_draft": True}
                                ),
                            }
                        ),
                    )
                    engine._require_request_artifacts(request.params, mode="batch")
                    window = engine._stream_window_seconds()
                    if request.audio.array.size > int(window * request.audio.sample_rate):
                        execution = "serial"
                        fallback_reason = "long recording uses bounded windows"
                        result = engine._transcribe_recording(request.audio, request.params)
                        result = engine._finalize_transcription_result(result, request)
                        outcomes[index] = BulkTranscriptionOutcome(
                            index, result, None, execution, fallback_reason
                        )
                    else:
                        prepared.append((index, request))
                except Exception as error:  # noqa: BLE001 - each input has an explicit error outcome
                    outcomes[index] = BulkTranscriptionOutcome(
                        index,
                        None,
                        _public_error(error) if execution == "serial" else error,
                        execution,
                        fallback_reason,
                    )
            if not prepared:
                continue
            native_requests = []
            for _, request in prepared:
                effective = request.params
                native_requests.append(
                    OfflineRecognitionRequest(
                        samples=request.audio.array,
                        language=qwen_language_key(
                            None if effective.language == "auto" else effective.language
                        ),
                        max_new_tokens=engine._generation_budget(effective),
                        context=effective.prompt or "",
                        candidate_language_names=[
                            LANGUAGE_NAMES[qwen_language_key(tag)]
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
                    decoded = runtime.transcribe_many(
                        native_requests, batch_head=engine._batch_head
                    )
                decoded = tuple(decoded)
                expected_indices = set(range(len(prepared)))
                received_indices = [getattr(item, "request_index", None) for item in decoded]
                if (
                    len(decoded) != len(prepared)
                    or any(type(index) is not int for index in received_indices)
                    or set(received_indices) != expected_indices
                    or any(
                        getattr(item, "execution", None) not in {"packed", "serial"}
                        or (getattr(item, "result", None) is None)
                        == (getattr(item, "error", None) is None)
                        or (
                            getattr(item, "error", None) is not None
                            and not isinstance(item.error, Exception)
                        )
                        for item in decoded
                    )
                ):
                    raise RuntimeError("Native batch did not return exactly one outcome per input")
            except Exception as error:  # noqa: BLE001 - a failed native group has per-input errors
                for index, _ in prepared:
                    outcomes[index] = BulkTranscriptionOutcome(
                        index, None, _public_error(error), "unknown" if dispatched else "not_run"
                    )
                continue
            for native in decoded:
                index, request = prepared[native.request_index]
                fallback_reason = native.fallback_reason
                if head_error is not None:
                    fallback_reason = (
                        f"configured batch head could not be loaded ({type(head_error).__name__})"
                        + (f"; {fallback_reason}" if fallback_reason else "")
                    )
                if native.error is not None:
                    outcomes[index] = BulkTranscriptionOutcome(
                        index,
                        None,
                        _public_error(native.error, params=request.params),
                        native.execution,
                        fallback_reason,
                    )
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
                    result = engine._finalize_transcription_result(result, request)
                    outcomes[index] = BulkTranscriptionOutcome(
                        index, result, None, native.execution, fallback_reason
                    )
                except Exception as error:  # noqa: BLE001 - preserve successful peers on projection failure
                    outcomes[index] = BulkTranscriptionOutcome(
                        index, None, _public_error(error), native.execution, fallback_reason
                    )
    return tuple(outcomes[index] for index in range(len(recordings)))
