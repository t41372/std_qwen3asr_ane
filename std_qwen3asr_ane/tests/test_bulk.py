"""Public bounded bulk recognition uses the real Standard ASR request pipeline."""

# ruff: noqa: F811

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from standard_asr.contract.exceptions import (
    InvalidProviderParamError,
    TranscriptionError,
)
from standard_asr.engine import DIARIZE, ProviderParams, RuntimeParams
from test_plugin import bundle, fake_runtime  # noqa: F401 - shared public-engine fixtures

from std_qwen3asr_ane.plugin import Qwen3ASRParams, create_engine


def _native_result(text: str, *, language: str | None = "English") -> SimpleNamespace:
    """Return the small native shape consumed by the plugin's result projector."""
    return SimpleNamespace(
        text=text,
        language=language,
        raw_model_language=language,
        raw_text=text,
        token_ids=(),
        audio_tokens=1,
        timings={},
    )


def _native_outcome(
    index: int,
    *,
    text: str | None = None,
    error: Exception | None = None,
    execution: str = "packed",
    fallback_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_index=index,
        result=None if error is not None else _native_result(text or f"item-{index}"),
        error=error,
        execution=execution,
        fallback_reason=fallback_reason,
    )


def test_bulk_prepares_each_input_with_standard_gating_and_preserves_order(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_params = []

    def require_request_artifacts(params, *, mode):
        assert mode == "batch"
        captured_params.append(params)

    def transcribe_many(self, requests, *, batch_head):
        assert batch_head is None
        self.native_requests = tuple(requests)
        # Native result order is deliberately unrelated to caller order.
        return tuple(
            _native_outcome(index, text=f"native-{index}")
            for index in reversed(range(len(requests)))
        )

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle, draft_dir=bundle / "optional-draft")
    monkeypatch.setattr(engine, "_require_request_artifacts", require_request_artifacts)
    recordings = [
        (np.zeros(4_000, np.float32), 8_000),
        (np.zeros(16_000, np.float32), 16_000),
        (np.zeros(16_000, np.float32), 16_000),
    ]
    params = [
        RuntimeParams(language="en-US", provider_params=Qwen3ASRParams(max_new_tokens=77)),
        RuntimeParams(candidate_languages=["en"]),
        RuntimeParams(phrase_hints=["Acme"], provider_params=Qwen3ASRParams(include_metrics=True)),
    ]

    outcomes = engine.transcribe_many(recordings, params, batch_size=3)

    assert [outcome.request_index for outcome in outcomes] == [0, 1, 2]
    assert [outcome.result_or_raise().text for outcome in outcomes] == [
        "native-0",
        "native-1",
        "native-2",
    ]
    (runtime,) = fake_runtime.instances
    native = runtime.native_requests
    assert [request.language for request in native] == ["en", None, None]
    assert native[0].samples.size == 8_000  # Standard ASR resampled the first input to 16 kHz.
    assert native[0].max_new_tokens == 77
    assert native[1].candidate_language_names == ["English"]
    assert native[2].phrase_hints == ["Acme"]
    assert all(item.provider_params.disable_draft for item in captured_params)
    assert captured_params[0].provider_params.max_new_tokens == 77
    assert captured_params[2].provider_params.include_metrics
    diagnostics = outcomes[0].result_or_raise().diagnostics
    assert [item.code for item in diagnostics].count("resampled_with") == 1
    assert [item.code for item in diagnostics].count("language_refinement_accepted") == 1


def test_bulk_uses_bounded_default_groups_and_rejects_invalid_group_sizes(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    group_sizes = []

    def transcribe_many(self, requests, *, batch_head):
        group_sizes.append(len(requests))
        return tuple(_native_outcome(index) for index in range(len(requests)))

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many([audio] * 5)

    assert group_sizes == [4, 1]
    assert [outcome.result_or_raise().text for outcome in outcomes] == [
        "item-0",
        "item-1",
        "item-2",
        "item-3",
        "item-0",
    ]
    for invalid_size in (0, 17, True):
        with pytest.raises(ValueError, match="1 to 16"):
            engine.transcribe_many([], batch_size=invalid_size)


def test_bulk_rejects_swapped_provider_before_internal_draft_disable_and_keeps_peers(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ForeignParams(ProviderParams):
        pass

    def transcribe_many(self, requests, *, batch_head):
        self.native_requests = tuple(requests)
        return tuple(_native_outcome(index) for index in range(len(requests)))

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many(
        [audio, object(), audio],
        [
            RuntimeParams(provider_params=ForeignParams()),
            RuntimeParams(),
            RuntimeParams(provider_params=Qwen3ASRParams(max_new_tokens=31)),
        ],
        batch_size=3,
    )

    assert type(outcomes[0].error) is InvalidProviderParamError
    assert type(outcomes[1].error) is TypeError
    assert outcomes[2].result_or_raise().text == "item-0"
    (runtime,) = fake_runtime.instances
    assert len(runtime.native_requests) == 1
    assert runtime.native_requests[0].max_new_tokens == 31


def test_bulk_surfaces_per_item_native_errors_without_losing_successful_peers(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def transcribe_many(self, requests, *, batch_head):
        return (
            _native_outcome(1, error=RuntimeError("one native lane failed")),
            _native_outcome(0, text="survived"),
        )

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many([audio, audio], batch_size=2)

    assert outcomes[0].result_or_raise().text == "survived"
    assert isinstance(outcomes[1].error, TranscriptionError)
    assert isinstance(outcomes[1].error.__cause__, RuntimeError)


def test_bulk_never_leaves_a_slot_unresolved_when_native_batch_output_is_malformed(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def transcribe_many(self, requests, *, batch_head):
        # A native bridge returning a partial group used to make the public
        # reordering dictionary raise KeyError, hiding every outcome.
        return (_native_outcome(0),)

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many([audio, audio], batch_size=2)

    assert len(outcomes) == 2
    assert all(outcome.execution == "unknown" for outcome in outcomes)
    assert all(isinstance(outcome.error, TranscriptionError) for outcome in outcomes)
    assert all(isinstance(outcome.error.__cause__, RuntimeError) for outcome in outcomes)


def test_bulk_reports_explicit_serial_fallback_when_no_compact_head_is_available(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def transcribe_many(self, requests, *, batch_head):
        assert batch_head is None
        return tuple(
            _native_outcome(
                index,
                execution="serial",
                fallback_reason="no target-bound compact batch head was supplied",
            )
            for index in range(len(requests))
        )

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)

    (outcome,) = engine.transcribe_many([audio])

    result = outcome.result_or_raise()
    assert outcome.execution == "serial"
    assert outcome.fallback_reason is not None
    assert [item.code for item in result.diagnostics] == ["native_batch_serial_fallback"]


def test_bulk_routes_long_recordings_to_the_windowed_serial_wrapper(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["max_audio_seconds"] = 1.0
    manifest_path.write_text(json.dumps(manifest))

    def transcribe_many(self, requests, *, batch_head):
        self.native_requests = tuple(requests)
        return tuple(_native_outcome(index, text="short") for index in range(len(requests)))

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    outcomes = engine.transcribe_many(
        [
            (np.zeros(32_000, np.float32), 16_000),
            (np.zeros(8_000, np.float32), 16_000),
        ],
        batch_size=2,
    )

    assert outcomes[0].execution == "serial"
    assert outcomes[0].fallback_reason == "long recording uses bounded windows"
    assert outcomes[0].result_or_raise().duration == 2.0
    assert outcomes[1].result_or_raise().text == "short"
    (runtime,) = fake_runtime.instances
    assert len(runtime.native_requests) == 1
    assert runtime.native_requests[0].samples.size == 8_000
    assert len(runtime.calls) == 2


def test_bulk_runs_optional_postprocessing_once_per_completed_native_item(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def transcribe_many(self, requests, *, batch_head):
        return tuple(_native_outcome(index) for index in range(len(requests)))

    class Auxiliary:
        def __init__(self) -> None:
            self.calls = []

        def annotate(self, result, samples, params, offset_seconds, **kwargs):
            self.calls.append(
                (samples.copy(), params, offset_seconds, kwargs.get("speaker_tracker"))
            )
            return result

        def new_speaker_tracker(self):
            return object()

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle, use_alignment=True, use_diarization=True)
    auxiliary = Auxiliary()
    monkeypatch.setattr(engine, "_auxiliary_models", lambda: auxiliary)
    monkeypatch.setattr(engine, "_require_request_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_require_artifacts", lambda **_kwargs: None)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many(
        [audio, audio],
        [RuntimeParams(word_timestamps="word"), RuntimeParams(diarization=DIARIZE)],
        batch_size=2,
    )

    assert [outcome.result_or_raise().text for outcome in outcomes] == ["item-0", "item-1"]
    assert [call[1].word_timestamps for call in auxiliary.calls] == ["word", None]
    assert [call[1].diarization is not None for call in auxiliary.calls] == [False, True]
    assert len({id(call[3]) for call in auxiliary.calls if call[3] is not None}) == 1


def test_close_waits_for_an_active_bulk_operation(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()

    def transcribe_many(self, requests, *, batch_head):
        self.active += 1
        entered.set()
        try:
            assert release.wait(2)
            return tuple(_native_outcome(index) for index in range(len(requests)))
        finally:
            self.active -= 1

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(engine.transcribe_many, [audio])
        assert entered.wait(1)
        closing = pool.submit(engine.close)
        try:
            assert not closing.done()
        finally:
            release.set()
        assert pending.result()[0].result_or_raise().text == "item-0"
        closing.result()
    assert fake_runtime.instances[0].closed


def test_bulk_retains_standard_speaker_synthesis(bundle, fake_runtime, monkeypatch) -> None:
    from standard_asr.engine import Segment, TranscriptionResult, Word

    monkeypatch.setattr(
        fake_runtime,
        "transcribe_many",
        lambda self, requests, **kwargs: tuple(_native_outcome(i) for i in range(len(requests))),
        raising=False,
    )
    engine = create_engine(model_dir=bundle)
    monkeypatch.setattr(
        engine,
        "_finalize_chunk",
        lambda *args, **kwargs: TranscriptionResult(
            text="hello",
            segments=[
                Segment(
                    start=0,
                    end=1,
                    text="hello",
                    words=[Word(start=0, end=1, text="hello", speaker="speaker-0")],
                )
            ],
        ),
    )

    (outcome,) = engine.transcribe_many([(np.zeros(16_000, np.float32), 16_000)])

    assert outcome.result_or_raise().segments[0].speaker == "speaker-0"


def test_bulk_rejects_undeclared_language_like_public_transcribe(
    bundle, fake_runtime, monkeypatch
) -> None:
    from standard_asr.contract.exceptions import UnsupportedFeatureError

    monkeypatch.setattr(
        fake_runtime,
        "transcribe_many",
        lambda *args, **kwargs: pytest.fail("Rejected inputs must not reach inference"),
        raising=False,
    )
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)
    params = RuntimeParams(language="sw")

    with pytest.raises(UnsupportedFeatureError) as single:
        engine.transcribe(audio, params)
    (outcome,) = engine.transcribe_many([audio], params)

    assert type(outcome.error) is type(single.value)
    assert str(outcome.error) == str(single.value)
    assert outcome.execution == "not_run"
    assert not fake_runtime.instances


def test_bulk_recording_limit_is_shared_with_single_transcription(
    bundle, fake_runtime, monkeypatch
) -> None:
    from standard_asr.contract.exceptions import AudioProcessingError

    monkeypatch.setattr(
        fake_runtime,
        "transcribe_many",
        lambda self, requests, **kwargs: tuple(_native_outcome(i) for i in range(len(requests))),
        raising=False,
    )
    engine = create_engine(model_dir=bundle, max_recording_seconds=1)
    too_long = (np.zeros(16_001, np.float32), 16_000)
    valid = (np.zeros(16_000, np.float32), 16_000)

    with pytest.raises(AudioProcessingError) as single:
        engine.transcribe(too_long)
    outcomes = engine.transcribe_many([too_long, valid])

    assert type(outcomes[0].error) is type(single.value)
    assert str(outcomes[0].error) == str(single.value)
    assert outcomes[0].execution == "not_run"
    assert outcomes[1].result_or_raise().text == "item-0"


def test_bulk_coordinator_interruption_releases_waiting_workers(
    bundle, fake_runtime, monkeypatch
) -> None:
    class Interrupted(BaseException):
        pass

    def transcribe_many(self, requests, **kwargs):
        raise Interrupted("native dispatch interrupted")

    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)
    # The injected interruption reaches the coordinator while every pipeline
    # worker waits for its reply. Returning proves shutdown released them all.
    with pytest.raises(Interrupted):
        engine.transcribe_many([audio, audio])
    engine.close()
    assert fake_runtime.instances[0].closed


def test_bulk_interruption_during_preparation_releases_late_workers(
    bundle, fake_runtime, monkeypatch
) -> None:
    from queue import Queue

    from std_qwen3asr_ane import bulk

    class Interrupted(BaseException):
        pass

    entered, release = Event(), Event()
    engine = create_engine(model_dir=bundle)
    prepare = engine._prepare_audio

    def delayed_prepare(audio):
        entered.set()
        assert release.wait(2)
        return prepare(audio)

    class InterruptedQueue(Queue):
        def get(self, *args, **kwargs):
            assert entered.wait(1)
            release.set()
            raise Interrupted("interrupted before the prepared group is ready")

    monkeypatch.setattr(engine, "_prepare_audio", delayed_prepare)
    monkeypatch.setattr(bulk, "Queue", InterruptedQueue)
    audio = (np.zeros(16_000, np.float32), 16_000)

    with pytest.raises(Interrupted):
        engine.transcribe_many([audio, audio])
    assert not fake_runtime.instances
