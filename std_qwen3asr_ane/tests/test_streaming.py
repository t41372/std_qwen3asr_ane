"""Recorded Standard ASR event/lifecycle tests with a deterministic fake recognizer."""

import asyncio
from itertools import pairwise
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from standard_asr import SyncSession
from standard_asr.compliance import (
    check_event_sequence,
    check_streaming_param_gating,
    check_sync_bridge,
)
from standard_asr.engine import (
    DIARIZE,
    AudioFormat,
    RuntimeParams,
    Segment,
    TranscriptionResult,
    Word,
)
from tokenizers import Tokenizer, models, pre_tokenizers

from std_qwen3asr_ane.errors import InferenceCancelled
from std_qwen3asr_ane.plugin import create_engine
from std_qwen3asr_ane.runtime import rollback_prefix

FORMAT = AudioFormat(sample_rate=16000, encoding="pcm_f32le")
WIRE = np.zeros(8000, dtype="<f4").tobytes()


@pytest.fixture
def engine():
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<unk>": 0,
                "language": 1,
                "English": 2,
                "hello": 3,
                "world": 4,
                "again": 5,
                "today": 6,
                "<asr_text>": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_tokens(["<asr_text>"])

    class Runtime:
        max_audio_seconds = 30.0

        def __init__(self):
            self.tokenizer = tokenizer
            self.calls = []
            self.entered = Event()
            self.release = Event()
            self.finished = Event()
            self.block = False
            self.contexts = []

        def new_decoder_context(self):
            context = SimpleNamespace(reset_count=0)

            def reset():
                context.reset_count += 1

            context.reset = reset
            self.contexts.append(context)
            return context

        def new_audio_context(self):
            return SimpleNamespace(reset=lambda: None)

        def transcribe(
            self,
            samples,
            *,
            language,
            max_new_tokens,
            context="",
            prefix_text="",
            decoder_context=None,
            audio_context=None,
            candidate_language_names=None,
            phrase_hints=None,
            cancel=None,
        ):
            self.entered.set()
            if self.block:
                assert self.release.wait(3), "Test did not release the worker"
            self.calls.append(
                {
                    "samples": samples.copy(),
                    "language": language,
                    "context": context,
                    "prefix": prefix_text,
                    "max_new_tokens": max_new_tokens,
                    "candidate_language_names": candidate_language_names,
                    "phrase_hints": phrase_hints,
                    "decoder_context": decoder_context,
                    "audio_context": audio_context,
                }
            )
            raw = "language English<asr_text>hello world again today"
            self.finished.set()
            return SimpleNamespace(
                text="hello world again today",
                language="en",
                raw_text=raw,
                raw_model_language="English",
            )

    result = create_engine(
        stream_chunk_seconds=0.5,
        stream_unfixed_chunks=2,
        stream_unfixed_tokens=1,
        stream_max_audio_seconds=30,
    )
    result._runtime = Runtime()
    return result


async def recorded(session, chunks=None):
    async with session:
        if chunks is not None:
            session.feed(chunks)
        return [event async for event in session]


def assert_compliant(events, engine):
    report = check_event_sequence(events, capabilities=engine.declared_capabilities)
    assert report.passed, report.issues


def test_incremental_pcm_tail_prefix_and_closed_event(engine):
    samples = np.arange(28000, dtype=np.int16)
    wire = samples.astype("<i2").tobytes()
    session = engine.start_transcription(
        audio_format=AudioFormat(sample_rate=16000, encoding="pcm_s16le"),
        params=RuntimeParams(prompt="technical vocabulary"),
    )
    events = asyncio.run(recorded(session, [wire[:1], wire[1:15999], wire[15999:]]))
    assert [len(call["samples"]) for call in engine._runtime.calls] == [8000, 16000, 24000, 28000]
    np.testing.assert_array_equal(
        engine._runtime.calls[-1]["samples"], samples.astype(np.float32) / 32768
    )
    assert all(call["context"] == "technical vocabulary" for call in engine._runtime.calls)
    assert len(engine._runtime.contexts) == 2
    assert all(
        call["decoder_context"] is engine._runtime.contexts[0]
        for call in engine._runtime.calls[:-1]
    )
    assert engine._runtime.calls[-1]["decoder_context"] is engine._runtime.contexts[1]
    assert [call["prefix"] for call in engine._runtime.calls[:2]] == ["", ""]
    expected = rollback_prefix(
        engine._runtime.tokenizer, "language English<asr_text>hello world again today", 1
    )
    assert engine._runtime.calls[2]["prefix"] == expected
    assert events[-1].type == "done"
    assert events[-2].type == "final" and events[-2].finality == "closed"
    assert all(event.stable_until == 0 for event in events if event.type == "partial")
    assert all(
        event.start is None and event.end is None and event.words is None for event in events
    )
    content = [event for event in events if event.type in ("partial", "final")]
    assert all(event.audio_processed_until == event.extra["input_end_seconds"] for event in content)
    assert session.result().text == "hello world again today"
    assert not session.diagnostics()
    assert_compliant(events, engine)


def test_whole_audio_streaming_output_and_language_override(engine):
    session = engine.start_transcription(
        audio=(np.zeros(17000, dtype=np.float32), 16000),
        params=RuntimeParams(language="yue", prompt="詞彙"),
    )
    events = asyncio.run(recorded(session))
    assert [len(call["samples"]) for call in engine._runtime.calls] == [8000, 16000, 17000]
    assert all(
        call["language"] == "yue" and call["context"] == "詞彙" for call in engine._runtime.calls
    )
    assert all(event.detected_language is None for event in events)
    assert_compliant(events, engine)


def test_finish_flush_and_fresh_session_reset(engine):
    async def scenario():
        session = engine.start_transcription(audio_format=FORMAT)
        async with session:
            await session.send_audio(np.zeros(1000, dtype="<f4").tobytes())
            await session.finish()
            events = [event async for event in session]
        return events

    first = asyncio.run(scenario())
    second = asyncio.run(scenario())
    assert len(engine._runtime.calls) == 2
    assert len(engine._runtime.contexts) == 2
    assert engine._runtime.contexts[0] is not engine._runtime.contexts[1]
    assert all(
        call["prefix"] == "" and len(call["samples"]) == 1000 for call in engine._runtime.calls
    )
    assert_compliant(first, engine)
    assert_compliant(second, engine)


def test_successful_incremental_session_sets_the_exact_input_duration(engine):
    session = engine.start_transcription(audio_format=FORMAT)
    events = asyncio.run(recorded(session, [np.zeros(8100, dtype="<f4").tobytes()]))

    assert events[-1].type == "done"
    assert session.result().duration == 8100 / 16000
    assert_compliant(events, engine)


def test_sync_bridge_empty_and_nonempty_finish(engine):
    factory = lambda: engine.start_transcription(audio_format=FORMAT)
    report = check_sync_bridge(factory, timeout=3, engine=engine)
    assert report.passed, report.issues
    assert not engine._runtime.calls
    with SyncSession(factory()) as session:
        session.send_audio(np.zeros(8100, dtype="<f4").tobytes())
        session.end_audio()
        events = list(session)
        assert session.result().text == "hello world again today"
    assert not session.is_loop_alive()
    assert_compliant(events, engine)


def test_cancel_interrupts_delivery_and_releases_audio_backpressure(engine):
    engine._runtime.block = True
    engine.config = engine.config.model_copy(update={"stream_audio_queue_size": 1})

    async def scenario():
        session = engine.start_transcription(audio_format=FORMAT)
        wire = np.zeros(8000, dtype="<f4").tobytes()
        async with session:
            consumer = asyncio.create_task(collect(session))
            await session.send_audio(wire)
            while not engine._runtime.entered.is_set():
                await asyncio.sleep(0.001)
            await session.send_audio(wire)
            blocked = asyncio.create_task(session.send_audio(wire))
            await asyncio.sleep(0.02)
            assert not blocked.done()
            await session.cancel()
            events = await asyncio.wait_for(consumer, 0.5)
            await asyncio.gather(blocked, return_exceptions=True)
            assert events[-1].code == "cancelled"
            assert session.partial_result().text == ""
            engine._runtime.release.set()
            return events

    try:
        events = asyncio.run(scenario())
    finally:
        engine._runtime.release.set()
    assert_compliant(events, engine)


async def collect(session):
    return [event async for event in session]


def test_input_backpressure_then_successful_tail_flush(engine):
    engine._runtime.block = True
    engine.config = engine.config.model_copy(update={"stream_audio_queue_size": 1})

    async def scenario():
        session = engine.start_transcription(audio_format=FORMAT)
        wire = np.zeros(8000, dtype="<f4").tobytes()
        async with session:
            consumer = asyncio.create_task(collect(session))
            await session.send_audio(wire)
            while not engine._runtime.entered.is_set():
                await asyncio.sleep(0.001)
            await session.send_audio(wire)
            sender = asyncio.create_task(session.send_audio(wire))
            await asyncio.sleep(0.02)
            assert not sender.done()
            engine._runtime.release.set()
            await asyncio.wait_for(sender, 1)
            await session.finish()
            return await consumer

    events = asyncio.run(scenario())
    assert [len(call["samples"]) for call in engine._runtime.calls] == [8000, 16000, 24000, 24000]
    assert_compliant(events, engine)


@pytest.mark.parametrize(
    "wire, code",
    [
        (b"\x00", "invalid_audio_or_context"),
        (np.array([np.nan], dtype="<f4").tobytes(), "invalid_audio_or_context"),
        (np.zeros(480001, dtype="<f4").tobytes(), "audio_limit_exceeded"),
    ],
)
def test_invalid_or_overlong_audio_is_terminal(engine, wire, code):
    session = engine.start_transcription(audio_format=FORMAT)
    events = asyncio.run(recorded(session, [wire]))
    assert events[-1].type == "error" and events[-1].code == code
    assert not engine._runtime.calls
    if code == "audio_limit_exceeded":
        assert events[-1].extra["received_audio_seconds"] > 30
        assert events[-1].extra["limit_source"] == "configured_recording_guard"
    assert_compliant(events, engine)


def test_bundle_limit_smaller_than_recording_is_segmented_without_audio_loss(engine):
    engine._runtime.max_audio_seconds = 0.25
    events = asyncio.run(
        recorded(
            engine.start_transcription(audio_format=FORMAT), [np.zeros(8000, dtype="<f4").tobytes()]
        )
    )
    assert events[-1].type == "done"
    assert [len(call["samples"]) for call in engine._runtime.calls] == [4000, 4000]
    closed = [event for event in events if event.type == "final" and event.finality == "closed"]
    assert [event.extra["input_end_seconds"] for event in closed] == [0.25, 0.5]
    assert_compliant(events, engine)


def test_bundle_capacity_errors_are_structured_not_caller_blamed(engine):
    from std_qwen3asr_ane.errors import ModelLimitError

    def fail(*args, **kwargs):
        raise ModelLimitError("Decoder KV cache exhausted before an end-of-sequence token")

    engine._runtime.transcribe = fail
    events = asyncio.run(
        recorded(
            engine.start_transcription(audio_format=FORMAT), [np.zeros(8000, dtype="<f4").tobytes()]
        )
    )
    assert events[-1].type == "error" and events[-1].code == "bundle_capacity_exceeded"
    assert "KV cache" in events[-1].extra["message"]
    assert events[-1].extra["received_audio_seconds"] == 0.5
    assert engine._runtime.contexts[0].reset_count == 1
    assert_compliant(events, engine)


def test_unmapped_model_language_is_disclosed_in_events(engine):
    original = engine._runtime.transcribe

    def transcribe(*args, **kwargs):
        result = original(*args, **kwargs)
        return SimpleNamespace(
            text=result.text,
            language="Klingon",
            raw_text=result.raw_text,
            raw_model_language="Klingon",
        )

    engine._runtime.transcribe = transcribe
    events = asyncio.run(
        recorded(
            engine.start_transcription(audio_format=FORMAT), [np.zeros(8000, dtype="<f4").tobytes()]
        )
    )
    texted = [event for event in events if event.type in ("partial", "closed")]
    assert texted and all(event.detected_language is None for event in texted)
    assert all(event.extra["unmapped_model_language"] == "Klingon" for event in texted)
    assert_compliant(events, engine)


def test_unmapped_raw_model_language_survives_in_events_and_session_diagnostics(engine):
    original = engine._runtime.transcribe

    def transcribe(*args, **kwargs):
        result = original(*args, **kwargs)
        return SimpleNamespace(
            text=result.text,
            language=None,
            raw_text=result.raw_text,
            raw_model_language="Klingon",
        )

    engine._runtime.transcribe = transcribe
    session = engine.start_transcription(audio_format=FORMAT)
    events = asyncio.run(recorded(session, [np.zeros(16000, dtype="<f4").tobytes()]))

    content = [event for event in events if event.type in ("partial", "final")]
    assert content and all(event.detected_language is None for event in content)
    assert all(event.extra["unmapped_model_language"] == "Klingon" for event in content)
    diagnostics = [item for item in session.diagnostics() if item.code == "detected_language_unmapped"]
    assert len(diagnostics) == 1
    assert diagnostics[0].provided == "Klingon"
    assert_compliant(events, engine)


def test_raw_none_model_language_is_not_reported_as_an_unknown_language(engine):
    original = engine._runtime.transcribe

    def transcribe(*args, **kwargs):
        result = original(*args, **kwargs)
        return SimpleNamespace(
            text=result.text,
            language=None,
            raw_text=result.raw_text,
            raw_model_language="None",
        )

    engine._runtime.transcribe = transcribe
    session = engine.start_transcription(audio_format=FORMAT)
    events = asyncio.run(recorded(session, [WIRE]))

    assert not any(item.code == "detected_language_unmapped" for item in session.diagnostics())
    assert all("unmapped_model_language" not in event.extra for event in events)
    assert_compliant(events, engine)


def test_default_wire_format_and_native_error_projection(engine):
    def fail(*args, **kwargs):
        raise RuntimeError("native decode failed")

    engine._runtime.transcribe = fail
    session = engine.start_transcription()
    events = asyncio.run(recorded(session, [np.zeros(8000, dtype="<i2").tobytes()]))
    assert session.audio_format.encoding == "pcm_s16le"
    assert events[-1].type == "error" and events[-1].code == "engine_error"
    assert engine._runtime.contexts[0].reset_count == 1
    assert_compliant(events, engine)


def test_native_value_error_is_an_engine_error_not_a_pcm_error(engine):
    def fail(*args, **kwargs):
        raise ValueError("native configuration failure")

    engine._runtime.transcribe = fail
    events = asyncio.run(recorded(engine.start_transcription(audio_format=FORMAT), [WIRE]))

    assert events[-1].code == "engine_error"
    assert_compliant(events, engine)


def test_closed_window_is_rescored_without_a_provisional_text_prefix(engine, monkeypatch):
    calls = []

    def recognize(samples, params, *, prefix_text="", **kwargs):
        calls.append({"samples": samples.size, "prefix": prefix_text})
        return SimpleNamespace(
            text=f"{'prefixed' if prefix_text else 'independent'}-{samples.size}",
            language="en",
            raw_text="language English<asr_text>hello world again today",
            raw_model_language="English",
        )

    monkeypatch.setattr(engine, "_recognize_chunk", recognize)
    samples = np.zeros(24000, dtype="<f4").tobytes()
    events = asyncio.run(recorded(engine.start_transcription(audio_format=FORMAT), [samples]))

    partial_calls = calls[:-1]
    assert partial_calls[-1]["prefix"]
    assert calls[-1] == {"samples": 24000, "prefix": ""}
    closed = next(event for event in events if event.type == "final" and event.finality == "closed")
    assert closed.text == "independent-24000"
    assert all(event.stable_until == 0 for event in events if event.type == "partial")
    assert_compliant(events, engine)


def test_native_cancellation_token_stops_at_a_safe_boundary_and_releases_engine(engine):
    runtime = engine._runtime
    runtime.cancel_observed = Event()
    original = runtime.transcribe

    def cancellable(*args, cancel=None, **kwargs):
        assert cancel is not None
        runtime.entered.set()
        while not cancel.is_set():
            assert not runtime.release.wait(0.001)
        runtime.cancel_observed.set()
        raise InferenceCancelled("cancelled after native prediction boundary")

    runtime.transcribe = cancellable

    async def scenario():
        first = engine.start_transcription(audio_format=FORMAT)
        async with first:
            consumer = asyncio.create_task(collect(first))
            await first.send_audio(WIRE)
            while not runtime.entered.is_set():
                await asyncio.sleep(0.001)
            await first.cancel()
            events = await asyncio.wait_for(consumer, 0.5)
            await asyncio.wait_for(asyncio.to_thread(runtime.cancel_observed.wait, 1), 1.1)

        runtime.transcribe = original
        second = engine.start_transcription(audio_format=FORMAT)
        async with second:
            second_events = asyncio.create_task(collect(second))
            await second.send_audio(WIRE)
            await second.finish()
            return events, await asyncio.wait_for(second_events, 1)

    cancelled, subsequent = asyncio.run(scenario())
    assert cancelled[-1].code == "cancelled"
    assert runtime.cancel_observed.is_set()
    assert subsequent[-1].type == "done"
    assert_compliant(cancelled, engine)
    assert_compliant(subsequent, engine)


def test_closed_windows_use_measured_finalizer_output_and_one_session_speaker_tracker(
    engine, monkeypatch
):
    engine.config = engine.config.model_copy(update={"use_alignment": True, "use_diarization": True})
    engine._runtime.max_audio_seconds = 0.25
    tracker = object()
    preflight = []
    finalizer_calls = []

    def require_artifacts(params, *, mode):
        preflight.append((params, mode, len(engine._runtime.calls)))

    def finalize(raw, samples, params, offset_seconds=0.0, *, cancel=None, speaker_tracker=None):
        finalizer_calls.append((offset_seconds, speaker_tracker, cancel, samples.size))
        first = Word(start=offset_seconds + 0.01, end=offset_seconds + 0.08, text="hello", speaker="A")
        second = Word(start=offset_seconds + 0.10, end=offset_seconds + 0.20, text="world", speaker="B")
        return TranscriptionResult(
            text="hello world",
            segments=[
                Segment(start=first.start, end=first.end, text="hello", words=[first], speaker="A"),
                Segment(start=second.start, end=second.end, text="world", words=[second], speaker="B"),
            ],
            words=[first, second],
            extra={
                "std_qwen3asr_ane_diarization_turns": [
                    {"start": offset_seconds, "end": offset_seconds + 0.1, "speaker": "A"},
                    {"start": offset_seconds + 0.1, "end": offset_seconds + 0.25, "speaker": "B"},
                ]
            },
        )

    monkeypatch.setattr(engine, "_require_request_artifacts", require_artifacts)
    monkeypatch.setattr(engine, "_new_speaker_tracker", lambda params: tracker)
    monkeypatch.setattr(engine, "_finalize_chunk", finalize)
    params = RuntimeParams(diarization=DIARIZE)
    events = asyncio.run(recorded(engine.start_transcription(audio_format=FORMAT, params=params), [WIRE]))

    closed = [event for event in events if event.type == "final" and event.finality == "closed"]
    assert len(closed) == 2
    assert len(preflight) == 1
    assert preflight[0][1:] == ("streaming", 0)
    assert preflight[0][0].diarization is not None
    assert all(call[1] is tracker for call in finalizer_calls)
    assert [event.start for event in closed] == [0.01, 0.26]
    assert [event.end for event in closed] == [0.20, 0.45]
    assert all(event.words and [word.speaker for word in event.words] == ["A", "B"] for event in closed)
    assert all(event.speaker is None for event in closed)
    assert all(event.extra["speaker_turns"] for event in closed)
    assert_compliant(events, engine)


def test_cancellation_reaches_the_auxiliary_finalizer(engine, monkeypatch):
    started = Event()
    observed = Event()

    def finalize(raw, samples, params, offset_seconds=0.0, *, cancel=None, speaker_tracker=None):
        assert cancel is not None
        started.set()
        while not cancel.is_set():
            assert not observed.wait(0.001)
        observed.set()
        raise InferenceCancelled("auxiliary cancellation observed")

    monkeypatch.setattr(engine, "_finalize_chunk", finalize)

    async def scenario():
        session = engine.start_transcription(audio_format=FORMAT)
        async with session:
            consumer = asyncio.create_task(collect(session))
            await session.send_audio(WIRE)
            await session.finish()
            while not started.is_set():
                await asyncio.sleep(0.001)
            await session.cancel()
            events = await asyncio.wait_for(consumer, 0.5)
            await asyncio.wait_for(asyncio.to_thread(observed.wait, 1), 1.1)
            return events

    events = asyncio.run(scenario())
    assert observed.is_set()
    assert events[-1].code == "cancelled"
    assert all(event.finality != "closed" for event in events if event.type == "final")
    assert_compliant(events, engine)


def test_cancel_idle_session_is_terminal(engine):
    async def scenario():
        session = engine.start_transcription(audio_format=FORMAT)
        async with session:
            consumer = asyncio.create_task(collect(session))
            await session.cancel()
            return await asyncio.wait_for(consumer, 0.5)

    events = asyncio.run(scenario())
    assert events[-1].code == "cancelled"
    assert not engine._runtime.calls
    assert_compliant(events, engine)


def test_prompt_and_phrase_hints_are_forwarded_as_real_capabilities(engine):
    params = RuntimeParams(phrase_hints=["OpenAI", "Core ML"], on_unsupported="degrade_to_prompt")
    result = engine.transcribe((np.zeros(8000, dtype=np.float32), 16000), params)
    assert engine._runtime.calls[-1]["phrase_hints"] == ["OpenAI", "Core ML"]
    assert not result.diagnostics
    session = engine.start_transcription(audio_format=FORMAT, params=params)
    events = asyncio.run(recorded(session, [np.zeros(8000, dtype="<f4").tobytes()]))
    assert engine._runtime.calls[-1]["phrase_hints"] == ["OpenAI", "Core ML"]
    assert not session.diagnostics()
    assert_compliant(events, engine)
    report = check_streaming_param_gating(engine)
    assert report.passed, report.issues


def test_three_minute_session_segments_without_losing_or_overlapping_audio(engine):
    engine.config = engine.config.model_copy(
        update={"stream_max_audio_seconds": 180, "stream_chunk_seconds": 30}
    )
    samples = np.linspace(-0.5, 0.5, 180 * 16000, dtype=np.float32)
    wire = samples.astype("<f4").tobytes()
    session = engine.start_transcription(audio_format=FORMAT)
    events = asyncio.run(recorded(session, [wire[:12345], wire[12345:]]))
    decoded = [call["samples"] for call in engine._runtime.calls]
    assert decoded and all(len(window) <= 30 * 16000 for window in decoded)
    assert all(event.stable_until == 0 for event in events if event.type == "partial")
    closed = [event for event in events if event.type == "final" and event.finality == "closed"]
    assert [event.segment_id for event in closed] == [f"utterance-{index}" for index in range(len(closed))]
    assert closed[0].extra["input_start_seconds"] == 0
    assert closed[-1].extra["input_end_seconds"] == 180
    assert all(
        left.extra["input_end_seconds"] == right.extra["input_start_seconds"]
        for left, right in pairwise(closed)
    )
    assert all(
        event.extra["input_end_seconds"] - event.extra["input_start_seconds"] <= 30
        for event in closed
    )
    assert all(event.start is None and event.end is None for event in closed)
    assert session._decoder_context is None
    assert_compliant(events, engine)


def test_recording_guards_are_optional_and_validate_explicit_finite_values():
    config = create_engine().config
    assert config.stream_max_audio_seconds is None
    assert config.max_recording_seconds is None
    assert create_engine(max_recording_seconds=600).config.max_recording_seconds == 600
    assert create_engine(stream_max_audio_seconds=600).config.stream_max_audio_seconds == 600
    for value in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            create_engine(max_recording_seconds=value)


def test_streaming_provider_budget_is_frozen_per_session(engine):
    from std_qwen3asr_ane.plugin import Qwen3ASRParams

    session = engine.start_transcription(
        audio_format=FORMAT,
        params=RuntimeParams(provider_params=Qwen3ASRParams(max_new_tokens=32)),
    )
    asyncio.run(recorded(session, [np.zeros(8000, dtype="<f4").tobytes()]))
    assert engine._runtime.calls[-1]["max_new_tokens"] == 32
    assert engine.config.max_new_tokens == 256


def test_processing_frontier_does_not_rewind_at_an_earlier_low_energy_cut(engine):
    engine.config = engine.config.model_copy(update={"stream_chunk_seconds": 0.1})
    engine._runtime.max_audio_seconds = 2.0
    session = engine.start_transcription(audio_format=FORMAT)
    frames = [
        np.full(1600, 0.0 if index in (14, 15) else 0.25, dtype="<f4").tobytes()
        for index in range(30)
    ]
    events = asyncio.run(recorded(session, frames))
    assert events[-1].type == "done"
    cursors = [event.audio_processed_until for event in events if event.audio_processed_until is not None]
    assert cursors == sorted(cursors)
    assert cursors[-1] == 3.0
    closed = [event for event in events if event.type == "final"]
    assert len(closed) >= 2
    assert session.result().duration == 3.0
    assert_compliant(events, engine)
