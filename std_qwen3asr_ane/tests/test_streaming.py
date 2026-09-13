"""Recorded Standard ASR event/lifecycle tests with a deterministic fake recognizer."""

import asyncio
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
from standard_asr.contract.exceptions import UnsupportedFeatureError
from standard_asr.engine import AudioFormat, RuntimeParams
from tokenizers import Tokenizer, models, pre_tokenizers

from std_qwen3asr_ane.plugin import create_engine
from std_qwen3asr_ane.runtime import rollback_prefix

FORMAT = AudioFormat(sample_rate=16000, encoding="pcm_f32le")


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

        def transcribe(
            self,
            samples,
            *,
            language,
            max_new_tokens,
            context="",
            prefix_text="",
            decoder_context=None,
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
                    "decoder_context": decoder_context,
                }
            )
            raw = "language English<asr_text>hello world again today"
            self.finished.set()
            return SimpleNamespace(text="hello world again today", language="en", raw_text=raw)

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
    assert len(engine._runtime.contexts) == 1
    assert all(
        call["decoder_context"] is engine._runtime.contexts[0] for call in engine._runtime.calls
    )
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
            assert session.result().text == ""
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
    assert [len(call["samples"]) for call in engine._runtime.calls] == [8000, 16000, 24000]
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
        assert events[-1].extra["limit_source"] == "configured_session_limit"
    assert_compliant(events, engine)


def test_bundle_limit_smaller_than_session_limit_is_structured(engine):
    engine._runtime.max_audio_seconds = 0.25
    events = asyncio.run(
        recorded(
            engine.start_transcription(audio_format=FORMAT), [np.zeros(8000, dtype="<f4").tobytes()]
        )
    )
    assert events[-1].code == "audio_limit_exceeded"
    assert events[-1].extra["max_audio_seconds"] == 0.25
    assert events[-1].extra["limit_source"] == "bundle_audio_limit"
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
        return SimpleNamespace(text=result.text, language="Klingon", raw_text=result.raw_text)

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


def test_prompt_and_phrase_hint_degradation_are_real_capabilities(engine):
    params = RuntimeParams(phrase_hints=["OpenAI", "Core ML"], on_unsupported="degrade_to_prompt")
    result = engine.transcribe((np.zeros(8000, dtype=np.float32), 16000), params)
    assert "OpenAI" in engine._runtime.calls[-1]["context"]
    assert result.diagnostics
    session = engine.start_transcription(audio_format=FORMAT, params=params)
    events = asyncio.run(recorded(session, [np.zeros(8000, dtype="<f4").tobytes()]))
    assert "Core ML" in engine._runtime.calls[-1]["context"]
    assert session.diagnostics()
    assert_compliant(events, engine)
    with pytest.raises(UnsupportedFeatureError):
        engine.start_transcription(
            audio_format=FORMAT, params=RuntimeParams(phrase_hints=["OpenAI"])
        )
    report = check_streaming_param_gating(engine)
    assert report.passed, report.issues


def test_three_minute_session_keeps_complete_audio_and_one_private_decoder_context(engine):
    engine.config = engine.config.model_copy(
        update={"stream_max_audio_seconds": 180, "stream_chunk_seconds": 30}
    )
    engine._runtime.max_audio_seconds = 180
    samples = np.linspace(-0.5, 0.5, 180 * 16000, dtype=np.float32)
    wire = samples.astype("<f4").tobytes()
    session = engine.start_transcription(audio_format=FORMAT)
    events = asyncio.run(recorded(session, [wire[:12345], wire[12345:]]))
    assert [len(call["samples"]) for call in engine._runtime.calls] == [
        seconds * 16000 for seconds in range(30, 181, 30)
    ]
    np.testing.assert_array_equal(engine._runtime.calls[-1]["samples"], samples)
    assert len(engine._runtime.contexts) == 1
    assert all(
        call["decoder_context"] is engine._runtime.contexts[0] for call in engine._runtime.calls
    )
    assert all(event.stable_until == 0 for event in events if event.type == "partial")
    assert events[-2].finality == "closed"
    assert events[-2].extra["audio_prefix_seconds"] == 180
    assert session._decoder_context is None
    assert_compliant(events, engine)


def test_stream_duration_is_configurable_beyond_thirty_seconds_and_finite():
    assert create_engine().config.stream_max_audio_seconds == 180
    assert create_engine(stream_max_audio_seconds=600).config.stream_max_audio_seconds == 600
    for value in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            create_engine(stream_max_audio_seconds=value)
