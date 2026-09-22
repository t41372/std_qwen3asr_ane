"""Release acceptance tests across the public Standard ASR boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from standard_asr import AudioArray
from standard_asr.contract.exceptions import (
    AudioProcessingError,
    ConfigError,
    StreamFailedError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.engine import RuntimeParams

from std_qwen3asr_ane.languages import LANGUAGE_NAMES
from std_qwen3asr_ane.plugin import (
    Qwen3ASREngine,
    ShortDictationEngine,
    create_engine,
)
from std_qwen3asr_ane.runtime import build_prompt, parse_output_details


class _PromptTokenizer:
    """Capture prompt text while the production prompt builder owns its shape."""

    def __init__(self, runtime: _ReleaseRuntime):
        self.runtime = runtime

    def encode(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        self.runtime.prompt_texts.append(text)
        return SimpleNamespace(ids=[1])

    def decode(self, _ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return ""


class _ReleaseRuntime:
    """Native-boundary spy that uses the production prompt and output parsers."""

    token_batch_size = 16

    def __init__(
        self,
        *,
        max_audio_seconds: float = 30.0,
        head_kind: str = "logits",
        auto_raw: str = "language English<asr_text>hello",
        native_error: Exception | None = None,
    ) -> None:
        self.max_audio_seconds = max_audio_seconds
        self.head_output = (
            {"kind": "logits", "token_batch_size": 1}
            if head_kind == "logits"
            else {
                "kind": "chunk_max",
                "token_batch_size": 1,
                "vocabulary_chunk": 4096,
            }
        )
        self.auto_raw = auto_raw
        self.native_error = native_error
        self.prompt_texts: list[str] = []
        self.calls: list[dict[str, object]] = []
        self.tokenizer = _PromptTokenizer(self)

    def new_decoder_context(self):
        return SimpleNamespace(reset=lambda: None)

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
        cancel=None,
        candidate_language_names=None,
        phrase_hints=None,
    ):
        if self.native_error is not None:
            raise self.native_error
        build_prompt(
            self.tokenizer,
            1,
            language,
            context=context,
            prefix_text=prefix_text,
        )
        raw = self.auto_raw
        if language is not None:
            raw = "hello"
        elif candidate_language_names:
            raw = f"language {candidate_language_names[0]}<asr_text>hello"
        parsed = parse_output_details(raw, language)
        self.calls.append(
            {
                "samples": np.array(samples, copy=True),
                "language": language,
                "max_new_tokens": max_new_tokens,
                "context": context,
                "prefix_text": prefix_text,
                "decoder_context": decoder_context,
                "audio_context": audio_context,
                "cancel": cancel,
                "candidate_language_names": candidate_language_names,
                "phrase_hints": phrase_hints,
            }
        )
        return SimpleNamespace(
            text=parsed.text,
            language=parsed.language,
            raw_model_language=parsed.raw_model_language,
            raw_text=raw,
        )


def _engine(
    engine_type: type[Qwen3ASREngine],
    runtime: _ReleaseRuntime,
    **config: object,
) -> Qwen3ASREngine:
    engine = engine_type(**config)
    engine._runtime = runtime
    return engine


async def _record_whole_input(engine: Qwen3ASREngine, params: RuntimeParams):
    audio = np.zeros(160, dtype=np.float32)
    session = engine.start_transcription(audio=(audio, 16000), params=params)
    async with session:
        events = [event async for event in session]
    return events, session


@pytest.mark.parametrize("engine_type", [Qwen3ASREngine, ShortDictationEngine])
@pytest.mark.parametrize("mode", ["batch", "streaming"])
def test_every_declared_language_refinement_reaches_native_control_and_stays_forced(
    engine_type: type[Qwen3ASREngine], mode: str
) -> None:
    runtime = _ReleaseRuntime()
    engine = _engine(engine_type, runtime)
    audio = np.zeros(160, dtype=np.float32)

    for tag, qwen_name in LANGUAGE_NAMES.items():
        refinements = [f"{tag}-x-release"]
        refinements.extend(
            {
                "en": ["en-US"],
                "zh": ["zh-Hant"],
                "yue": ["yue-Hant-HK"],
            }.get(tag, [])
        )
        for refinement in refinements:
            params = RuntimeParams(language=refinement)
            if mode == "batch":
                result = engine.transcribe((audio, 16000), params)
            else:
                events, session = asyncio.run(_record_whole_input(engine, params))
                assert events[-1].type == "done"
                assert all(event.detected_language is None for event in events)
                result = session.result()
            call = runtime.calls[-1]
            assert call["language"] == tag
            assert runtime.prompt_texts[-1].endswith(f"language {qwen_name}<asr_text>")
            assert result.detected_language is None
            assert not any(item.code == "detected_language_unmapped" for item in result.diagnostics)


@pytest.mark.parametrize("engine_type", [Qwen3ASREngine, ShortDictationEngine])
@pytest.mark.parametrize("mode", ["batch", "streaming"])
@pytest.mark.parametrize(
    ("raw", "text", "diagnostic"),
    [
        ("language Klingon<asr_text>qaQ", "qaQ", "detected_language_unmapped"),
        ("language None<asr_text>", "", None),
    ],
)
def test_raw_auto_language_metadata_survives_parser_to_public_result(
    engine_type: type[Qwen3ASREngine], mode: str, raw: str, text: str, diagnostic: str | None
) -> None:
    runtime = _ReleaseRuntime(auto_raw=raw)
    engine = _engine(engine_type, runtime)
    params = RuntimeParams(language="auto")
    if mode == "batch":
        result = engine.transcribe((np.zeros(160, np.float32), 16000), params)
        diagnostics = result.diagnostics
    else:
        events, session = asyncio.run(_record_whole_input(engine, params))
        result = session.result()
        diagnostics = session.diagnostics()
        if diagnostic is not None:
            content = [event for event in events if event.type in ("partial", "final")]
            assert content and content[-1].extra["unmapped_model_language"] == "Klingon"
    assert result.text == text
    assert result.detected_language is None
    codes = [item.code for item in diagnostics]
    if diagnostic is None:
        assert "detected_language_unmapped" not in codes
    else:
        assert diagnostic in codes
        note = next(item for item in diagnostics if item.code == diagnostic)
        assert note.provided == "Klingon"


def test_native_value_error_is_runtime_failure_but_bad_array_is_audio_error() -> None:
    native = ValueError("native decoder state failed")
    runtime = _ReleaseRuntime(native_error=native)
    engine = _engine(Qwen3ASREngine, runtime)
    with pytest.raises(TranscriptionError) as caught:
        engine.transcribe((np.zeros(160, np.float32), 16000))
    assert caught.value.__cause__ is native

    healthy = _ReleaseRuntime()
    engine = _engine(Qwen3ASREngine, healthy)
    with pytest.raises(AudioProcessingError, match="at least one sample"):
        engine.transcribe(AudioArray(np.empty(0, np.float32), 16000))
    assert not healthy.calls


def test_streaming_native_value_error_is_engine_error_not_invalid_pcm() -> None:
    runtime = _ReleaseRuntime(native_error=ValueError("native decoder state failed"))
    engine = _engine(Qwen3ASREngine, runtime)
    events, session = asyncio.run(_record_whole_input(engine, RuntimeParams()))
    assert events[-1].type == "error"
    assert events[-1].code == "engine_error"
    with pytest.raises(StreamFailedError) as caught:
        session.result()
    assert caught.value.code == "engine_error"


@pytest.mark.parametrize("engine_type", [Qwen3ASREngine, ShortDictationEngine])
def test_public_array_boundary_canonicalizes_raw_stereo(
    engine_type: type[Qwen3ASREngine],
) -> None:
    runtime = _ReleaseRuntime()
    engine = _engine(engine_type, runtime)
    stereo = np.array([[2.0, np.nan], [-2.0, np.inf]], dtype=np.float32)
    result = engine.transcribe(AudioArray(stereo, 16000))
    np.testing.assert_array_equal(runtime.calls[-1]["samples"], np.array([0.5, 0.0], np.float32))
    assert [item.code for item in result.diagnostics[:3]] == [
        "audio_conversion",
        "non_finite_audio",
        "audio_clipped",
    ]


def test_general_profile_rejects_short_alias_and_presets_are_statically_unbounded() -> None:
    with pytest.raises(ConfigError, match="profile"):
        create_engine(profile="short-dictation")
    assert Qwen3ASREngine.properties.max_audio_duration is None
    assert ShortDictationEngine.properties.max_audio_duration is None


@pytest.mark.parametrize("engine_type", [Qwen3ASREngine, ShortDictationEngine])
def test_optional_recording_guard_rejects_before_native_runtime(
    engine_type: type[Qwen3ASREngine],
) -> None:
    runtime = _ReleaseRuntime(max_audio_seconds=0.1)
    engine = _engine(engine_type, runtime, max_recording_seconds=0.01)
    with pytest.raises(AudioProcessingError, match="max_audio_duration"):
        engine.transcribe((np.zeros(161, np.float32), 16000))
    assert not runtime.calls


@pytest.mark.parametrize("engine_type", [Qwen3ASREngine, ShortDictationEngine])
def test_public_batch_longform_processes_every_native_window(
    engine_type: type[Qwen3ASREngine],
) -> None:
    runtime = _ReleaseRuntime(max_audio_seconds=0.1)
    engine = _engine(engine_type, runtime)
    samples = np.zeros(4000, dtype=np.float32)
    result = engine.transcribe((samples, 16000))
    assert [len(call["samples"]) for call in runtime.calls] == [1600, 1600, 800]
    assert sum(len(call["samples"]) for call in runtime.calls) == len(samples)
    assert result.text == "hello hello hello"
    assert result.duration == 0.25
    assert result.segments is not None and len(result.segments) == 3


def test_full_logits_public_guidance_is_forwarded_without_changing_default_path() -> None:
    runtime = _ReleaseRuntime(head_kind="logits")
    engine = _engine(Qwen3ASREngine, runtime)
    audio = (np.zeros(160, np.float32), 16000)
    baseline = engine.transcribe(audio, RuntimeParams())
    baseline_call = runtime.calls[-1]
    guided = engine.transcribe(
        audio,
        RuntimeParams(
            language="auto",
            candidate_languages=["yue", "en"],
            phrase_hints=["OpenAI"],
        ),
    )
    guided_call = runtime.calls[-1]
    assert baseline.text == guided.text == "hello"
    assert baseline_call["candidate_language_names"] is None
    assert baseline_call["phrase_hints"] is None
    assert guided_call["candidate_language_names"] == ["Cantonese", "English"]
    assert guided_call["phrase_hints"] == ["OpenAI"]
    assert guided.detected_language == "yue"
    assert not any(
        item.param in {"candidate_languages", "phrase_hints"} for item in guided.diagnostics
    )


def test_compact_head_narrows_guidance_with_standard_diagnostics_and_strict_failure() -> None:
    audio = (np.zeros(160, np.float32), 16000)
    compact = _ReleaseRuntime(head_kind="chunk_max")
    strict = _engine(Qwen3ASREngine, compact, strict=True)
    candidate_result = strict.transcribe(
        audio,
        RuntimeParams(language="auto", candidate_languages=["en"]),
    )
    assert any(item.code == "candidate_languages_ignored" for item in candidate_result.diagnostics)
    assert compact.calls[-1]["candidate_language_names"] is None
    with pytest.raises(UnsupportedFeatureError) as phrase_error:
        strict.transcribe(audio, RuntimeParams(phrase_hints=["OpenAI"]))
    assert phrase_error.value.param == "phrase_hints"

    best_runtime = _ReleaseRuntime(head_kind="chunk_max")
    best_effort = _engine(Qwen3ASREngine, best_runtime, strict=False)
    result = best_effort.transcribe(audio, RuntimeParams(phrase_hints=["OpenAI"]))
    assert best_runtime.calls[-1]["phrase_hints"] is None
    assert any(item.param == "phrase_hints" for item in result.diagnostics)
