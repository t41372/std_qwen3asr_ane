"""Lightweight, model-free reproductions for the Standard ASR contract audit.

Run with ``.venv/bin/python research/standard-asr-audit-2026-09-22/probe_contracts.py``.
The probe never loads a Core ML bundle or writes artifacts. Its recording runtime
only makes the adapter's public EngineBase pipeline observable.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from standard_asr.contract.exceptions import InvalidProviderParamError
from standard_asr.engine import ProviderParams, RuntimeParams

from std_qwen3asr_ane.plugin import (
    Qwen3ASREngine,
    Qwen3ASRParams,
    ShortDictationEngine,
    create_engine,
    detected_language,
)
from std_qwen3asr_ane.runtime import build_prompt, parse_output


class RecordingRuntime:
    """Minimal native-runtime stand-in; records values passed after base gating."""

    max_audio_seconds = 30.0

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        max_new_tokens: int,
        context: str = "",
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "samples": samples.copy(),
                "language": language,
                "max_new_tokens": max_new_tokens,
                "context": context,
            }
        )
        return SimpleNamespace(text="ok", language="English")


class ForeignParams(ProviderParams):
    """A deliberately different engine's terminal params type."""

    value: int = 1


class NeverEncodeTokenizer:
    """The language validator runs before tokenizer encoding is needed."""

    def encode(self, *_: object, **__: object) -> object:  # pragma: no cover - must not run
        raise AssertionError("build_prompt accepted an unsupported language")


def make_recording_engine() -> tuple[Qwen3ASREngine, RecordingRuntime]:
    engine = create_engine()
    runtime = RecordingRuntime()
    engine._runtime = runtime
    return engine, runtime


def probe_declared_preset_drift() -> None:
    general = create_engine(profile="short-dictation")
    short = ShortDictationEngine()
    assert type(general) is Qwen3ASREngine
    assert general.properties.model_id == "std-qwen3asr-ane/1.7b"
    assert general.properties.max_audio_duration == 30.0
    assert general.config.profile == "short-dictation"
    assert general.config.max_new_tokens == 128
    assert general.config.model_dir.name == "qwen3-asr-1.7b-short-dictation"
    assert short.properties.model_id == "std-qwen3asr-ane/1.7b-short-dictation"
    assert short.properties.max_audio_duration == 12.0
    print("preset drift: general entry point accepts the short-dictation bundle selection")


def probe_language_handoff_and_candidates() -> None:
    engine, runtime = make_recording_engine()
    # EngineBase accepts en-US through RFC 4647 lookup against declared "en".
    engine.transcribe((np.zeros(1600, dtype=np.float32), 16000), RuntimeParams(language="en-US"))
    assert runtime.calls[-1]["language"] == "en-US"
    try:
        build_prompt(NeverEncodeTokenizer(), 1, "en-US")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unsupported language" in str(exc)
    else:  # pragma: no cover - assertion makes the current defect obvious
        raise AssertionError("CoreML prompt builder unexpectedly accepts en-US")

    # Candidate lists are intentionally unsupported: the base layer removes them
    # and reports a diagnostic before the adapter reaches native inference.
    result = engine.transcribe(
        (np.zeros(1600, dtype=np.float32), 16000),
        RuntimeParams(candidate_languages=["en", "zh"]),
    )
    assert runtime.calls[-1]["language"] is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "candidate_languages_ignored"
    ]
    print(
        "current defect reproduced: EngineBase forwards en-US unchanged and "
        "CoreML prompt validation rejects it; candidates are correctly diagnosed"
    )


def probe_guidance_and_swap_safety() -> None:
    engine, runtime = make_recording_engine()
    result = engine.transcribe(
        (np.zeros(1600, dtype=np.float32), 16000),
        RuntimeParams(
            phrase_hints=["OpenAI", "Core ML"], on_unsupported="degrade_to_prompt"
        ),
    )
    assert "OpenAI" in runtime.calls[-1]["context"]
    assert "Core ML" in runtime.calls[-1]["context"]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "guidance_degraded_to_prompt"
    ]

    try:
        engine.transcribe(
            (np.zeros(1600, dtype=np.float32), 16000),
            RuntimeParams(provider_params=ForeignParams()),
        )
    except InvalidProviderParamError:
        pass
    else:  # pragma: no cover - assertion makes a failed contract obvious
        raise AssertionError("wrong-engine provider params were not rejected")
    assert all(call["max_new_tokens"] == 256 for call in runtime.calls)
    assert Qwen3ASRParams(max_new_tokens=32).max_new_tokens == 32
    print("guidance degradation and exact provider-params swap safety work through EngineBase")


def probe_unknown_detection_loss() -> None:
    text, parsed_language = parse_output("language Klingon<asr_text>hello", None)
    assert (text, parsed_language) == ("hello", None)
    assert detected_language(parsed_language, None) == (None, [])
    print("unknown detection: parse_output discards the raw unknown name before disclosure")


if __name__ == "__main__":
    probe_declared_preset_drift()
    probe_language_handoff_and_candidates()
    probe_guidance_and_swap_safety()
    probe_unknown_detection_loss()
    print("all model-free contract probes passed")
