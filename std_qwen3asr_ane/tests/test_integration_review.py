"""Focused regressions found during the final Standard ASR integration review."""

# ruff: noqa: F811

from __future__ import annotations

import shlex
from types import SimpleNamespace

import numpy as np
import pytest
from standard_asr.contract.exceptions import TranscriptionError, UnsupportedFeatureError
from standard_asr.engine import DIARIZE, RuntimeParams, Segment, TranscriptionResult, Word
from test_plugin import bundle, fake_runtime  # noqa: F401 - shared engine fixtures

from std_qwen3asr_ane.auxiliary import AuxiliaryModels
from std_qwen3asr_ane.plugin import _merge_chunk_results, create_engine


def _native_result(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        language="English",
        raw_model_language="English",
        raw_text=text,
        token_ids=(),
        audio_tokens=1,
        timings={},
    )


def test_corrupt_optional_batch_head_falls_back_to_serial_target(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An optional accelerator artifact must not turn viable serial work into failure."""
    batch_head = bundle.parent / "corrupt-batch-head"
    batch_head.mkdir()

    def reject_batch_head(self, path):
        raise ValueError(f"invalid batch head at {path}")

    def transcribe_many(self, requests, *, batch_head):
        assert batch_head is None
        return tuple(
            SimpleNamespace(
                request_index=index,
                result=_native_result(f"item-{index}"),
                error=None,
                execution="serial",
                fallback_reason="no usable target-bound compact batch head was supplied",
            )
            for index in range(len(requests))
        )

    monkeypatch.setattr(fake_runtime, "load_batch_head", reject_batch_head, raising=False)
    monkeypatch.setattr(fake_runtime, "transcribe_many", transcribe_many, raising=False)
    engine = create_engine(model_dir=bundle, batch_head_dir=batch_head)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many([audio, audio], batch_size=2)

    assert [outcome.result_or_raise().text for outcome in outcomes] == ["item-0", "item-1"]
    assert [outcome.execution for outcome in outcomes] == ["serial", "serial"]
    assert all(outcome.fallback_reason for outcome in outcomes)


def test_diarization_pull_remedy_preserves_its_implicit_alignment_directory(tmp_path) -> None:
    """Diarization requires alignment, so its remedy must target both configured caches."""
    alignment_dir = tmp_path / "custom alignment"
    diarization_dir = tmp_path / "custom diarization"
    engine = create_engine(
        download_root=tmp_path,
        use_diarization=True,
        alignment_dir=alignment_dir,
        diarization_dir=diarization_dir,
    )

    command = shlex.split(engine._pull_command())

    assert "use_diarization=true" in command
    assert f"diarization_dir={diarization_dir}" in command
    assert f"alignment_dir={alignment_dir}" in command


def test_diarization_only_language_failure_names_the_requested_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alignment-language limit must not blame an unrequested timestamp channel."""
    monkeypatch.setattr("std_qwen3asr_ane.auxiliary.importlib.util.find_spec", lambda _: object())
    owner = AuxiliaryModels(SimpleNamespace(use_alignment=False, use_diarization=True))

    with pytest.raises(UnsupportedFeatureError) as caught:
        owner.validate_request(RuntimeParams(language="ar", diarization=DIARIZE))

    assert caught.value.param == "diarization"


def test_long_recording_failure_reports_serial_execution(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outcome scheduling metadata must describe work that actually entered native serial code."""
    engine = create_engine(model_dir=bundle)
    monkeypatch.setattr(engine, "_stream_window_seconds", lambda: 0.25)

    def fail_after_entering_serial(*args, **kwargs):
        raise TranscriptionError("second serial window failed")

    monkeypatch.setattr(engine, "_transcribe_recording", fail_after_entering_serial)
    audio = (np.zeros(8_000, np.float32), 16_000)

    (outcome,) = engine.transcribe_many([audio])

    assert isinstance(outcome.error, TranscriptionError)
    assert outcome.execution == "serial"
    assert outcome.fallback_reason == "long recording uses bounded windows"


def test_native_group_dispatch_failure_reports_unknown_execution(
    bundle, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed group call may have predicted; it cannot truthfully be labelled not-run."""

    def fail_during_dispatch(self, requests, *, batch_head):
        raise RuntimeError("native group failed after dispatch")

    monkeypatch.setattr(fake_runtime, "transcribe_many", fail_during_dispatch, raising=False)
    engine = create_engine(model_dir=bundle)
    audio = (np.zeros(16_000, np.float32), 16_000)

    outcomes = engine.transcribe_many([audio, audio], batch_size=2)

    assert all(isinstance(outcome.error, TranscriptionError) for outcome in outcomes)
    assert [outcome.execution for outcome in outcomes] == ["unknown", "unknown"]


def test_longform_merge_preserves_exact_segment_composition_and_global_source_offsets() -> None:
    """Flattened window details must use the final transcript's text coordinate space."""

    def chunk(text: str, first: str, second: str, split: int, input_start: float):
        words = [
            Word(
                start=input_start,
                end=input_start + 0.4,
                text=first.strip(),
                extra={"source_start": 0, "source_end": len(first.strip())},
            ),
            Word(
                start=input_start + 0.5,
                end=input_start + 0.9,
                text=second,
                extra={"source_start": split, "source_end": len(text)},
            ),
        ]
        return TranscriptionResult(
            text=text,
            segments=[
                Segment(
                    start=words[0].start,
                    end=words[0].end,
                    text=first,
                    text_separator="",
                    words=[words[0]],
                    extra={"source_start": 0, "source_end": split},
                ),
                Segment(
                    start=words[1].start,
                    end=words[1].end,
                    text=second,
                    text_separator="",
                    words=[words[1]],
                    extra={"source_start": split, "source_end": len(text)},
                ),
            ],
            words=words,
            extra={"input_start_seconds": input_start},
        )

    merged = _merge_chunk_results(
        [
            chunk("hello world", "hello ", "world", 6, 0.0),
            chunk("again now", "again ", "now", 6, 1.0),
        ],
        duration=2.0,
    )

    assert merged.text == "hello world again now"
    assert merged.segments is not None
    first, *rest = merged.segments
    composed = first.text + "".join(segment.text_separator + segment.text for segment in rest)
    assert composed == merged.text
    for segment in merged.segments:
        start = segment.extra["source_start"]
        end = segment.extra["source_end"]
        assert merged.text[start:end] == segment.text
    assert merged.words is not None
    for word in merged.words:
        start = word.extra["source_start"]
        end = word.extra["source_end"]
        assert merged.text[start:end] == word.text
    assert [item["input_start_seconds"] for item in merged.extra["windows"]] == [0.0, 1.0]
