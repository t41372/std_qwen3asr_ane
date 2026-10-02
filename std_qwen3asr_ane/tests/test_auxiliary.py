"""Measured auxiliary data and boundary repairs remain visible in standard results."""

from types import SimpleNamespace

import numpy as np
import pytest
from standard_asr import UnsupportedFeatureError
from standard_asr.engine import DIARIZE, RuntimeParams, TranscriptionResult

from std_qwen3asr_ane.alignment import AlignmentSpan
from std_qwen3asr_ane.auxiliary import AuxiliaryModels
from std_qwen3asr_ane.diarization import (
    DiarizationBoundaryAdjustment,
    DiarizationMeasurement,
    RawSpeakerTurn,
    SpeakerTurn,
)


def test_frame_support_repair_is_disclosed_with_raw_local_evidence(monkeypatch):
    monkeypatch.setattr("std_qwen3asr_ane.auxiliary.importlib.util.find_spec", lambda _: object())
    owner = AuxiliaryModels(SimpleNamespace(use_alignment=True, use_diarization=True))
    owner._aligner = SimpleNamespace(
        align=lambda *args, **kwargs: [AlignmentSpan("hello", 0.2, 0.9, 0, 5)]
    )
    owner._diarizer = SimpleNamespace(
        measure=lambda samples: DiarizationMeasurement(
            turns=(SpeakerTurn(0.1, 1.0, "speaker_0"),),
            raw_turns=(RawSpeakerTurn(0.1, 1.03097, "speaker_0"),),
            boundary_adjustments=(
                DiarizationBoundaryAdjustment(0.1, 1.03097, 0.1, 1.0, "speaker_0", "clamped"),
            ),
        )
    )
    result = owner.annotate(
        TranscriptionResult(text="hello", detected_language="en", duration=1.0),
        np.zeros(16000, dtype=np.float32),
        RuntimeParams(language="auto", diarization=DIARIZE),
        10.0,
    )
    assert result.text == "hello"
    assert result.words is None
    assert result.segments is not None
    assert result.segments[0].start == 10.2
    assert result.segments[0].end == 10.9
    assert result.segments[0].speaker == "speaker_0"
    assert "diarization_boundary_adjusted" in [item.code for item in result.diagnostics]
    evidence = result.extra["std_qwen3asr_ane_diarization_measurement"]
    assert evidence["input_start_seconds"] == 10.0
    assert evidence["time_reference"] == "window_relative"
    assert evidence["raw_speaker_turns"][0]["end"] == 1.03097
    assert evidence["speaker_boundary_adjustments"][0]["adjusted_end"] == 1.0


@pytest.mark.parametrize(
    ("granularity", "with_diarization"),
    [
        ("word", False),
        ("word", True),
        ("char", False),
        ("char", True),
        ("segment", False),
        ("segment", True),
        (None, True),
    ],
)
def test_auxiliary_exposes_only_requested_timestamp_detail(
    monkeypatch, granularity, with_diarization
):
    monkeypatch.setattr("std_qwen3asr_ane.auxiliary.importlib.util.find_spec", lambda _: object())
    owner = AuxiliaryModels(SimpleNamespace(use_alignment=True, use_diarization=True))
    alignment_calls = []

    def align(samples, text, language, *, granularity, cancelled):
        alignment_calls.append(granularity)
        return [AlignmentSpan(text, 0.1, 0.9, 0, len(text))]

    owner._aligner = SimpleNamespace(align=align)
    owner._diarizer = SimpleNamespace(
        measure=lambda samples: DiarizationMeasurement(
            turns=(SpeakerTurn(0.0, 1.0, "speaker_0"),),
            raw_turns=(RawSpeakerTurn(0.0, 1.0, "speaker_0"),),
            boundary_adjustments=(),
        )
    )
    params = RuntimeParams(
        language="en",
        word_timestamps=granularity,
        diarization=DIARIZE if with_diarization else None,
    )

    annotated = owner.annotate(
        TranscriptionResult(text="hello", detected_language=None, duration=1.0),
        np.zeros(16000, dtype=np.float32),
        params,
        0.0,
    )

    assert alignment_calls == ["char" if granularity == "char" else "word"]
    assert annotated.segments is not None
    assert annotated.segments[0].speaker == ("speaker_0" if with_diarization else None)
    assert (annotated.words is not None) is (granularity in ("word", "char"))
    if annotated.words is not None:
        assert annotated.words[0].speaker == ("speaker_0" if with_diarization else None)


@pytest.mark.parametrize(
    ("timestamps", "diarization", "expected_param"),
    [
        ("word", None, "word_timestamps"),
        (None, DIARIZE, "diarization"),
        ("segment", DIARIZE, "word_timestamps"),
    ],
)
def test_forced_unsupported_alignment_language_names_the_active_channel(
    monkeypatch, timestamps, diarization, expected_param
):
    monkeypatch.setattr("std_qwen3asr_ane.auxiliary.importlib.util.find_spec", lambda _: object())
    owner = AuxiliaryModels(SimpleNamespace(use_alignment=True, use_diarization=True))
    params = RuntimeParams(
        language="zu",
        word_timestamps=timestamps,
        diarization=diarization,
    )

    with pytest.raises(UnsupportedFeatureError) as caught:
        owner.validate_request(params)

    assert caught.value.param == expected_param


@pytest.mark.parametrize(
    ("timestamps", "diarization", "expected_param"),
    [
        ("word", None, "word_timestamps"),
        (None, DIARIZE, "diarization"),
        ("segment", DIARIZE, "word_timestamps"),
    ],
)
def test_auto_detected_unsupported_alignment_language_names_the_active_channel(
    monkeypatch, timestamps, diarization, expected_param
):
    monkeypatch.setattr("std_qwen3asr_ane.auxiliary.importlib.util.find_spec", lambda _: object())
    owner = AuxiliaryModels(SimpleNamespace(use_alignment=True, use_diarization=True))
    params = RuntimeParams(
        language="auto",
        word_timestamps=timestamps,
        diarization=diarization,
    )

    with pytest.raises(UnsupportedFeatureError) as caught:
        owner.annotate(
            TranscriptionResult(text="sawubona", detected_language="zu", duration=1.0),
            np.zeros(16000, dtype=np.float32),
            params,
            0.0,
        )

    assert caught.value.param == expected_param
