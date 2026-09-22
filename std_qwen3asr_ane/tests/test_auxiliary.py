"""Measured auxiliary data and boundary repairs remain visible in standard results."""

from types import SimpleNamespace

import numpy as np
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
    assert result.words[0].start == 10.2
    assert result.words[0].end == 10.9
    assert result.words[0].speaker == "speaker_0"
    assert "diarization_boundary_adjusted" in [item.code for item in result.diagnostics]
    evidence = result.extra["std_qwen3asr_ane_diarization_measurement"]
    assert evidence["input_start_seconds"] == 10.0
    assert evidence["time_reference"] == "window_relative"
    assert evidence["raw_speaker_turns"][0]["end"] == 1.03097
    assert evidence["speaker_boundary_adjustments"][0]["adjusted_end"] == 1.0
