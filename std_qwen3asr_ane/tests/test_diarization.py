"""Focused tests for optional model-backed speaker diarization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import std_qwen3asr_ane.diarization as diarization_module
from std_qwen3asr_ane.diarization import (
    DIARIZATION_ARTIFACTS,
    DIARIZATION_MODEL_DIR_ENV,
    EMBEDDING_ARTIFACT,
    PYANNOTE_BOUNDARY_SUPPORT_SAMPLES,
    SEGMENTATION_ARTIFACT,
    DiarizationArtifact,
    DiarizationConfig,
    OfflineSpeakerDiarizer,
    SpeakerTracker,
    SpeakerTrackingCapacityError,
    SpeakerTurn,
    diarization_artifact_status,
)


@dataclass
class RawSegment:
    start: float
    end: float
    speaker: int


class FakeBackend:
    sample_rate = 16000

    def __init__(self, segments=()):
        self.segments = list(segments)
        self.inputs: list[np.ndarray] = []

    def process(self, samples: np.ndarray):
        self.inputs.append(np.array(samples, copy=True))
        return self.segments


class QueueEmbeddingBackend:
    def __init__(self, embeddings):
        self.embeddings = [np.asarray(value, dtype=np.float32) for value in embeddings]
        self.inputs: list[tuple[np.ndarray, int]] = []

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.inputs.append((np.array(samples, copy=True), sample_rate))
        return self.embeddings.pop(0)


def test_artifact_specs_are_pinned_public_files():
    assert DIARIZATION_ARTIFACTS == (SEGMENTATION_ARTIFACT, EMBEDDING_ARTIFACT)
    assert SEGMENTATION_ARTIFACT.revision == "9403a6902bb58e3d5ae8c7e77c3422de279db2e0"
    assert SEGMENTATION_ARTIFACT.sha256 == (
        "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
    )
    assert SEGMENTATION_ARTIFACT.license == "MIT"
    assert EMBEDDING_ARTIFACT.revision == "8be2a75c9ed7a590538b268e46fbb65e1aa9d208"
    assert EMBEDDING_ARTIFACT.sha256 == (
        "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"
    )
    assert EMBEDDING_ARTIFACT.license == "Apache-2.0"
    assert all("/resolve/" in artifact.url for artifact in DIARIZATION_ARTIFACTS)


def test_artifact_status_checks_presence_size_and_hash(tmp_path, monkeypatch):
    payload = b"learned-model"
    artifact = DiarizationArtifact(
        filename="model.onnx",
        url="https://example.invalid/model.onnx",
        revision="revision",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        license="MIT",
    )
    monkeypatch.setattr(diarization_module, "DIARIZATION_ARTIFACTS", (artifact,))

    assert diarization_artifact_status(tmp_path).errors == ("missing model.onnx",)
    (tmp_path / artifact.filename).write_bytes(b"short")
    assert "wrong size" in diarization_artifact_status(tmp_path).errors[0]
    (tmp_path / artifact.filename).write_bytes(b"wrong-content")
    same_size = DiarizationArtifact(
        filename=artifact.filename,
        url=artifact.url,
        revision=artifact.revision,
        sha256=artifact.sha256,
        size=len(b"wrong-content"),
        license=artifact.license,
    )
    monkeypatch.setattr(diarization_module, "DIARIZATION_ARTIFACTS", (same_size,))
    assert "wrong SHA-256" in diarization_artifact_status(tmp_path).errors[0]
    (tmp_path / artifact.filename).write_bytes(payload)
    monkeypatch.setattr(diarization_module, "DIARIZATION_ARTIFACTS", (artifact,))
    assert diarization_artifact_status(tmp_path).ready


def test_model_boundaries_and_overlaps_are_preserved_and_sorted():
    backend = FakeBackend(
        [
            RawSegment(2.0, 3.0, 1),
            RawSegment(0.5, 2.5, 0),
            RawSegment(2.0, 2.8, 0),
        ]
    )
    diarizer = OfflineSpeakerDiarizer(backend)

    turns = diarizer.diarize(np.zeros(4 * 16000, dtype=np.float32))

    assert turns == [
        SpeakerTurn(0.5, 2.5, "speaker_00"),
        SpeakerTurn(2.0, 2.8, "speaker_00"),
        SpeakerTurn(2.0, 3.0, "speaker_01"),
    ]
    assert turns[0].end > turns[2].start
    assert len(backend.inputs) == 1
    assert backend.inputs[0].dtype == np.float32


def test_expected_pyannote_frame_support_overshoot_is_clamped_with_raw_evidence():
    backend = FakeBackend([RawSegment(0.2, 1.03097, 7)])
    diarizer = OfflineSpeakerDiarizer(backend)

    measurement = diarizer.measure(np.zeros(16000, dtype=np.float32))

    assert PYANNOTE_BOUNDARY_SUPPORT_SAMPLES == 496
    assert measurement.turns == (SpeakerTurn(0.2, 1.0, "speaker_07"),)
    assert measurement.raw_turns[0].end == 1.03097
    assert measurement.raw_turns[0].speaker == "speaker_07"
    assert measurement.boundary_adjustments[0].kind == "clamped"
    assert measurement.boundary_adjustments[0].adjusted_end == 1.0
    assert measurement.evidence()["raw_speaker_turns"] == [
        {"start": 0.2, "end": 1.03097, "speaker": "speaker_07"}
    ]
    assert diarizer.diarize(np.zeros(16000, dtype=np.float32)) == [
        SpeakerTurn(0.2, 1.0, "speaker_07")
    ]


def test_pyannote_boundary_overflow_beyond_frame_support_is_rejected():
    tolerance = PYANNOTE_BOUNDARY_SUPPORT_SAMPLES / 16000
    backend = FakeBackend([RawSegment(0.2, 1.0 + tolerance + 1e-6, 0)])

    with pytest.raises(RuntimeError, match="frame support"):
        OfflineSpeakerDiarizer(backend).diarize(np.zeros(16000, dtype=np.float32))


def test_turn_wholly_inside_padded_tail_is_retained_only_as_raw_evidence():
    backend = FakeBackend([RawSegment(1.01, 1.02, 3)])

    measurement = OfflineSpeakerDiarizer(backend).measure(
        np.zeros(16000, dtype=np.float32)
    )

    assert measurement.turns == ()
    assert measurement.raw_turns[0].speaker == "speaker_03"
    assert measurement.boundary_adjustments[0].kind == "discarded_padding"


def test_non_16khz_audio_is_resampled_before_model_inference():
    backend = FakeBackend([RawSegment(0.1, 0.9, 0)])
    samples = np.linspace(-0.1, 0.1, 8000, dtype=np.float32)

    turns = OfflineSpeakerDiarizer(backend).diarize(samples, sample_rate=8000)

    assert turns == [SpeakerTurn(0.1, 0.9, "speaker_00")]
    assert backend.inputs[0].shape == (16000,)
    assert backend.inputs[0].flags.c_contiguous


def test_empty_audio_does_not_invoke_the_model():
    backend = FakeBackend()
    assert OfflineSpeakerDiarizer(backend).diarize(np.empty(0, dtype=np.float32)) == []
    assert backend.inputs == []


@pytest.mark.parametrize(
    "samples,sample_rate,message",
    [
        (np.zeros((2, 10), dtype=np.float32), 16000, "mono"),
        (np.full(10, np.nan, dtype=np.float32), 16000, "non-finite"),
        (np.zeros(10, dtype=np.float32), 0, "positive integer"),
    ],
)
def test_audio_validation(samples, sample_rate, message):
    with pytest.raises(ValueError, match=message):
        OfflineSpeakerDiarizer(FakeBackend()).diarize(samples, sample_rate)


@pytest.mark.parametrize(
    "segment,message",
    [
        (RawSegment(-0.1, 1.0, 0), "invalid interval"),
        (RawSegment(0.1, 1.1, 0), "beyond"),
        (RawSegment(0.1, 0.9, -1), "speaker index"),
    ],
)
def test_backend_output_is_fail_closed(segment, message):
    with pytest.raises(RuntimeError, match=message):
        OfflineSpeakerDiarizer(FakeBackend([segment])).diarize(
            np.zeros(16000, dtype=np.float32)
        )


def test_top_level_api_requires_explicit_model_directory(monkeypatch):
    monkeypatch.delenv(DIARIZATION_MODEL_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=DIARIZATION_MODEL_DIR_ENV):
        diarization_module.diarize(np.zeros(16000, dtype=np.float32))


def test_sherpa_configuration_is_cpu_only_and_passes_clustering_settings(tmp_path):
    calls = {}

    class ConfigObject:
        def __init__(self, kind, **kwargs):
            calls[kind] = kwargs
            self.kind = kind

        def validate(self):
            return True

    class NativeDiarizer:
        sample_rate = 16000

        def __init__(self, config):
            calls["diarizer"] = config

    sherpa = SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda **kwargs: ConfigObject(
            "pyannote", **kwargs
        ),
        OfflineSpeakerSegmentationModelConfig=lambda **kwargs: ConfigObject(
            "segmentation", **kwargs
        ),
        SpeakerEmbeddingExtractorConfig=lambda **kwargs: ConfigObject("embedding", **kwargs),
        FastClusteringConfig=lambda **kwargs: ConfigObject("clustering", **kwargs),
        OfflineSpeakerDiarizationConfig=lambda **kwargs: ConfigObject("native", **kwargs),
        OfflineSpeakerDiarization=NativeDiarizer,
    )
    config = DiarizationConfig(
        model_dir=tmp_path,
        num_threads=3,
        num_speakers=2,
        cluster_threshold=0.72,
    )

    backend = diarization_module._create_sherpa_backend(config, sherpa)

    assert backend.sample_rate == 16000
    assert calls["segmentation"]["provider"] == "cpu"
    assert calls["segmentation"]["num_threads"] == 3
    assert calls["embedding"]["provider"] == "cpu"
    assert calls["clustering"] == {"num_clusters": 2, "threshold": 0.72}


def test_tracker_keeps_identity_when_local_labels_change_between_windows():
    embeddings = QueueEmbeddingBackend(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.05, 0.99],
            [0.99, 0.05],
        ]
    )
    tracker = SpeakerTracker(embeddings, match_threshold=0.8, min_embedding_seconds=0.5)
    audio = np.zeros(40, dtype=np.float32)

    first = tracker.track_window(
        audio,
        [SpeakerTurn(0.0, 2.0, "local_a"), SpeakerTurn(2.0, 4.0, "local_b")],
        window_start=0,
        sample_rate=10,
    )
    second = tracker.track_window(
        audio,
        [SpeakerTurn(0.0, 2.0, "reset_0"), SpeakerTurn(2.0, 4.0, "reset_1")],
        window_start=4,
        sample_rate=10,
    )

    assert [turn.speaker for turn in first] == ["speaker_00", "speaker_01"]
    assert [turn.speaker for turn in second] == ["speaker_01", "speaker_00"]
    assert [(turn.start, turn.end) for turn in second] == [(4.0, 6.0), (6.0, 8.0)]
    assert tracker.stable_speaker_count == 2


def test_tracker_excludes_overlap_from_embeddings_but_retains_overlapping_turns():
    backend = QueueEmbeddingBackend([[1.0, 0.0], [0.0, 1.0]])
    tracker = SpeakerTracker(backend, min_embedding_seconds=0.5)
    audio = np.concatenate(
        [
            np.full(10, 1.0, dtype=np.float32),
            np.full(10, 9.0, dtype=np.float32),
            np.full(10, -1.0, dtype=np.float32),
        ]
    )

    turns = tracker.track_window(
        audio,
        [SpeakerTurn(0.0, 2.0, "a"), SpeakerTurn(1.0, 3.0, "b")],
        window_start=5.0,
        sample_rate=10,
    )

    np.testing.assert_array_equal(backend.inputs[0][0], np.ones(10, dtype=np.float32))
    np.testing.assert_array_equal(backend.inputs[1][0], -np.ones(10, dtype=np.float32))
    assert turns == [
        SpeakerTurn(5.0, 7.0, "speaker_00"),
        SpeakerTurn(6.0, 8.0, "speaker_01"),
    ]
    assert turns[0].end > turns[1].start


def test_tracker_enforces_one_to_one_mapping_within_a_window():
    backend = QueueEmbeddingBackend([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    tracker = SpeakerTracker(backend, match_threshold=0.9, min_embedding_seconds=0.5)
    tracker.track_window(
        np.zeros(10, dtype=np.float32),
        [SpeakerTurn(0.0, 1.0, "first")],
        window_start=0,
        sample_rate=10,
    )

    turns = tracker.track_window(
        np.zeros(20, dtype=np.float32),
        [SpeakerTurn(0.0, 1.0, "a"), SpeakerTurn(1.0, 2.0, "b")],
        window_start=1,
        sample_rate=10,
    )

    assert {turn.speaker for turn in turns} == {"speaker_00", "speaker_01"}


def test_tracker_marks_short_exclusive_speech_as_window_scoped_unresolved():
    backend = QueueEmbeddingBackend([])
    tracker = SpeakerTracker(backend, min_embedding_seconds=1.0)

    first = tracker.track_window(
        np.zeros(10, dtype=np.float32),
        [SpeakerTurn(0.0, 0.4, "local")],
        window_start=0,
        sample_rate=10,
    )
    second = tracker.track_window(
        np.zeros(10, dtype=np.float32),
        [SpeakerTurn(0.0, 0.4, "local")],
        window_start=1,
        sample_rate=10,
    )

    assert first[0].speaker == "unresolved_000000_local"
    assert second[0].speaker == "unresolved_000001_local"
    assert tracker.stable_speaker_count == 0


def test_tracker_fails_instead_of_reusing_an_identity_past_capacity():
    backend = QueueEmbeddingBackend([[1.0, 0.0], [0.0, 1.0]])
    tracker = SpeakerTracker(
        backend,
        match_threshold=0.9,
        min_embedding_seconds=0.5,
        max_speakers=1,
    )
    tracker.track_window(
        np.zeros(10, dtype=np.float32),
        [SpeakerTurn(0.0, 1.0, "a")],
        window_start=0,
        sample_rate=10,
    )

    with pytest.raises(SpeakerTrackingCapacityError, match="max_speakers=1"):
        tracker.track_window(
            np.zeros(10, dtype=np.float32),
            [SpeakerTurn(0.0, 1.0, "b")],
            window_start=1,
            sample_rate=10,
        )


def test_tracker_rejects_invalid_embeddings_and_nonmonotonic_windows():
    tracker = SpeakerTracker(
        QueueEmbeddingBackend([[0.0, 0.0]]), min_embedding_seconds=0.5
    )
    with pytest.raises(RuntimeError, match="zero vector"):
        tracker.track_window(
            np.zeros(10, dtype=np.float32),
            [SpeakerTurn(0.0, 1.0, "a")],
            window_start=2,
            sample_rate=10,
        )

    tracker = SpeakerTracker(QueueEmbeddingBackend([]))
    tracker.track_window(np.zeros(10, dtype=np.float32), [], window_start=2, sample_rate=10)
    with pytest.raises(ValueError, match="monotonic"):
        tracker.track_window(np.zeros(10, dtype=np.float32), [], window_start=1, sample_rate=10)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"num_threads": 0}, "num_threads"),
        ({"num_speakers": 0}, "num_speakers"),
        ({"cluster_threshold": 0}, "cluster_threshold"),
        ({"window_shift_ratio": 1.1}, "window_shift_ratio"),
        ({"min_duration_on": -1}, "min_duration_on"),
        ({"min_duration_off": -1}, "min_duration_off"),
    ],
)
def test_config_validation(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        DiarizationConfig(model_dir=tmp_path, **changes)
