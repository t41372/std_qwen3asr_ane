"""Optional, model-backed CPU speaker diarization.

This module deliberately keeps model acquisition separate from inference.  It
uses sherpa-onnx's offline diarization pipeline: a pyannote segmentation model,
a learned 3D-Speaker embedding model, and clustering.  Callers provide the two
verified ONNX files; importing or running this module never downloads them.

Overlapping speech is represented by overlapping :class:`SpeakerTurn` values.
No turn is trimmed, merged, or assigned to a single "dominant" speaker.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from math import ceil, floor, gcd, isfinite
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from scipy.signal import resample_poly

DIARIZATION_MODEL_DIR_ENV = "STD_QWEN3ASR_ANE_DIARIZATION_MODEL_DIR"

# The pinned pyannote segmentation model records these values in ONNX metadata.
# sherpa-onnx 1.13.8 converts a frame index to time by adding half the receptive
# field. Its final frame can therefore end just beyond the unpadded input. Use
# the integer sample support radius as the only permitted boundary adjustment.
PYANNOTE_RECEPTIVE_FIELD_SIZE_SAMPLES = 991
PYANNOTE_BOUNDARY_SUPPORT_SAMPLES = ceil(PYANNOTE_RECEPTIVE_FIELD_SIZE_SAMPLES / 2)


@dataclass(frozen=True)
class DiarizationArtifact:
    """One immutable public model file required by diarization."""

    filename: str
    url: str
    revision: str
    sha256: str
    size: int
    license: str


SEGMENTATION_ARTIFACT = DiarizationArtifact(
    filename="pyannote-segmentation-3.0.onnx",
    url=(
        "https://huggingface.co/csukuangfj/"
        "sherpa-onnx-pyannote-segmentation-3-0/resolve/"
        "9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx?download=true"
    ),
    revision="9403a6902bb58e3d5ae8c7e77c3422de279db2e0",
    sha256="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    size=5_992_913,
    license="MIT",
)
EMBEDDING_ARTIFACT = DiarizationArtifact(
    filename="3dspeaker-eres2net-base-16k.onnx",
    url=(
        "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/"
        "8be2a75c9ed7a590538b268e46fbb65e1aa9d208/"
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx?download=true"
    ),
    revision="8be2a75c9ed7a590538b268e46fbb65e1aa9d208",
    sha256="1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b",
    size=39_593_761,
    license="Apache-2.0",
)
DIARIZATION_ARTIFACTS = (SEGMENTATION_ARTIFACT, EMBEDDING_ARTIFACT)


@dataclass(frozen=True)
class DiarizationArtifactStatus:
    """Read-only validation result for an explicitly populated model directory."""

    directory: Path
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class DiarizationConfig:
    """Configuration for sherpa-onnx's CPU offline diarization pipeline."""

    model_dir: Path
    num_threads: int = 1
    num_speakers: int = -1
    cluster_threshold: float = 0.5
    window_shift_ratio: float = 0.1
    min_duration_on: float = 0.3
    min_duration_off: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_dir", Path(self.model_dir).expanduser().resolve())
        if type(self.num_threads) is not int or self.num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        if type(self.num_speakers) is not int or self.num_speakers == 0 or self.num_speakers < -1:
            raise ValueError("num_speakers must be -1 for automatic clustering or a positive integer")
        if not isfinite(self.cluster_threshold) or self.cluster_threshold <= 0:
            raise ValueError("cluster_threshold must be finite and positive")
        if not isfinite(self.window_shift_ratio) or not 0 < self.window_shift_ratio <= 1:
            raise ValueError("window_shift_ratio must be within (0, 1]")
        if not isfinite(self.min_duration_on) or self.min_duration_on < 0:
            raise ValueError("min_duration_on must be finite and non-negative")
        if not isfinite(self.min_duration_off) or self.min_duration_off < 0:
            raise ValueError("min_duration_off must be finite and non-negative")

    @property
    def segmentation_model(self) -> Path:
        return self.model_dir / SEGMENTATION_ARTIFACT.filename

    @property
    def embedding_model(self) -> Path:
        return self.model_dir / EMBEDDING_ARTIFACT.filename


@dataclass(frozen=True, order=True)
class SpeakerTurn:
    """One measured half-open speaker interval, in seconds.

    Different speakers may have intersecting intervals when the segmentation
    model detects overlapping speech.
    """

    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if not isfinite(self.start) or not isfinite(self.end) or self.start < 0 or self.end <= self.start:
            raise ValueError("SpeakerTurn requires a finite, positive-duration interval")
        if not self.speaker or self.speaker != self.speaker.strip():
            raise ValueError("SpeakerTurn requires a nonempty, whitespace-normalized label")


@dataclass(frozen=True, order=True)
class RawSpeakerTurn:
    """One unmodified model turn, retained even when it touches model padding."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DiarizationBoundaryAdjustment:
    """A bounded intersection between one raw model turn and actual input."""

    raw_start: float
    raw_end: float
    adjusted_start: float | None
    adjusted_end: float | None
    speaker: str
    kind: Literal["clamped", "discarded_padding"]


@dataclass(frozen=True)
class DiarizationMeasurement:
    """Adjusted turns plus raw model evidence for diagnostics and result extras."""

    turns: tuple[SpeakerTurn, ...]
    raw_turns: tuple[RawSpeakerTurn, ...]
    boundary_adjustments: tuple[DiarizationBoundaryAdjustment, ...]

    def evidence(self) -> dict[str, list[dict[str, float | str | None]]]:
        """Return JSON-ready raw/adjusted records for a Standard result ``extra``."""

        return {
            "raw_speaker_turns": [
                {"start": turn.start, "end": turn.end, "speaker": turn.speaker}
                for turn in self.raw_turns
            ],
            "speaker_boundary_adjustments": [
                {
                    "raw_start": item.raw_start,
                    "raw_end": item.raw_end,
                    "adjusted_start": item.adjusted_start,
                    "adjusted_end": item.adjusted_end,
                    "speaker": item.speaker,
                    "kind": item.kind,
                }
                for item in self.boundary_adjustments
            ],
        }


class _BackendSegment(Protocol):
    start: float
    end: float
    speaker: int


class DiarizationBackend(Protocol):
    """Small boundary around a learned diarization runtime."""

    sample_rate: int

    def process(self, samples: np.ndarray) -> Iterable[_BackendSegment]: ...


class SpeakerEmbeddingBackend(Protocol):
    """Learned voice embedding boundary used for cross-window identity."""

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


def diarization_artifact_status(model_dir: str | Path) -> DiarizationArtifactStatus:
    """Validate required local model files without importing the runtime."""

    directory = Path(model_dir).expanduser().resolve()
    errors: list[str] = []
    for artifact in DIARIZATION_ARTIFACTS:
        path = directory / artifact.filename
        try:
            stat = path.stat()
        except FileNotFoundError:
            errors.append(f"missing {artifact.filename}")
            continue
        except OSError as exc:
            errors.append(f"cannot inspect {artifact.filename}: {exc}")
            continue
        if not path.is_file():
            errors.append(f"not a regular file: {artifact.filename}")
            continue
        if stat.st_size != artifact.size:
            errors.append(
                f"wrong size for {artifact.filename}: expected {artifact.size}, got {stat.st_size}"
            )
            continue
        digest = _sha256(path)
        if digest != artifact.sha256:
            errors.append(
                f"wrong SHA-256 for {artifact.filename}: expected {artifact.sha256}, got {digest}"
            )
    return DiarizationArtifactStatus(directory=directory, errors=tuple(errors))


class OfflineSpeakerDiarizer:
    """Run a genuine learned diarization backend and preserve its measured turns."""

    def __init__(self, backend: DiarizationBackend):
        if type(backend.sample_rate) is not int or backend.sample_rate < 1:
            raise ValueError("The diarization backend must expose a positive sample rate")
        self._backend = backend

    @classmethod
    def from_config(cls, config: DiarizationConfig) -> OfflineSpeakerDiarizer:
        """Load sherpa-onnx after validating the explicit model artifacts."""

        status = diarization_artifact_status(config.model_dir)
        if not status.ready:
            details = "; ".join(status.errors)
            raise FileNotFoundError(
                f"Diarization models are not ready in {status.directory}: {details}. "
                "Acquire the pinned DIARIZATION_ARTIFACTS explicitly before inference."
            )
        try:
            sherpa_onnx = import_module("sherpa_onnx")
        except ImportError as exc:
            raise RuntimeError(
                "Optional speaker diarization requires sherpa-onnx>=1.10.28. "
                "Install the diarization extra before enabling it."
            ) from exc
        return cls(_create_sherpa_backend(config, sherpa_onnx))

    def diarize(self, samples: np.ndarray, sample_rate: int = 16000) -> list[SpeakerTurn]:
        """Return model-measured speaker turns, including overlapping turns."""

        return list(self.measure(samples, sample_rate).turns)

    def measure(
        self, samples: np.ndarray, sample_rate: int = 16000
    ) -> DiarizationMeasurement:
        """Return adjusted turns and the unmodified model boundary evidence."""

        values = _validate_samples(samples, sample_rate)
        if not values.size:
            return DiarizationMeasurement((), (), ())
        original_duration = values.size / sample_rate
        model_samples = _resample(values, sample_rate, self._backend.sample_rate)
        raw_segments = self._backend.process(model_samples)
        turns: list[SpeakerTurn] = []
        raw_turns: list[RawSpeakerTurn] = []
        adjustments: list[DiarizationBoundaryAdjustment] = []
        for segment in raw_segments:
            raw, adjusted, adjustment = _speaker_turn(
                segment,
                original_duration,
                model_sample_rate=self._backend.sample_rate,
            )
            raw_turns.append(raw)
            if adjusted is not None:
                turns.append(adjusted)
            if adjustment is not None:
                adjustments.append(adjustment)
        return DiarizationMeasurement(
            turns=tuple(sorted(turns)),
            raw_turns=tuple(sorted(raw_turns)),
            boundary_adjustments=tuple(
                sorted(
                    adjustments,
                    key=lambda item: (item.raw_start, item.raw_end, item.speaker),
                )
            ),
        )


class _SherpaBackend:
    def __init__(self, diarizer: Any):
        self._diarizer = diarizer
        self.sample_rate = int(diarizer.sample_rate)

    def process(self, samples: np.ndarray) -> Iterable[_BackendSegment]:
        return self._diarizer.process(samples).sort_by_start_time()


class _SherpaEmbeddingBackend:
    def __init__(self, extractor: Any):
        self._extractor = extractor

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise RuntimeError("Speaker embedding model rejected the available speech samples")
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)


class SpeakerTrackingCapacityError(RuntimeError):
    """A bounded tracker cannot register another measured speaker identity."""


@dataclass
class _SpeakerCentroid:
    embedding: np.ndarray
    speech_seconds: float


class SpeakerTracker:
    """Track learned speaker identities across independently diarized windows.

    Local labels are matched by cosine similarity of learned voice embeddings.
    Audio attributed to simultaneous speakers is retained in returned turns but
    excluded from identity embeddings to avoid knowingly mixing two voices.
    A local speaker with too little exclusive speech receives a unique
    ``unresolved_*`` label; that label makes no cross-window identity claim.
    """

    def __init__(
        self,
        embedding_backend: SpeakerEmbeddingBackend,
        *,
        match_threshold: float = 0.6,
        min_embedding_seconds: float = 1.0,
        max_speakers: int = 32,
    ) -> None:
        if not isfinite(match_threshold) or not -1 <= match_threshold <= 1:
            raise ValueError("match_threshold must be finite and within [-1, 1]")
        if not isfinite(min_embedding_seconds) or min_embedding_seconds <= 0:
            raise ValueError("min_embedding_seconds must be finite and positive")
        if type(max_speakers) is not int or max_speakers < 1:
            raise ValueError("max_speakers must be a positive integer")
        self._embedding_backend = embedding_backend
        self.match_threshold = match_threshold
        self.min_embedding_seconds = min_embedding_seconds
        self.max_speakers = max_speakers
        self._centroids: dict[str, _SpeakerCentroid] = {}
        self._next_speaker_index = 0
        self._window_index = 0
        self._last_window_start = 0.0

    @classmethod
    def from_config(
        cls,
        config: DiarizationConfig,
        *,
        match_threshold: float = 0.6,
        min_embedding_seconds: float = 1.0,
        max_speakers: int = 32,
    ) -> SpeakerTracker:
        """Build a bounded tracker from the pinned local embedding artifact."""

        status = diarization_artifact_status(config.model_dir)
        if not status.ready:
            details = "; ".join(status.errors)
            raise FileNotFoundError(
                f"Diarization models are not ready in {status.directory}: {details}. "
                "Acquire the pinned DIARIZATION_ARTIFACTS explicitly before inference."
            )
        try:
            sherpa_onnx = import_module("sherpa_onnx")
        except ImportError as exc:
            raise RuntimeError(
                "Optional speaker tracking requires sherpa-onnx>=1.10.28. "
                "Install the diarization extra before enabling it."
            ) from exc
        native_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(config.embedding_model),
            num_threads=config.num_threads,
            debug=False,
            provider="cpu",
        )
        if not native_config.validate():
            raise RuntimeError("sherpa-onnx rejected the speaker embedding configuration")
        backend = _SherpaEmbeddingBackend(sherpa_onnx.SpeakerEmbeddingExtractor(native_config))
        return cls(
            backend,
            match_threshold=match_threshold,
            min_embedding_seconds=min_embedding_seconds,
            max_speakers=max_speakers,
        )

    @property
    def stable_speaker_count(self) -> int:
        return len(self._centroids)

    def track_window(
        self,
        samples: np.ndarray,
        turns: Iterable[SpeakerTurn],
        *,
        window_start: float,
        sample_rate: int = 16000,
    ) -> list[SpeakerTurn]:
        """Map window-local turns to bounded global identities.

        ``turns`` use time relative to ``samples``.  Returned turns use absolute
        recording time after adding ``window_start``.
        """

        if not isfinite(window_start) or window_start < self._last_window_start:
            raise ValueError("window_start must be finite, non-negative, and monotonic")
        values = _validate_samples(samples, sample_rate)
        duration = values.size / sample_rate
        local_turns = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
        for turn in local_turns:
            if turn.end > duration:
                raise ValueError(
                    f"Speaker turn ends at {turn.end:.6f}s beyond its {duration:.6f}s window"
                )
        local_labels = list(dict.fromkeys(turn.speaker for turn in local_turns))
        candidates: dict[str, tuple[np.ndarray, float]] = {}
        for label in local_labels:
            exclusive = _exclusive_speaker_audio(values, local_turns, label, sample_rate)
            speech_seconds = exclusive.size / sample_rate
            if speech_seconds < self.min_embedding_seconds:
                continue
            embedding = _normalize_embedding(
                self._embedding_backend.embed(exclusive, sample_rate)
            )
            candidates[label] = (embedding, speech_seconds)

        assignments = self._match_candidates(candidates)
        for label in local_labels:
            assignments.setdefault(
                label,
                f"unresolved_{self._window_index:06d}_{label}",
            )
        result = [
            SpeakerTurn(
                start=window_start + turn.start,
                end=window_start + turn.end,
                speaker=assignments[turn.speaker],
            )
            for turn in local_turns
        ]
        self._window_index += 1
        self._last_window_start = window_start
        return result

    def _match_candidates(
        self, candidates: dict[str, tuple[np.ndarray, float]]
    ) -> dict[str, str]:
        assignments: dict[str, str] = {}
        available_local = set(candidates)
        available_global = set(self._centroids)
        scored_pairs: list[tuple[float, str, str]] = []
        for local_label, (embedding, _) in candidates.items():
            for global_label, centroid in self._centroids.items():
                score = float(np.dot(embedding, centroid.embedding))
                if score >= self.match_threshold:
                    scored_pairs.append((score, local_label, global_label))
        for _, local_label, global_label in sorted(
            scored_pairs, key=lambda item: (-item[0], item[1], item[2])
        ):
            if local_label in available_local and global_label in available_global:
                assignments[local_label] = global_label
                available_local.remove(local_label)
                available_global.remove(global_label)

        new_count = len(available_local)
        if len(self._centroids) + new_count > self.max_speakers:
            raise SpeakerTrackingCapacityError(
                f"Speaker tracker needs {len(self._centroids) + new_count} identities, "
                f"exceeding max_speakers={self.max_speakers}"
            )
        for local_label in sorted(available_local):
            global_label = f"speaker_{self._next_speaker_index:02d}"
            self._next_speaker_index += 1
            assignments[local_label] = global_label

        for local_label, global_label in assignments.items():
            embedding, speech_seconds = candidates[local_label]
            old = self._centroids.get(global_label)
            if old is None:
                self._centroids[global_label] = _SpeakerCentroid(embedding, speech_seconds)
                continue
            combined = (
                old.embedding * old.speech_seconds + embedding * speech_seconds
            )
            self._centroids[global_label] = _SpeakerCentroid(
                _normalize_embedding(combined),
                old.speech_seconds + speech_seconds,
            )
        return assignments


def diarize(samples: np.ndarray, sample_rate: int = 16000) -> list[SpeakerTurn]:
    """Diarize with models from ``STD_QWEN3ASR_ANE_DIARIZATION_MODEL_DIR``.

    The environment variable names an already populated directory.  This
    function never downloads models.  Applications that need per-request
    settings should retain an :class:`OfflineSpeakerDiarizer` instead.
    """

    model_dir = os.environ.get(DIARIZATION_MODEL_DIR_ENV)
    if not model_dir:
        raise RuntimeError(
            f"Set {DIARIZATION_MODEL_DIR_ENV} to an explicitly acquired model directory, "
            "or construct OfflineSpeakerDiarizer.from_config(...)."
        )
    return _default_diarizer(str(Path(model_dir).expanduser().resolve())).diarize(
        samples, sample_rate
    )


@lru_cache(maxsize=4)
def _default_diarizer(model_dir: str) -> OfflineSpeakerDiarizer:
    return OfflineSpeakerDiarizer.from_config(DiarizationConfig(model_dir=Path(model_dir)))


def _create_sherpa_backend(config: DiarizationConfig, sherpa_onnx: Any) -> _SherpaBackend:
    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(config.segmentation_model),
            window_shift_ratio=config.window_shift_ratio,
        ),
        num_threads=config.num_threads,
        debug=False,
        provider="cpu",
    )
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(config.embedding_model),
        num_threads=config.num_threads,
        debug=False,
        provider="cpu",
    )
    clustering = sherpa_onnx.FastClusteringConfig(
        num_clusters=config.num_speakers,
        threshold=config.cluster_threshold,
    )
    native_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=clustering,
        min_duration_on=config.min_duration_on,
        min_duration_off=config.min_duration_off,
    )
    if not native_config.validate():
        raise RuntimeError("sherpa-onnx rejected the validated diarization configuration")
    return _SherpaBackend(sherpa_onnx.OfflineSpeakerDiarization(native_config))


def _validate_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if type(sample_rate) is not int or sample_rate < 1:
        raise ValueError("sample_rate must be a positive integer")
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Diarization requires mono audio")
    if not np.isfinite(values).all():
        raise ValueError("Diarization audio contains non-finite samples")
    return np.ascontiguousarray(values)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    common = gcd(source_rate, target_rate)
    result = resample_poly(samples, target_rate // common, source_rate // common)
    return np.ascontiguousarray(result, dtype=np.float32)


def _speaker_turn(
    segment: _BackendSegment,
    audio_duration: float,
    *,
    model_sample_rate: int,
) -> tuple[
    RawSpeakerTurn,
    SpeakerTurn | None,
    DiarizationBoundaryAdjustment | None,
]:
    start, end = float(segment.start), float(segment.end)
    speaker = segment.speaker
    if type(speaker) is not int or speaker < 0:
        raise RuntimeError(f"Diarization backend returned an invalid speaker index: {speaker!r}")
    if not isfinite(start) or not isfinite(end) or start < 0 or end <= start:
        raise RuntimeError(f"Diarization backend returned an invalid interval: {start!r}--{end!r}")
    label = f"speaker_{speaker:02d}"
    raw = RawSpeakerTurn(start=start, end=end, speaker=label)
    tolerance = PYANNOTE_BOUNDARY_SUPPORT_SAMPLES / model_sample_rate
    if start > audio_duration + tolerance or end > audio_duration + tolerance:
        raise RuntimeError(
            f"Diarization backend returned {start:.6f}--{end:.6f}s beyond "
            f"{audio_duration:.6f}s audio and its {tolerance:.6f}s frame support"
        )
    adjusted_start = min(max(start, 0.0), audio_duration)
    adjusted_end = min(max(end, 0.0), audio_duration)
    if adjusted_end <= adjusted_start:
        return raw, None, DiarizationBoundaryAdjustment(
            raw_start=start,
            raw_end=end,
            adjusted_start=None,
            adjusted_end=None,
            speaker=label,
            kind="discarded_padding",
        )
    adjusted = SpeakerTurn(start=adjusted_start, end=adjusted_end, speaker=label)
    if adjusted_start == start and adjusted_end == end:
        return raw, adjusted, None
    return raw, adjusted, DiarizationBoundaryAdjustment(
        raw_start=start,
        raw_end=end,
        adjusted_start=adjusted_start,
        adjusted_end=adjusted_end,
        speaker=label,
        kind="clamped",
    )


def _exclusive_speaker_audio(
    samples: np.ndarray,
    turns: list[SpeakerTurn],
    speaker: str,
    sample_rate: int,
) -> np.ndarray:
    own = _merge_intervals(
        (
            max(0, ceil(turn.start * sample_rate)),
            min(samples.size, floor(turn.end * sample_rate)),
        )
        for turn in turns
        if turn.speaker == speaker
    )
    other = _merge_intervals(
        (
            max(0, floor(turn.start * sample_rate)),
            min(samples.size, ceil(turn.end * sample_rate)),
        )
        for turn in turns
        if turn.speaker != speaker
    )
    exclusive = own
    for blocked in other:
        exclusive = [part for interval in exclusive for part in _subtract(interval, blocked)]
    chunks = [samples[start:end] for start, end in exclusive if end > start]
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(interval for interval in intervals if interval[1] > interval[0]):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _subtract(interval: tuple[int, int], blocked: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = interval
    blocked_start, blocked_end = blocked
    if blocked_end <= start or blocked_start >= end:
        return [interval]
    result = []
    if start < blocked_start:
        result.append((start, blocked_start))
    if blocked_end < end:
        result.append((blocked_end, end))
    return result


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise RuntimeError("Speaker embedding backend returned an invalid vector")
    norm = float(np.linalg.norm(values))
    if not isfinite(norm) or norm <= 0:
        raise RuntimeError("Speaker embedding backend returned a zero vector")
    return np.ascontiguousarray(values / norm, dtype=np.float32)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()
