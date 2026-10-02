"""Explicit lifecycle and inference ownership for optional CPU speech models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    ConfigError,
    UnsupportedFeatureError,
)
from standard_asr.engine import (
    ArtifactAction,
    ArtifactContext,
    ArtifactProgress,
    ArtifactRequirement,
    Diagnostic,
    RuntimeParams,
    TranscriptionResult,
    allow_downloads,
)

from .alignment import (
    ALIGNMENT_MODEL_REVISION,
    SUPPORTED_ALIGNMENT_LANGUAGES,
    ForcedAligner,
    acquire_forced_aligner,
    inspect_forced_aligner,
)
from .diarization import (
    DIARIZATION_ARTIFACTS,
    DiarizationConfig,
    OfflineSpeakerDiarizer,
    SpeakerTracker,
    diarization_artifact_status,
)
from .errors import CancellationToken, raise_if_cancelled

if TYPE_CHECKING:
    import numpy as np

    from .plugin import Qwen3ASRConfig

ALIGNMENT_ARTIFACT_ID = "qwen3-forced-aligner-0.6b"
DIARIZATION_ARTIFACT_ID = "speaker-diarization-cpu"
ProgressObserver = Callable[[ArtifactProgress], None] | None


class AuxiliaryModels:
    """Own lazy, serialized optional models without acquiring during inference."""

    def __init__(self, config: Qwen3ASRConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._aligner: ForcedAligner | None = None
        self._diarizer: OfflineSpeakerDiarizer | None = None

    @property
    def alignment_enabled(self) -> bool:
        return self.config.use_alignment or self.config.use_diarization

    def requirements(self, context: ArtifactContext) -> tuple[ArtifactRequirement, ...]:
        params = context.params
        needed = params.word_timestamps is not None or params.diarization is not None
        requirements = []
        if self.alignment_enabled:
            status = inspect_forced_aligner(self.config.alignment_dir)
            broken = status.model_state in {"corrupt", "incomplete"} or (
                not status.runtime.ready and status.runtime.reason != "missing"
            )
            requirements.append(
                self._requirement(
                    ALIGNMENT_ARTIFACT_ID,
                    "Qwen3 forced alignment model and isolated CPU runtime",
                    status.state,
                    self.config.alignment_dir,
                    needed,
                    ALIGNMENT_MODEL_REVISION,
                    broken=broken,
                )
            )
        if self.config.use_diarization:
            status = diarization_artifact_status(self.config.diarization_dir)
            state = (
                "ready"
                if status.ready
                else ("incomplete" if self.config.diarization_dir.exists() else "missing")
            )
            requirements.append(
                self._requirement(
                    DIARIZATION_ARTIFACT_ID,
                    "CPU speech segmentation and speaker embedding models",
                    state,
                    self.config.diarization_dir,
                    params.diarization is not None,
                    ":".join(item.revision for item in DIARIZATION_ARTIFACTS),
                    broken=state == "incomplete",
                )
            )
        return tuple(requirements)

    @staticmethod
    def _requirement(artifact_id, label, state, root, required, revision, *, broken):
        actions = ()
        blocker = None
        if state != "ready":
            if broken:
                blocker = "action_required"
                actions = (
                    ArtifactAction(
                        kind="provide_artifacts",
                        message=f"Move the incompatible auxiliary directory away and pull again: {root}",
                    ),
                )
            elif not allow_downloads():
                blocker = "downloads_disabled"
        return ArtifactRequirement(
            artifact_id=artifact_id,
            label=label,
            state=state,
            location=root.expanduser().absolute(),
            required_for_inference=required,
            can_acquire_now=state != "ready" and blocker is None,
            may_acquire_during_inference=False,
            source_is_mutable=False,
            acquisition_blocker=blocker,
            required_actions=actions,
            artifact_version=revision,
        )

    def acquire(self, targets: set[str], progress: ProgressObserver) -> None:
        if ALIGNMENT_ARTIFACT_ID in targets:

            def alignment_progress(message: str) -> None:
                phase = "transferring" if "model" in message.lower() else "resolving"
                _emit(progress, phase, ALIGNMENT_ARTIFACT_ID)

            acquire_forced_aligner(
                self.config.alignment_dir,
                allow_downloads=allow_downloads(),
                progress=alignment_progress,
            )
        if DIARIZATION_ARTIFACT_ID in targets:
            _acquire_diarization(self.config.diarization_dir, progress)

    def validate_request(self, params: RuntimeParams) -> None:
        if params.word_timestamps is None and params.diarization is None:
            return
        if not self.alignment_enabled:
            raise ConfigError("Enable use_alignment before requesting timestamps.")
        if params.diarization is not None:
            if not self.config.use_diarization:
                raise ConfigError("Enable use_diarization before requesting speaker labels.")
            if importlib.util.find_spec("sherpa_onnx") is None:
                raise ConfigError(
                    "Speaker diarization needs the diarization extra.",
                    hint="Install std-qwen3asr-ane[diarization] in the engine's environment.",
                )
        if params.language not in (None, "auto"):
            self._check_language(
                params.language,
                param=("word_timestamps" if params.word_timestamps is not None else "diarization"),
            )

    @staticmethod
    def _check_language(language: str | None, *, param: str) -> str:
        if (
            language is None
            or language.split("-", 1)[0].casefold() not in SUPPORTED_ALIGNMENT_LANGUAGES
        ):
            raise UnsupportedFeatureError(
                "The auxiliary aligner cannot align this detected or requested language.",
                param=param,
                hint="Use one of the documented alignment languages or transcribe without auxiliary output.",
            )
        return language.split("-", 1)[0].casefold()

    def new_speaker_tracker(self) -> SpeakerTracker:
        with self._lock:
            return SpeakerTracker.from_config(DiarizationConfig(self.config.diarization_dir))

    def annotate(
        self,
        result: TranscriptionResult,
        samples: np.ndarray,
        params: RuntimeParams,
        offset_seconds: float,
        *,
        cancel: CancellationToken | None = None,
        speaker_tracker: SpeakerTracker | None = None,
    ) -> TranscriptionResult:
        """Apply genuine measured alignment and optional speaker evidence to final text."""
        if params.word_timestamps is None and params.diarization is None:
            return result
        self.validate_request(params)
        if not result.text.strip():
            return result
        language = self._check_language(
            result.detected_language if params.language in (None, "auto") else params.language,
            param=("word_timestamps" if params.word_timestamps is not None else "diarization"),
        )
        from .postprocessing import annotate_result

        with self._lock:
            raise_if_cancelled(cancel)
            if self._aligner is None:
                inspection = inspect_forced_aligner(self.config.alignment_dir)
                if inspection.state != "ready":
                    raise ArtifactUnavailableError(
                        "The configured forced aligner is not ready. Run standard-asr pull first.",
                        reason="action_required",
                    )
                self._aligner = ForcedAligner(self.config.alignment_dir)
            requested_granularity = getattr(params.word_timestamps, "value", None)
            alignment_granularity = "char" if requested_granularity == "char" else "word"
            spans = self._aligner.align(
                samples,
                result.text,
                language,
                granularity=alignment_granularity,
                cancelled=cancel,
            )
            turns = None
            if params.diarization is not None:
                if self._diarizer is None:
                    self._diarizer = OfflineSpeakerDiarizer.from_config(
                        DiarizationConfig(self.config.diarization_dir)
                    )
                raise_if_cancelled(cancel)
                measurement = self._diarizer.measure(samples)
                turns = list(measurement.turns)
                result.extra["std_qwen3asr_ane_diarization_measurement"] = {
                    "input_start_seconds": offset_seconds,
                    "time_reference": "window_relative",
                    **measurement.evidence(),
                }
                if measurement.boundary_adjustments:
                    result.diagnostics.append(
                        Diagnostic(
                            code="diarization_boundary_adjusted",
                            message="Diarization frame padding was intersected with the input window.",
                            provided={"adjusted_turns": len(measurement.boundary_adjustments)},
                            effective="input_window_bounds",
                        )
                    )
                raise_if_cancelled(cancel)
                if speaker_tracker is not None:
                    turns = speaker_tracker.track_window(
                        samples, turns, window_start=offset_seconds
                    )
                else:
                    from .diarization import SpeakerTurn

                    turns = [
                        SpeakerTurn(t.start + offset_seconds, t.end + offset_seconds, t.speaker)
                        for t in turns
                    ]
            return annotate_result(
                result,
                spans,
                offset_seconds=offset_seconds,
                speaker_turns=turns,
                # Diarization needs alignment internally, but it does not grant
                # unrequested word details. Segment output carries speaker turns
                # without populating TranscriptionResult.words.
                granularity=requested_granularity or "segment",
            )

    def close(self) -> None:
        with self._lock:
            if self._aligner is not None:
                self._aligner.close()
                self._aligner = None
            self._diarizer = None


def _emit(progress: ProgressObserver, phase: str, artifact_id: str, **fields) -> None:
    if progress is not None:
        progress(ArtifactProgress(phase=phase, artifact_id=artifact_id, **fields))


def _acquire_diarization(root: Path, progress: ProgressObserver) -> None:
    root = root.expanduser().resolve()
    if diarization_artifact_status(root).ready:
        return
    if root.exists():
        raise ArtifactAcquisitionError(
            "The diarization directory is incomplete; choose a new directory before pulling.",
            reason="action_required",
        )
    if not allow_downloads():
        raise ArtifactAcquisitionError(
            "Diarization model downloads are disabled.", reason="downloads_disabled"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
        staged = Path(temporary) / "models"
        staged.mkdir()
        total = sum(item.size for item in DIARIZATION_ARTIFACTS)
        completed = 0
        for item in DIARIZATION_ARTIFACTS:
            if not allow_downloads():
                raise ArtifactAcquisitionError(
                    "Diarization model downloads are disabled.", reason="downloads_disabled"
                )
            digest = hashlib.sha256()
            with (
                urllib.request.urlopen(item.url, timeout=60) as response,
                (staged / item.filename).open("wb") as output,
            ):
                while data := response.read(1024 * 1024):
                    output.write(data)
                    digest.update(data)
                    completed += len(data)
                    _emit(
                        progress,
                        "transferring",
                        DIARIZATION_ARTIFACT_ID,
                        completed_units=completed,
                        total_units=total,
                        unit="bytes",
                    )
            if (
                digest.hexdigest() != item.sha256
                or (staged / item.filename).stat().st_size != item.size
            ):
                raise ArtifactAcquisitionError(
                    "A downloaded diarization model failed integrity verification.", reason="failed"
                )
        (staged / "sources.json").write_text(
            json.dumps(
                [
                    {
                        "filename": item.filename,
                        "revision": item.revision,
                        "sha256": item.sha256,
                        "license": item.license,
                    }
                    for item in DIARIZATION_ARTIFACTS
                ],
                indent=2,
            )
            + "\n"
        )
        staged.rename(root)
