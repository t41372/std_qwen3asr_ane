"""Standard ASR 0.2 batch and bounded streaming adapter for local Qwen3-ASR."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import Field
from standard_asr.contract.exceptions import ArtifactUnavailableError, TranscriptionError
from standard_asr.engine import (
    ArtifactAction,
    ArtifactContext,
    ArtifactDeclaration,
    ArtifactRequirement,
    AudioFormat,
    BaseConfig,
    BaseProperties,
    BatchCapabilities,
    DeclaredCapabilities,
    DeclaredEngineMetadata,
    Diagnostic,
    EngineBase,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    InputKind,
    LanguageCaps,
    LanguageConfigMixin,
    PreparedAudio,
    PromptCap,
    RuntimeParams,
    StreamingCapabilities,
    StreamingGuidanceCaps,
    TranscriptionResult,
    TranscriptionSession,
)

from .languages import LANGUAGE_NAMES, normalize_model_language

if TYPE_CHECKING:
    from .runtime import CoreMLRuntime


ENGINE_ID = "std-qwen3asr-ane"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
_REQUIRED_ROLES = {
    "frontend",
    "encoder",
    "decoder",
    "lm_head",
    "embedding",
    "tokenizer",
    "mel_filters",
}


class Qwen3ASRConfig(LanguageConfigMixin, BaseConfig[Literal["std-qwen3asr-ane"]]):
    """Local settings; a relative model directory is relative to the working directory."""

    engine: Literal["std-qwen3asr-ane"] = ENGINE_ID
    default_language: str = "auto"
    model_dir: Path = Field(
        default=Path("artifacts/qwen3-asr-1.7b"),
        description="Locally built model directory, relative to the current working directory.",
    )
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    stream_chunk_seconds: float = Field(default=2.0, ge=0.1, le=30.0)
    stream_unfixed_chunks: int = Field(default=2, ge=0)
    stream_unfixed_tokens: int = Field(default=5, ge=0)
    stream_max_audio_seconds: float = Field(default=30.0, gt=0, le=30.0)
    stream_audio_queue_size: int = Field(default=4, ge=1, le=128)


class Qwen3ASREngine(EngineBase):
    """A lazy, serialized batch engine with an externally built artifact lifecycle."""

    config_type: ClassVar[type[BaseConfig]] = Qwen3ASRConfig
    properties: ClassVar[BaseProperties] = BaseProperties(
        engine_id=ENGINE_ID,
        model_name="1.7b",
        protocol_version="0.2.0",
        accepted_input={InputKind.ARRAY},
        native_sample_rate=16000,
        accepted_sample_rates=[16000],
        required_input_sample_rate=16000,
        wire_encodings=["pcm_s16le", "pcm_f32le"],
        selectable_languages=[*LANGUAGE_NAMES, "auto"],
        detectable_languages=list(LANGUAGE_NAMES),
        description="Qwen3-ASR 1.7B using locally converted Core ML models targeting ANE.",
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
            guidance=GuidanceCaps(prompt=PromptCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
        streaming=StreamingCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
            guidance=StreamingGuidanceCaps(prompt=PromptCap(supported=True)),
            emits_partials=FlagCap(supported=True),
            finality_level=FinalityCap(mode="closed"),
        ),
    )
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=ArtifactDeclaration(
            applicable=True,
            supports_explicit_acquisition=False,
            may_acquire_during_inference=False,
        ),
        x_qwen3asr_streaming={
            "algorithm": "cumulative_audio_prefix_rollback",
            "max_session_audio_seconds": 30,
            "persistent_causal_encoder_state": False,
            "segment_rollover_supported": False,
        },
    )
    provider_params_type = None

    def __init__(self, **kwargs: object) -> None:
        # Construction must remain free of filesystem, device and network access.
        self.config = Qwen3ASRConfig.from_env(ENGINE_ID, **kwargs)
        self._runtime: CoreMLRuntime | None = None
        self._inference_lock = Lock()

    def _build_command(self) -> str:
        return f"qwen3-asr-ane build --output {shlex.quote(str(self.config.model_dir))}"

    def _artifact_requirements(
        self, context: ArtifactContext
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        root = self.config.model_dir.expanduser().resolve()
        state, revision = _inspect_bundle(root)
        ready = state == "ready"
        requirement = ArtifactRequirement(
            artifact_id="qwen3-asr-1.7b-coreml",
            label="Qwen3-ASR 1.7B local Core ML bundle",
            state=state,
            required_for_inference=True,
            can_acquire_now=False,
            may_acquire_during_inference=False,
            source_is_mutable=False,
            acquisition_blocker=None if ready else "action_required",
            required_actions=()
            if ready
            else (
                ArtifactAction(
                    kind="provide_artifacts",
                    message=f"Build the local model bundle: {self._build_command()}",
                ),
            ),
            location=root,
            artifact_version=revision,
        )
        return True, (requirement,), ()

    def _ensure_model_loaded(self) -> CoreMLRuntime:
        """Load once while the caller holds the inference lock; never acquire weights."""
        if self._runtime is None:
            report = self.artifact_status()
            if report.readiness != "ready":
                raise ArtifactUnavailableError(
                    "The local Qwen3-ASR model bundle is unavailable.",
                    reason="action_required",
                    report=report,
                    hint=self._build_command(),
                )
            from .runtime import CoreMLRuntime

            self._runtime = CoreMLRuntime(self.config.model_dir.expanduser().resolve())
        return self._runtime

    def prepare(self) -> None:
        """Load the local runtime once without acquiring persistent artifacts."""
        with self._inference_lock:
            self._ensure_model_loaded()

    def close(self, *, timeout: float = 5.0) -> None:
        """Wait for active inference and close the runtime; a later prepare may reopen.

        A failed close retains the runtime and its buffer owners so callers can
        retry cleanup. Do not treat an exception as successful model disposal.
        """
        with self._inference_lock:
            if self._runtime is not None:
                self._runtime.close(timeout=timeout)
                self._runtime = None

    def __enter__(self) -> Self:
        self.prepare()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        if prepared.array is None:
            raise TranscriptionError("Standard ASR did not supply the declared array input.")
        language = None if params.language == "auto" else params.language
        try:
            # Core ML stateful decoding and first-time loading share one critical section.
            with self._inference_lock:
                runtime = self._ensure_model_loaded()
                result = runtime.transcribe(
                    prepared.array,
                    language=language,
                    max_new_tokens=self.config.max_new_tokens,
                    context=params.prompt or "",
                )
            return TranscriptionResult(
                text=result.text,
                detected_language=normalize_model_language(result.language)
                if language is None
                else None,
                duration=len(prepared.array) / prepared.sample_rate,
            )
        except (ArtifactUnavailableError, TranscriptionError):
            raise
        except Exception as exc:
            raise TranscriptionError("Qwen3-ASR Core ML inference failed.") from exc

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        from .streaming import Qwen3ASRSession

        return Qwen3ASRSession(self, gated_params, audio_format, prepared_audio)


def _inspect_package(package: Path) -> str:
    """Check package-declared resources without loading Core ML or reading weights."""
    manifest_path = package / "Manifest.json"
    try:
        if not manifest_path.resolve().is_relative_to(package):
            return "corrupt"
        if not manifest_path.is_file():
            return "incomplete"
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            return "corrupt"
        entries = manifest.get("itemInfoEntries")
        root_id = manifest.get("rootModelIdentifier")
        version = manifest.get("fileFormatVersion")
        if (
            not isinstance(entries, dict)
            or not entries
            or not isinstance(root_id, str)
            or root_id not in entries
            or not isinstance(version, str)
            or not version.strip()
        ):
            return "corrupt"
        data_root = (package / "Data").resolve()
        if not data_root.is_relative_to(package):
            return "corrupt"
        state = "ready"
        for identifier, entry in entries.items():
            if not isinstance(entry, dict):
                return "corrupt"
            relative = entry.get("path")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                return "corrupt"
            payload = (data_root / relative).resolve()
            if payload == data_root or not payload.is_relative_to(data_root):
                return "corrupt"
            if payload.is_file():
                if payload.stat().st_size == 0:
                    state = "incomplete"
            elif payload.is_dir():
                if identifier == root_id:
                    return "corrupt"
                nonempty_file = False
                # Inspect every descendant, including symlinks, before calling
                # the directory complete. Weight file names are not prescribed.
                for child in payload.rglob("*"):
                    if not child.resolve().is_relative_to(data_root):
                        return "corrupt"
                    if child.is_file() and child.stat().st_size > 0:
                        nonempty_file = True
                if not nonempty_file:
                    state = "incomplete"
            else:
                state = "incomplete"
        return state
    except (ValueError, UnicodeError, RuntimeError):
        return "corrupt"
    except FileNotFoundError:
        return "incomplete"


def _inspect_bundle(root: Path) -> tuple[str, str | None]:
    """Inspect local completeness without loading models or claiming device placement.

    Core ML validates its model contents when loaded. This inexpensive check verifies
    manifest identity, safe paths and payload presence; it does not certify numerical
    accuracy or that the operating system assigns any operation to ANE.
    """
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return "missing", None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeError):
        return "corrupt", None
    if not isinstance(manifest, dict):
        return "corrupt", None
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        return "corrupt", None
    if manifest.get("model_id") != MODEL_ID:
        return "corrupt", None
    files = manifest.get("files")
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        return "corrupt", None
    if not isinstance(files, dict) or not _REQUIRED_ROLES.issubset(files):
        return "incomplete", revision
    for relative in files.values():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            return "corrupt", revision
        payload = (root / relative).resolve()
        if not payload.is_relative_to(root) or payload == root:
            return "corrupt", revision
        if payload.suffix == ".mlpackage":
            state = _inspect_package(payload)
            if state != "ready":
                return state, revision
        elif not payload.is_file() or payload.stat().st_size == 0:
            return "incomplete", revision
    return "ready", revision


def create_engine(**kwargs: object) -> Qwen3ASREngine:
    """Construct the registered Qwen3-ASR 1.7B preset without loading models."""
    return Qwen3ASREngine(**kwargs)
