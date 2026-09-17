"""Standard ASR 0.2 batch and bounded streaming adapter for local Qwen3-ASR."""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import Field, TypeAdapter, model_validator
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    ConfigError,
    TranscriptionError,
)
from standard_asr.engine import (
    ArtifactAction,
    ArtifactContext,
    ArtifactDeclaration,
    ArtifactProgress,
    ArtifactRequirement,
    AudioFormat,
    BaseConfig,
    BaseProperties,
    BatchCapabilities,
    DeclaredCapabilities,
    DeclaredEngineMetadata,
    Diagnostic,
    DownloadConfigMixin,
    EngineBase,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    InputKind,
    LanguageCaps,
    LanguageConfigMixin,
    PreparedAudio,
    PromptCap,
    PromptConstraints,
    ProviderParams,
    RuntimeParams,
    StreamingCapabilities,
    StreamingGuidanceCaps,
    TranscriptionResult,
    TranscriptionSession,
    allow_downloads,
    resolve_download_root,
)

from .bundle import language_head_output, offline_frontend_batch_size
from .conversion.build import SOURCE_REVISION
from .conversion.toolchain import conversion_toolchain_available as _conversion_toolchain_available
from .embedding import embedding_quantization
from .errors import ModelLimitError
from .languages import LANGUAGE_NAMES, classify_model_language
from .profiles import PROFILES, ProfileName

if TYPE_CHECKING:
    from .draft import DraftRuntime
    from .runtime import CoreMLRuntime


ENGINE_ID = "std-qwen3asr-ane"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_KEY = f"{ENGINE_ID}/1.7b"
BUNDLE_ARTIFACT_ID = "qwen3-asr-1.7b-coreml"
DRAFT_ARTIFACT_ID = "qwen3-asr-0.6b-gpu-draft"
# The recipe `standard-asr pull` builds: the measured default bundle.
BUILD_RECIPE = {"cache_length": 1024, "token_batch_size": 16, "layers_per_partition": 14}
COMPRESS_RECIPE = {"scheme": "palette", "bits": 8, "group_size": 32}
# The shipped bundles have a 1024-position decoder cache. Thirty seconds of audio
# occupies 390 positions, the template about 20, the default generation budget
# 256, and a streaming session also replays up to a transcript's worth of prefix
# text; roughly 250 positions remain in the worst case. The standard layer
# enforces this bound with a word-based estimate that can under-count BPE tokens
# of URLs and digit runs several times over, so declare well below the limit.
PROMPT_MAX_TOKENS = 128
_REQUIRED_ROLES = {
    "frontend",
    "encoder",
    "decoder",
    "lm_head",
    "embedding",
    "tokenizer",
    "mel_filters",
}


class Qwen3ASRConfig(
    DownloadConfigMixin, LanguageConfigMixin, BaseConfig[Literal["std-qwen3asr-ane"]]
):
    """Standard ASR cache defaults, with explicit paths for existing local bundles."""

    engine: Literal["std-qwen3asr-ane"] = ENGINE_ID
    default_language: str = "auto"
    profile: ProfileName = "general"
    model_dir: Path = Field(
        default_factory=lambda: resolve_download_root() / ENGINE_ID / "qwen3-asr-1.7b",
        description="Bundle path; defaults to the selected profile in the Standard ASR cache.",
    )
    source_dir: Path = Field(
        default_factory=lambda: resolve_download_root() / ENGINE_ID / "source/Qwen3-ASR-1.7B",
        description=(
            "Where `standard-asr pull` keeps the pinned official checkpoint it converts "
            "from; reused when present."
        ),
    )
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    use_draft: bool = Field(
        default=False,
        description="Enable the optional GPU draft using its managed cache path.",
    )
    draft_source_dir: Path = Field(
        default_factory=lambda: resolve_download_root() / ENGINE_ID / "source/Qwen3-ASR-0.6B",
        description="Pinned 0.6B checkpoint cache, used by explicit draft acquisition.",
    )
    draft_dir: Path | None = Field(
        default=None,
        description=(
            "Optional draft bundle from `qwen3-asr-ane build-draft`. When set, batch "
            "transcription proposes tokens with Qwen3-ASR 0.6B on the GPU and verifies "
            "them on the Neural Engine; output is unchanged. Needs the gpu-draft extra."
        ),
    )
    draft_lookahead: int = Field(default=15, ge=1, le=15)
    draft_bits: Literal[4, 8] | None = Field(
        default=4, description="In-memory quantization of the draft decoder; None keeps bf16."
    )
    stream_chunk_seconds: float = Field(default=2.0, ge=0.1, le=30.0)
    stream_unfixed_chunks: int = Field(default=2, ge=0)
    stream_unfixed_tokens: int = Field(default=5, ge=0)
    stream_max_audio_seconds: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    stream_audio_queue_size: int = Field(default=4, ge=1, le=128)

    @model_validator(mode="before")
    @classmethod
    def _profile_defaults(cls, values):
        if isinstance(values, Mapping):
            values = dict(values)
            explicit_root = TypeAdapter(Path | None).validate_python(values.get("download_root"))
            root = resolve_download_root(explicit_root)
            assert root is not None  # Core ML has no native model-download cache.
            root = root.absolute() / ENGINE_ID
            short = values.get("profile", cls.model_fields["profile"].default) == "short-dictation"
            name = "qwen3-asr-1.7b-short-dictation" if short else "qwen3-asr-1.7b"
            values.setdefault("model_dir", root / name)
            values.setdefault("source_dir", root / "source/Qwen3-ASR-1.7B")
            values.setdefault("draft_source_dir", root / "source/Qwen3-ASR-0.6B")
            if short:
                values.setdefault("max_new_tokens", PROFILES["short-dictation"].max_new_tokens)
            if (
                TypeAdapter(bool).validate_python(values.get("use_draft", False))
                and values.get("draft_dir") is None
            ):
                target = TypeAdapter(Path).validate_python(values["model_dir"])
                values["draft_dir"] = target.with_name(target.name + "-draft")
        return values


class Qwen3ASRParams(ProviderParams):
    """Per-request decoding budget; omitted values use the engine's defaults."""

    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)


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
        max_audio_duration=30.0,
        wire_encodings=["pcm_s16le", "pcm_f32le"],
        selectable_languages=[*LANGUAGE_NAMES, "auto"],
        detectable_languages=list(LANGUAGE_NAMES),
        description="Qwen3-ASR 1.7B using locally converted Core ML models targeting ANE.",
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
            guidance=GuidanceCaps(
                prompt=PromptCap(
                    supported=True, constraints=PromptConstraints(max_tokens=PROMPT_MAX_TOKENS)
                )
            ),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
        streaming=StreamingCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
            guidance=StreamingGuidanceCaps(
                prompt=PromptCap(
                    supported=True, constraints=PromptConstraints(max_tokens=PROMPT_MAX_TOKENS)
                )
            ),
            emits_partials=FlagCap(supported=True),
            finality_level=FinalityCap(mode="closed"),
        ),
    )
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=ArtifactDeclaration(
            applicable=True,
            # `standard-asr pull` downloads the pinned checkpoint and runs the
            # conversion recipe below; inference itself never acquires anything.
            supports_explicit_acquisition=True,
            may_acquire_during_inference=False,
        ),
        x_qwen3asr_streaming={
            "algorithm": "cumulative_audio_prefix_rollback",
            "effective_session_audio_limit": "minimum_of_configuration_and_loaded_bundle_duration",
            "decoder_prefix_reuse": "exact_embeddings_at_token_batch_boundaries",
            "audio_graph_reuse": "exact_padded_inputs_and_masks",
            "audio_feature_reuse": "aligned_stable_stft_and_unclipped_mel_prefix",
            "context_limits": "session_configuration_and_bundle_audio_and_decoder_token_capacity",
            "persistent_causal_encoder_state": False,
            "segment_rollover_supported": False,
        },
    )
    provider_params_type = Qwen3ASRParams

    def __init__(self, **kwargs: object) -> None:
        # Construction must remain free of filesystem, device and network access.
        self.config = self.config_type.from_env(ENGINE_ID, **kwargs)
        self._runtime: CoreMLRuntime | None = None
        self._draft: DraftRuntime | None = None
        self._inference_lock = Lock()

    def _pull_command(self) -> str:
        command = f"standard-asr pull {self.properties.model_id}"
        # A remedy must address this configured instance, even after changing cwd.
        settings = {
            "profile": self.config.profile,
            "model_dir": self.config.model_dir.expanduser().absolute(),
            "source_dir": self.config.source_dir.expanduser().absolute(),
        }
        if self.config.draft_dir is not None:
            settings["draft_dir"] = self.config.draft_dir.expanduser().absolute()
            settings["draft_source_dir"] = self.config.draft_source_dir.expanduser().absolute()
        for key, value in settings.items():
            command += " --set " + shlex.quote(f"{key}={value}")
        return command

    def _acquisition_gate(self, state: str, *, needs_draft: bool = False) -> dict:
        """Fields describing whether `standard-asr pull` can run for one requirement."""
        if state == "ready":
            return {"can_acquire_now": False, "acquisition_blocker": None, "required_actions": ()}
        if state in ("incomplete", "corrupt"):
            # Never delete a directory the plugin did not just create.
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "action_required",
                "required_actions": (
                    ArtifactAction(
                        kind="provide_artifacts",
                        message=(
                            f"The directory is {state}; move it away, then run "
                            f"{self._pull_command()}"
                        ),
                    ),
                ),
            }
        sources = [self.config.source_dir]
        if needs_draft:
            sources.append(self.config.draft_source_dir)
        if not allow_downloads() and any(
            not (source.expanduser() / "source.json").is_file() for source in sources
        ):
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "downloads_disabled",
                "required_actions": (),
            }
        return {"can_acquire_now": True, "acquisition_blocker": None, "required_actions": ()}

    def _artifact_requirements(
        self, context: ArtifactContext
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        root = self.config.model_dir.expanduser().resolve()
        state, revision = _inspect_bundle(root)
        requirement = ArtifactRequirement(
            artifact_id=BUNDLE_ARTIFACT_ID,
            label="Qwen3-ASR 1.7B local Core ML bundle",
            state=state,
            required_for_inference=True,
            may_acquire_during_inference=False,
            source_is_mutable=False,
            location=root,
            artifact_version=revision,
            **self._acquisition_gate(state),
        )
        if self.config.draft_dir is None or context.mode == "streaming":
            return True, (requirement,), ()
        draft_root = self.config.draft_dir.expanduser().resolve()
        draft_state, draft_revision = _inspect_draft_bundle(draft_root)
        draft_requirement = ArtifactRequirement(
            artifact_id=DRAFT_ARTIFACT_ID,
            label="Qwen3-ASR 0.6B draft checkpoint and verify head",
            state=draft_state,
            required_for_inference=True,
            may_acquire_during_inference=False,
            source_is_mutable=False,
            location=draft_root,
            artifact_version=draft_revision,
            **self._acquisition_gate(draft_state, needs_draft=True),
        )
        return True, (requirement, draft_requirement), ()

    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress,
    ) -> None:
        """Download the pinned checkpoint and run the measured conversion recipe.

        The bundle is built in a private work directory next to ``model_dir``
        and only the compiled result lands at ``model_dir``; the draft bundle is
        built after the target it binds to. Missing or incompatible conversion
        dependencies are supplied by a managed worker, never by inference.
        """
        targets = {requirement.artifact_id for requirement in requirements}

        def emit(phase: str, artifact_id: str) -> None:
            if progress is not None:
                progress(ArtifactProgress(phase=phase, artifact_id=artifact_id))

        try:
            from .acquisition import acquisition_lock

            with acquisition_lock(self.config, include_draft=DRAFT_ARTIFACT_ID in targets):
                # Another process may have completed acquisition after preflight.
                targets = {
                    item.artifact_id
                    for item in self._artifact_requirements(context)[1]
                    if item.artifact_id in targets and item.state != "ready"
                }
                self._acquire_targets(targets, emit, progress)
        except ArtifactAcquisitionError:
            raise
        except Exception as exc:
            raise ArtifactAcquisitionError(
                "Building the local model bundle failed; inspect the conversion log.",
                reason="failed",
                hint=f"After resolving the reported failure, retry: {self._pull_command()}",
            ) from exc

    def _acquire_targets(self, targets, emit, progress) -> None:
        if targets:
            if not _conversion_toolchain_available():
                from .acquisition import acquire_in_worker

                acquire_in_worker(self.config, targets, progress)
                return
            # Keep `standard-asr pull --json` parseable even in an application
            # environment that already carries the conversion dependencies.
            with redirect_stdout(sys.stderr):
                if BUNDLE_ARTIFACT_ID in targets:
                    self._acquire_bundle(emit)
                if DRAFT_ARTIFACT_ID in targets:
                    self._acquire_draft(emit)

    def _ensure_source(self, emit) -> Path:
        from .conversion.build import download_source

        source = self.config.source_dir.expanduser().resolve()
        if not (source / "source.json").is_file():
            emit("transferring", BUNDLE_ARTIFACT_ID)
            download_source(source, revision=SOURCE_REVISION, model_id=MODEL_ID)
        try:
            provenance = json.loads((source / "source.json").read_text())
        except (ValueError, OSError) as exc:
            raise ArtifactAcquisitionError(
                "Cannot read the source checkpoint's provenance. Select a valid source_dir.",
                reason="action_required",
            ) from exc
        if provenance != {"model_id": MODEL_ID, "revision": SOURCE_REVISION}:
            raise ArtifactAcquisitionError(
                "source_dir does not contain the pinned Qwen3-ASR 1.7B checkpoint. "
                "Select a matching source_dir or a new directory for acquisition.",
                reason="action_required",
            )
        return source

    def _acquire_bundle(self, emit) -> None:
        from .compiled import compile_bundle
        from .conversion.build import build_bundle
        from .conversion.compress import compress_bundle

        target = self.config.model_dir.expanduser().resolve()
        if target.exists():
            raise ArtifactAcquisitionError(
                f"{target} already exists; move it away before pulling.",
                reason="action_required",
            )
        source = self._ensure_source(emit)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as directory:
            work = Path(directory)
            emit("converting", BUNDLE_ARTIFACT_ID)
            recipe = dict(BUILD_RECIPE)
            if self.config.profile != "general":
                recipe.update(
                    profile=self.config.profile,
                    cache_length=PROFILES[self.config.profile].cache_length,
                )
            build_bundle(source, work / "fp16", **recipe)
            compress_bundle(work / "fp16", work / "lut8", **COMPRESS_RECIPE)
            emit("verifying", BUNDLE_ARTIFACT_ID)
            compile_bundle(work / "lut8", work / "compiled")
            (work / "compiled").rename(target)

    def _acquire_draft(self, emit) -> None:
        from .conversion.draft import build_draft_bundle

        draft = self.config.draft_dir.expanduser().resolve()
        if draft.exists():
            raise ArtifactAcquisitionError(
                f"{draft} already exists; move it away before pulling.",
                reason="action_required",
            )
        source = self._ensure_source(emit)
        from .conversion.build import download_source
        from .draft import DRAFT_MODEL_ID, DRAFT_REVISION

        draft_source = self.config.draft_source_dir.expanduser().resolve()
        if not (draft_source / "source.json").is_file():
            emit("transferring", DRAFT_ARTIFACT_ID)
            download_source(draft_source, revision=DRAFT_REVISION, model_id=DRAFT_MODEL_ID)
        emit("converting", DRAFT_ARTIFACT_ID)
        draft.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{draft.name}-", dir=draft.parent) as directory:
            staged = Path(directory) / "bundle"
            build_draft_bundle(
                self.config.model_dir.expanduser().resolve(),
                source,
                staged,
                draft_source=draft_source,
            )
            staged.rename(draft)

    def _ensure_model_loaded(self) -> CoreMLRuntime:
        """Load once while the caller holds the inference lock; never acquire weights."""
        if self._runtime is None:
            self._require_artifacts(include_draft=False)
            if self.config.profile == "short-dictation":
                manifest = json.loads(
                    (self.config.model_dir / "manifest.json").expanduser().read_text()
                )
                duration = manifest.get("max_audio_seconds")
                if manifest.get("max_sequence_length") != 512 or not (
                    isinstance(duration, (float, int)) and 0 < duration <= 12
                ):
                    raise ConfigError(
                        "The short-dictation profile requires a cache-512 bundle with at most 12 seconds of audio.",
                        hint="Select the short-dictation preset and acquire its matching bundle with standard-asr pull.",
                    )
            from .runtime import CoreMLRuntime

            self._runtime = CoreMLRuntime(self.config.model_dir.expanduser().resolve())
        return self._runtime

    def _require_artifacts(self, *, include_draft: bool) -> None:
        """Check the artifacts used by this operation, retaining the full status report."""
        report = self.artifact_status(
            ArtifactContext(mode="batch" if include_draft else "streaming")
        )
        required = {BUNDLE_ARTIFACT_ID}
        if include_draft and self.config.draft_dir is not None:
            required.add(DRAFT_ARTIFACT_ID)
        ready = {item.artifact_id for item in report.requirements if item.state == "ready"}
        if required <= ready:
            return
        raise ArtifactUnavailableError(
            "The local Qwen3-ASR artifacts required by this operation are unavailable.",
            reason="action_required",
            report=report,
            hint=self._pull_command(),
        )

    def _load_draft(self, runtime: CoreMLRuntime) -> DraftRuntime:
        from .draft import DraftDependencyError, DraftRuntime

        if self.config.draft_lookahead >= runtime.token_batch_size:
            raise ConfigError(
                f"draft_lookahead={self.config.draft_lookahead} does not fit the bundle's "
                f"{runtime.token_batch_size}-token graph (held token plus proposals).",
                hint="Lower draft_lookahead or build the bundle with --token-batch-size 16.",
            )
        try:
            return DraftRuntime(
                self.config.draft_dir.expanduser().resolve(),
                runtime,
                quantize_bits=self.config.draft_bits,
            )
        except DraftDependencyError as exc:
            raise ConfigError(
                "draft_dir is set but MLX is not installed in this environment.",
                hint="Install std-qwen3asr-ane with its gpu-draft extra, or unset draft_dir.",
            ) from exc

    def prepare(self) -> None:
        """Warm the ANE target; the optional GPU draft loads on its first batch request."""
        with self._inference_lock:
            self._ensure_model_loaded()

    def close(self, *, timeout: float = 5.0) -> None:
        """Wait for active inference and close the runtime; a later prepare may reopen.

        A failed close retains the runtime and its buffer owners so callers can
        retry cleanup. Do not treat an exception as successful model disposal.
        """
        with self._inference_lock:
            if self._draft is not None:
                # The draft/verify head can still own target buffers after a
                # failed close. Keep both owners intact until cleanup succeeds.
                self._draft.close(timeout=timeout)
                self._draft = None
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
                needs_draft = self.config.draft_dir is not None and self._draft is None
                if needs_draft:
                    self._require_artifacts(include_draft=True)
                runtime = self._ensure_model_loaded()
                if needs_draft:
                    # A failed draft load leaves the valid ANE target available
                    # to streaming and to a later explicit retry.
                    self._draft = self._load_draft(runtime)
                if self._draft is not None:
                    result = runtime.transcribe_speculative(
                        prepared.array,
                        self._draft,
                        language=language,
                        max_new_tokens=self._generation_budget(params),
                        context=params.prompt or "",
                        lookahead=self.config.draft_lookahead,
                    )
                else:
                    result = runtime.transcribe(
                        prepared.array,
                        language=language,
                        max_new_tokens=self._generation_budget(params),
                        context=params.prompt or "",
                    )
            detected, diagnostics = detected_language(result.language, language)
            return TranscriptionResult(
                text=result.text,
                detected_language=detected,
                duration=len(prepared.array) / prepared.sample_rate,
                diagnostics=diagnostics,
            )
        except (ArtifactUnavailableError, ConfigError, TranscriptionError):
            raise
        except ModelLimitError as exc:
            raise TranscriptionError(
                f"Request exceeds the loaded bundle's capacity: {exc}"
            ) from exc
        except Exception as exc:
            raise TranscriptionError("Qwen3-ASR Core ML inference failed.") from exc

    def _generation_budget(self, params: RuntimeParams) -> int:
        provider = params.provider_params
        if isinstance(provider, Qwen3ASRParams) and provider.max_new_tokens is not None:
            return provider.max_new_tokens
        return self.config.max_new_tokens

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        from .streaming import Qwen3ASRSession

        return Qwen3ASRSession(self, gated_params, audio_format, prepared_audio)


class ShortDictationConfig(Qwen3ASRConfig):
    profile: Literal["short-dictation"] = "short-dictation"
    max_new_tokens: int = Field(default=128, ge=1, le=4096)


class ShortDictationEngine(Qwen3ASREngine):
    """Discoverable short-utterance preset with its own static duration boundary."""

    config_type = ShortDictationConfig
    properties = Qwen3ASREngine.properties.model_copy(
        update={
            "model_name": "1.7b-short-dictation",
            "max_audio_duration": 12.0,
        }
    )


def detected_language(
    model_language: str | None, requested: str | None
) -> tuple[str | None, list[Diagnostic]]:
    """Map the model's language line to BCP-47, disclosing names we cannot map.

    A forced language reports no detection. In ``auto`` mode the model may emit
    a name outside its published list; the result then carries ``None`` plus a
    diagnostic instead of silently dropping the model's answer.
    """
    detected, unmapped = classify_model_language(model_language, requested)
    if unmapped is None:
        return detected, []
    return None, [
        Diagnostic(
            level="info",
            code="detected_language_unmapped",
            message="The model reported a language name outside its published language list.",
            param="language",
            provided=unmapped,
            effective=None,
        )
    ]


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


def _inspect_compiled_package(package: Path) -> str:
    """Check our compiled ML Program layout without starting device specialization."""
    if not package.is_dir():
        return "incomplete"
    for child in package.rglob("*"):
        if not child.resolve().is_relative_to(package):
            return "corrupt"
    # These are the Core ML compiler's ML Program metadata and executable MIL
    # files. Our ASR graphs also all have weights; discovery remains a presence
    # check, with binary compatibility validated by Core ML during prepare().
    required = (package / "coremldata.bin", package / "model.mil", package / "weights/weight.bin")
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return "incomplete"
    return "ready"


def _inspect_draft_bundle(root: Path) -> tuple[str, str | None]:
    """Check the draft bundle's layout without importing MLX or loading models."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ("incomplete" if root.exists() else "missing"), None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeError):
        return "corrupt", None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "qwen3-asr-ane-draft"
        or not isinstance(manifest.get("draft"), dict)
        or not isinstance(manifest.get("verify_head"), dict)
    ):
        return "corrupt", None
    revision = manifest["draft"].get("revision")
    if not isinstance(revision, str) or not revision.strip():
        return "corrupt", None
    for relative in (manifest["draft"].get("path"), manifest["verify_head"].get("path")):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            return "corrupt", revision
        payload = (root / relative).resolve()
        if not payload.is_relative_to(root) or payload == root:
            return "corrupt", revision
    head = (root / manifest["verify_head"]["path"]).resolve()
    if head.suffix == ".mlmodelc":
        state = _inspect_compiled_package(head)
    elif head.suffix == ".mlpackage":
        state = _inspect_package(head)
    else:
        return "corrupt", revision
    if state != "ready":
        return state, revision
    checkpoint = (root / manifest["draft"]["path"]).resolve()
    for name in ("config.json", "model.safetensors"):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            return "incomplete", revision
    return "ready", revision


def _inspect_bundle(root: Path) -> tuple[str, str | None]:
    """Inspect local completeness without loading models or claiming device placement.

    Core ML validates its model contents when loaded. This inexpensive check verifies
    manifest identity, safe paths and payload presence; it does not certify numerical
    accuracy or that the operating system assigns any operation to ANE.
    """
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ("incomplete" if root.exists() else "missing"), None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeError):
        return "corrupt", None
    if not isinstance(manifest, dict):
        return "corrupt", None
    try:
        language_head_output(manifest)
        batch_size = offline_frontend_batch_size(manifest)
        quantization = embedding_quantization(manifest)
    except (ValueError, TypeError, AttributeError):
        return "corrupt", None
    if manifest.get("model_id") != MODEL_ID:
        return "corrupt", None
    files = manifest.get("files")
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        return "corrupt", None
    required_roles = _REQUIRED_ROLES | ({"frontend_batched"} if batch_size > 1 else set())
    if quantization is not None:
        required_roles |= {"embedding_scales"}
    if not isinstance(files, dict) or not required_roles.issubset(files):
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
        elif payload.suffix == ".mlmodelc":
            state = _inspect_compiled_package(payload)
            if state != "ready":
                return state, revision
        elif not payload.is_file() or payload.stat().st_size == 0:
            return "incomplete", revision
    return "ready", revision


def create_engine(**kwargs: object) -> Qwen3ASREngine:
    """Construct the registered Qwen3-ASR 1.7B preset without loading models."""
    return Qwen3ASREngine(**kwargs)
