"""Standard ASR batch and streaming adapter for local Qwen3-ASR."""

from __future__ import annotations

import importlib.util
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import Field, TypeAdapter, model_validator
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    AudioProcessingError,
    ConfigError,
    StructuredError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.contract.results import compose_segment_text
from standard_asr.engine import (
    ArtifactContext,
    ArtifactDeclaration,
    AudioFormat,
    BaseConfig,
    BaseProperties,
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    DeclaredEngineMetadata,
    Diagnostic,
    DiarizationCap,
    DiarizationConstraints,
    DownloadConfigMixin,
    EngineBase,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    InputKind,
    LanguageCaps,
    LanguageConfigMixin,
    PhraseHintsCap,
    PhraseHintsConstraints,
    PreparedAudio,
    PromptCap,
    PromptConstraints,
    ProviderParams,
    RuntimeParams,
    Segment,
    StreamingCapabilities,
    StreamingGuidanceCaps,
    StreamTimestampsCap,
    TranscriptionResult,
    TranscriptionSession,
    WordTimestampsCap,
    resolve_download_root,
)

from .artifact_lifecycle import (
    BATCH_HEAD_ARTIFACT_ID,
    BUNDLE_ARTIFACT_ID,
    DRAFT_ARTIFACT_ID,
    ArtifactManager,
)
from .bundle import language_head_output
from .decoding_guidance import GuidanceRequestError
from .deployment import unsupported_host_reason
from .errors import CancellationToken, ModelLimitError, raise_if_cancelled
from .languages import LANGUAGE_NAMES, classify_model_language, qwen_control_language
from .profiles import PROFILES
from .result_text import needs_join_space, shift_source_offsets

if TYPE_CHECKING:
    from .draft import DraftRuntime
    from .runtime import CoreMLRuntime


ENGINE_ID = "std-qwen3asr-ane"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_KEY = f"{ENGINE_ID}/1.7b"
# The shipped bundles have a 1024-position decoder cache. Thirty seconds of audio
# occupies 390 positions, the template about 20, the default generation budget
# 256, and a streaming session also replays up to a transcript's worth of prefix
# text; roughly 250 positions remain in the worst case. The standard layer
# enforces this bound with a word-based estimate that can under-count BPE tokens
# of URLs and digit runs several times over, so declare well below the limit.
PROMPT_MAX_TOKENS = 128


class Qwen3ASRConfig(
    DownloadConfigMixin, LanguageConfigMixin, BaseConfig[Literal["std-qwen3asr-ane"]]
):
    """Standard ASR cache defaults, with explicit paths for existing local bundles."""

    engine: Literal["std-qwen3asr-ane"] = ENGINE_ID
    default_language: str = "auto"
    profile: Literal["general"] = "general"
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
    max_recording_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    stream_max_audio_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    stream_audio_queue_size: int = Field(default=4, ge=1, le=128)
    use_batching: bool = Field(
        default=False, description="Acquire a compact head for packed offline requests."
    )
    batch_head_dir: Path | None = Field(
        default=None, description="Optional target-bound offline batch head."
    )
    use_alignment: bool = Field(
        default=False, description="Enable explicitly acquired Qwen forced alignment on the CPU."
    )
    alignment_dir: Path = Field(
        default_factory=lambda: resolve_download_root() / ENGINE_ID / "auxiliary/alignment",
        description="Managed forced-aligner model and isolated runtime directory.",
    )
    use_diarization: bool = Field(
        default=False, description="Enable explicitly acquired speaker diarization on the CPU."
    )
    diarization_dir: Path = Field(
        default_factory=lambda: resolve_download_root() / ENGINE_ID / "auxiliary/diarization",
        description="Directory containing the pinned segmentation and speaker-embedding models.",
    )

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
            values.setdefault("alignment_dir", root / "auxiliary/alignment")
            values.setdefault("diarization_dir", root / "auxiliary/diarization")
            if short:
                values.setdefault("max_new_tokens", PROFILES["short-dictation"].max_new_tokens)
            if (
                TypeAdapter(bool).validate_python(values.get("use_draft", False))
                and values.get("draft_dir") is None
            ):
                target = TypeAdapter(Path).validate_python(values["model_dir"])
                values["draft_dir"] = target.with_name(target.name + "-draft")
            if (
                TypeAdapter(bool).validate_python(values.get("use_batching", False))
                and values.get("batch_head_dir") is None
            ):
                target = TypeAdapter(Path).validate_python(values["model_dir"])
                values["batch_head_dir"] = target.with_name(target.name + "-batch-head")
        return values


class Qwen3ASRParams(ProviderParams):
    """Per-request decoding budget; omitted values use the engine's defaults."""

    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    disable_draft: bool = Field(
        default=False,
        description="Use the ANE target without a configured GPU draft for this request.",
    )
    include_metrics: bool = Field(
        default=False,
        description="Include native timings, token IDs and raw decoder output in result.extra.",
    )


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
        max_audio_duration=None,
        wire_encodings=["pcm_s16le", "pcm_f32le"],
        selectable_languages=[*LANGUAGE_NAMES, "auto"],
        detectable_languages=list(LANGUAGE_NAMES),
        description="Qwen3-ASR 1.7B using locally converted Core ML models targeting ANE.",
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            x_qwen3asr_multi_input={"supported": True, "constraints": {"max_group_size": 16}},
            language=LanguageCaps(
                runtime_override=FlagCap(supported=True),
                candidate_languages=CandidateLanguagesCap(
                    supported=True, constraints=CandidateLanguagesConstraints(max=8)
                ),
            ),
            guidance=GuidanceCaps(
                prompt=PromptCap(
                    supported=True, constraints=PromptConstraints(max_tokens=PROMPT_MAX_TOKENS)
                ),
                phrase_hints=PhraseHintsCap(
                    supported=True,
                    constraints=PhraseHintsConstraints(max_terms=16, max_chars_per_term=128),
                ),
            ),
            word_timestamps=WordTimestampsCap(
                supported=True, granularities=["word", "segment", "char"]
            ),
            diarization=DiarizationCap(
                supported=True, constraints=DiarizationConstraints(max_speakers=32)
            ),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
        streaming=StreamingCapabilities(
            language=LanguageCaps(
                runtime_override=FlagCap(supported=True),
                candidate_languages=CandidateLanguagesCap(
                    supported=True, constraints=CandidateLanguagesConstraints(max=8)
                ),
            ),
            guidance=StreamingGuidanceCaps(
                prompt=PromptCap(
                    supported=True, constraints=PromptConstraints(max_tokens=PROMPT_MAX_TOKENS)
                ),
                phrase_hints=PhraseHintsCap(
                    supported=True,
                    constraints=PhraseHintsConstraints(max_terms=16, max_chars_per_term=128),
                ),
            ),
            emits_partials=FlagCap(supported=True),
            finality_level=FinalityCap(mode="closed"),
            audio_progress=FlagCap(supported=True),
            word_timestamps=WordTimestampsCap(
                supported=True, granularities=["word", "segment", "char"]
            ),
            timestamps=StreamTimestampsCap(mode="post_align"),
            diarization=DiarizationCap(
                supported=True, constraints=DiarizationConstraints(max_speakers=32)
            ),
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
            "algorithm": "bounded_windows_with_low_energy_boundaries_and_prefix_rollback",
            "effective_session_audio_limit": "configured_recording_and_streaming_limits",
            "decoder_prefix_reuse": "exact_embeddings_at_token_batch_boundaries",
            "audio_graph_reuse": "exact_padded_inputs_and_masks",
            "audio_feature_reuse": "aligned_stable_stft_power_with_full_width_mel_projection",
            "context_limits": "session_configuration_and_bundle_audio_and_decoder_token_capacity",
            "persistent_causal_encoder_state": False,
            "segment_rollover_supported": True,
            "native_window_seconds": PROFILES["general"].max_audio_seconds,
        },
        x_qwen3asr_auxiliary={
            "compute": "cpu",
            "alignment_languages": [
                "zh",
                "en",
                "yue",
                "fr",
                "de",
                "it",
                "ja",
                "ko",
                "pt",
                "ru",
                "es",
            ],
            "diarization_requires_alignment": True,
            "overlapping_word_attribution": "unassigned_when_evidence_is_ambiguous",
        },
        x_qwen3asr_deployment={"os": "macos", "minimum_os_major": 15, "architectures": ["arm64"]},
        x_qwen3asr_batching={
            "method": "transcribe_many",
            "transport": "python",
            "execution": "packed_target_when_eligible_otherwise_serial_target",
            "gpu_draft_used": False,
        },
    )
    provider_params_type = Qwen3ASRParams

    def __init__(self, **kwargs: object) -> None:
        # Construction must remain free of filesystem, device and network access.
        self.config = self.config_type.from_env(ENGINE_ID, **kwargs)
        self._runtime: CoreMLRuntime | None = None
        self._draft: DraftRuntime | None = None
        self._batch_head = None
        self._inference_lock = Lock()
        self._operation_lock = RLock()
        self._auxiliary_lock = RLock()
        self._auxiliary = None
        self._artifacts = ArtifactManager(self.config, self.properties.model_id)

    def _max_audio_duration(self, mode: str) -> float | None:
        """Apply recording policy separately from the native decoding window."""
        limits = [self.config.max_recording_seconds]
        if mode == "streaming":
            limits.append(self.config.stream_max_audio_seconds)
        finite = [value for value in limits if value is not None]
        return min(finite) if finite else None

    @property
    def effective_capabilities(self) -> DeclaredCapabilities:
        """Compact vocabulary heads cannot apply token masks or score biases."""
        capabilities = self.declared_capabilities.model_dump()
        head = getattr(self._runtime, "head_output", None)
        if head is None:
            try:
                manifest = json.loads(
                    (self.config.model_dir.expanduser() / "manifest.json").read_text()
                )
                head = language_head_output(manifest)
            except (OSError, ValueError, TypeError):
                # A missing or broken artifact is reported by artifact_status;
                # it does not redefine the shipped full-logits recipe.
                head = {"kind": "logits"}
        if head.get("kind") == "chunk_max":
            for name in ("batch", "streaming"):
                mode = capabilities[name]
                mode["language"]["candidate_languages"]["supported"] = False
                mode["guidance"]["phrase_hints"]["supported"] = False
        alignment = self.config.use_alignment or self.config.use_diarization
        for name in ("batch", "streaming"):
            mode = capabilities[name]
            mode["word_timestamps"]["supported"] = alignment
            if not alignment:
                mode["word_timestamps"]["granularities"] = []
            mode["diarization"]["supported"] = self.config.use_diarization
        if not alignment:
            capabilities["streaming"]["timestamps"]["mode"] = "none"
        return DeclaredCapabilities.model_validate(capabilities)

    def _pull_command(self) -> str:
        command = self._artifacts.pull_command()
        if self.config.use_alignment or self.config.use_diarization:
            directory = self.config.alignment_dir.expanduser().absolute()
            command += " --set " + shlex.quote(f"alignment_dir={directory}")
        for enabled, field in (
            (self.config.use_alignment, "alignment"),
            (self.config.use_diarization, "diarization"),
        ):
            if enabled:
                command += f" --set use_{field}=true"
                if field == "diarization":
                    directory = self.config.diarization_dir.expanduser().absolute()
                    command += " --set " + shlex.quote(f"diarization_dir={directory}")
        return command

    def _artifact_requirements(self, context: ArtifactContext):
        applicable, requirements, diagnostics = self._artifacts.requirements(context)
        if self.config.use_alignment or self.config.use_diarization:
            requirements += self._auxiliary_models().requirements(context)
        host_error = unsupported_host_reason()
        if host_error is not None:
            requirements = tuple(
                item
                if item.state == "ready"
                else item.model_copy(
                    update={
                        "can_acquire_now": False,
                        "acquisition_blocker": "unsupported",
                        "required_actions": (),
                    }
                )
                for item in requirements
            )
            diagnostics += (
                Diagnostic(level="warning", code="unsupported_host", message=host_error),
            )
        return applicable, requirements, diagnostics

    def _acquire_artifacts(self, context, requirements, refresh, progress) -> None:
        native_ids = {BUNDLE_ARTIFACT_ID, DRAFT_ARTIFACT_ID, BATCH_HEAD_ARTIFACT_ID}
        native = tuple(item for item in requirements if item.artifact_id in native_ids)
        auxiliary = {item.artifact_id for item in requirements} - native_ids
        try:
            if native:
                self._artifacts.acquire(context, native, refresh, progress)
            if auxiliary:
                self._auxiliary_models().acquire(auxiliary, progress)
        except ArtifactAcquisitionError:
            raise
        except Exception as error:
            raise ArtifactAcquisitionError(
                "Model acquisition failed; inspect the underlying operation before retrying.",
                reason="failed",
                hint=self._pull_command(),
            ) from error

    def _acquire_bundle(self, emit) -> None:
        # The worker already runs under the parent acquisition lock.
        self._artifacts.acquire_bundle(emit)

    def _acquire_draft(self, emit) -> None:
        self._artifacts.acquire_draft(emit)

    def _acquire_batch_head(self, emit) -> None:
        self._artifacts.acquire_batch_head(emit)

    def _auxiliary_models(self):
        with self._auxiliary_lock:
            if self._auxiliary is None:
                from .auxiliary import AuxiliaryModels

                self._auxiliary = AuxiliaryModels(self.config)
            return self._auxiliary

    def _require_request_artifacts(self, params: RuntimeParams, *, mode: str) -> None:
        draft_needed = mode == "batch" and self._draft_requested(params)
        auxiliary_needed = params.word_timestamps is not None or params.diarization is not None
        if (
            self._runtime is not None
            and not auxiliary_needed
            and (not draft_needed or self._draft is not None)
        ):
            return
        report = self.artifact_status(ArtifactContext(mode=mode, params=params))
        loaded = set()
        if self._runtime is not None:
            loaded.add(BUNDLE_ARTIFACT_ID)
        if self._draft is not None:
            loaded.add(DRAFT_ARTIFACT_ID)
        if any(
            item.required_for_inference and item.state != "ready" and item.artifact_id not in loaded
            for item in report.requirements
        ):
            raise ArtifactUnavailableError(
                "The models required by this request are not ready.",
                reason="action_required",
                report=report,
                hint=self._pull_command(),
            )
        if auxiliary_needed:
            self._auxiliary_models().validate_request(params)
        if draft_needed and self._draft is None:
            self._check_draft_dependencies()

    @staticmethod
    def _check_draft_dependencies() -> None:
        if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_audio") is None:
            raise ConfigError(
                "The configured GPU draft needs the gpu-draft extra.",
                hint="Install std-qwen3asr-ane[gpu-draft], or disable the draft.",
            )

    def _draft_requested(self, params: RuntimeParams) -> bool:
        provider = params.provider_params
        disabled = isinstance(provider, Qwen3ASRParams) and provider.disable_draft
        return (
            self.config.draft_dir is not None
            and not disabled
            and not (params.candidate_languages or params.phrase_hints)
        )

    def _ensure_model_loaded(self) -> CoreMLRuntime:
        """Load once while the caller holds the inference lock; never acquire weights."""
        if self._runtime is None:
            host_error = unsupported_host_reason()
            if host_error is not None:
                raise ConfigError(host_error, hint="Run the plugin on a supported Mac.")
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
        with self._operation_lock, self._inference_lock:
            if self._auxiliary is not None:
                self._auxiliary.close()
            if self._batch_head is not None:
                self._batch_head.close(timeout=timeout)
                self._batch_head = None
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
        with self._operation_lock:
            self._require_request_artifacts(params, mode="batch")
            return self._transcribe_recording(prepared, params)

    def transcribe_many(self, recordings, params=None, *, batch_size: int = 4):
        """Transcribe independent recordings through the standard preparation pipeline.

        Returns ordered outcomes containing either a standard result or a typed
        error. Packed execution requires an acquired batch head; unsupported
        groups use a disclosed serial target fallback. This bulk path uses the
        ANE target, leaving the optional GPU draft to ordinary single requests.
        """
        from .bulk import transcribe_many

        return transcribe_many(self, recordings, params, batch_size=batch_size)

    def _transcribe_recording(
        self, prepared: PreparedAudio, params: RuntimeParams
    ) -> TranscriptionResult:
        if prepared.array is None:
            raise TranscriptionError("Standard ASR did not supply the declared array input.")
        import numpy as np

        from .longform import LongFormCoordinator

        samples = prepared.array
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise AudioProcessingError("Provide nonempty, finite mono audio through Standard ASR.")
        try:
            window_seconds = self._stream_window_seconds()
            tracker = self._new_speaker_tracker(params)
            if len(samples) <= int(window_seconds * prepared.sample_rate):
                raw = self._recognize_chunk(samples, params, use_draft=True)
                return self._finalize_chunk(raw, samples, params, speaker_tracker=tracker)
            coordinator = LongFormCoordinator(
                sample_rate=prepared.sample_rate,
                native_window_samples=int(window_seconds * prepared.sample_rate),
                max_total_samples=None,
            )
            chunks = []
            for span in coordinator.append(samples):
                raw = self._recognize_chunk(span.samples, params, use_draft=True)
                chunks.append(
                    self._finalize_chunk(
                        raw,
                        span.samples,
                        params,
                        span.start_sample / prepared.sample_rate,
                        speaker_tracker=tracker,
                    )
                )
            tail = coordinator.finish()
            if tail is not None:
                raw = self._recognize_chunk(tail.samples, params, use_draft=True)
                chunks.append(
                    self._finalize_chunk(
                        raw,
                        tail.samples,
                        params,
                        tail.start_sample / prepared.sample_rate,
                        speaker_tracker=tracker,
                    )
                )
            return _merge_chunk_results(chunks, len(samples) / prepared.sample_rate)
        except GuidanceRequestError as error:
            raise UnsupportedFeatureError(
                str(error),
                param="phrase_hints" if params.phrase_hints else "candidate_languages",
            ) from error
        except StructuredError:
            raise
        except ModelLimitError as exc:
            raise TranscriptionError(
                f"Request exceeds the loaded bundle's capacity: {exc}"
            ) from exc
        except Exception as exc:
            raise TranscriptionError("Qwen3-ASR Core ML inference failed.") from exc

    def _stream_window_seconds(self, *, cancel: CancellationToken | None = None) -> float:
        """Read the loaded native window without imposing a recording-length limit."""
        with self._inference_lock:
            raise_if_cancelled(cancel)
            runtime = self._ensure_model_loaded()
            raise_if_cancelled(cancel)
            return runtime.max_audio_seconds

    def _new_stream_contexts(self, *, cancel: CancellationToken | None = None):
        """Create decoder/audio state owned exclusively by one streaming session."""
        with self._inference_lock:
            raise_if_cancelled(cancel)
            runtime = self._ensure_model_loaded()
            return runtime.new_decoder_context(), runtime.new_audio_context()

    def _recognize_chunk(
        self,
        samples,
        params: RuntimeParams,
        *,
        prefix_text: str = "",
        decoder_context=None,
        audio_context=None,
        cancel: CancellationToken | None = None,
        use_draft: bool = False,
    ):
        """Run one bounded native request with shared language/guidance semantics."""
        language = qwen_control_language(None if params.language == "auto" else params.language)
        draft_requested = use_draft and self._draft_requested(params)
        with self._inference_lock:
            raise_if_cancelled(cancel)
            if draft_requested and self._draft is None:
                # Optional dependencies are checked before the expensive ANE load.
                self._check_draft_dependencies()
                self._require_artifacts(include_draft=True)
            runtime = self._ensure_model_loaded()
            kwargs = {
                "language": language,
                "max_new_tokens": self._generation_budget(params),
                "context": params.prompt or "",
            }
            if cancel is not None:
                kwargs["cancel"] = cancel
            if draft_requested:
                if self._draft is None:
                    self._draft = self._load_draft(runtime)
                return runtime.transcribe_speculative(
                    samples, self._draft, lookahead=self.config.draft_lookahead, **kwargs
                )
            if prefix_text:
                kwargs["prefix_text"] = prefix_text
            if decoder_context is not None:
                kwargs["decoder_context"] = decoder_context
            if audio_context is not None:
                kwargs["audio_context"] = audio_context
            if params.candidate_languages:
                kwargs["candidate_language_names"] = [
                    LANGUAGE_NAMES[qwen_control_language(tag)] for tag in params.candidate_languages
                ]
            if params.phrase_hints:
                kwargs["phrase_hints"] = params.phrase_hints
            return runtime.transcribe(samples, **kwargs)

    def _new_speaker_tracker(self, params: RuntimeParams):
        if params.diarization is None:
            return None
        return self._auxiliary_models().new_speaker_tracker()

    def _finalize_chunk(
        self,
        result,
        samples,
        params: RuntimeParams,
        offset_seconds: float = 0.0,
        *,
        cancel: CancellationToken | None = None,
        speaker_tracker=None,
    ):
        """Project one native result, keeping raw language disclosure reachable."""
        requested = None if params.language == "auto" else params.language
        raw_language = getattr(result, "raw_model_language", None) or result.language
        detected, diagnostics = detected_language(raw_language, requested)
        if self.config.draft_dir is not None and (
            params.candidate_languages or params.phrase_hints
        ):
            diagnostics.append(
                Diagnostic(
                    code="draft_bypassed_for_guidance",
                    message="Token-score guidance uses the target model's full vocabulary scores.",
                )
            )
        projected = TranscriptionResult(
            text=result.text,
            detected_language=detected,
            duration=len(samples) / 16000,
            diagnostics=diagnostics,
            extra={"input_start_seconds": offset_seconds},
        )
        if (
            isinstance(params.provider_params, Qwen3ASRParams)
            and params.provider_params.include_metrics
        ):
            projected.extra["native"] = {
                "raw_text": getattr(result, "raw_text", result.text),
                "token_ids": list(getattr(result, "token_ids", ())),
                "audio_tokens": getattr(result, "audio_tokens", None),
                "timings": getattr(result, "timings", {}),
            }
        if params.word_timestamps is not None or params.diarization is not None:
            return self._auxiliary_models().annotate(
                projected,
                samples,
                params,
                offset_seconds,
                cancel=cancel,
                speaker_tracker=speaker_tracker,
            )
        return projected

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
    """Discoverable preset using a smaller native window for short-utterance latency."""

    config_type = ShortDictationConfig
    properties = Qwen3ASREngine.properties.model_copy(
        update={
            "model_name": "1.7b-short-dictation",
            "max_audio_duration": None,
        }
    )
    declared_metadata = DeclaredEngineMetadata.model_validate(
        {
            **Qwen3ASREngine.declared_metadata.model_dump(),
            "x_qwen3asr_streaming": {
                **Qwen3ASREngine.declared_metadata.model_dump()["x_qwen3asr_streaming"],
                "native_window_seconds": PROFILES["short-dictation"].max_audio_seconds,
            },
        }
    )


def _merge_chunk_results(chunks: list[TranscriptionResult], duration: float) -> TranscriptionResult:
    """Join complete windows in input order without manufacturing speech timestamps."""
    if len(chunks) == 1:
        return chunks[0].model_copy(update={"duration": duration})
    text = ""
    segments = []
    words = []
    for chunk in chunks:
        separator = (
            " " if text and chunk.text and needs_join_space(text[-1], chunk.text[0]) else ""
        )
        offset = len(text) + len(separator)
        chunk_segments = chunk.segments
        if chunk_segments is None:
            chunk_segments = [Segment(text=chunk.text, start=None, end=None, text_separator="")]
        for index, segment in enumerate(chunk_segments):
            segments.append(
                segment.model_copy(
                    update={
                        "text_separator": separator if index == 0 else segment.text_separator,
                        "extra": shift_source_offsets(segment.extra, offset),
                        "words": [
                            word.model_copy(
                                update={"extra": shift_source_offsets(word.extra, offset)}
                            )
                            for word in segment.words
                        ]
                        if segment.words is not None
                        else None,
                    }
                )
            )
        words.extend(
            word.model_copy(update={"extra": shift_source_offsets(word.extra, offset)})
            for word in (chunk.words or [])
        )
        text += separator + chunk.text
    composed = compose_segment_text(segments)
    if composed != text:
        raise RuntimeError("Window segments do not preserve the complete transcript text")
    languages = list(
        dict.fromkeys(chunk.detected_language for chunk in chunks if chunk.detected_language)
    )
    diagnostics = [item for chunk in chunks for item in chunk.diagnostics]
    if len(languages) > 1:
        diagnostics.append(
            Diagnostic(
                code="multiple_detected_languages",
                message="Different recording windows produced different detected languages.",
                param="language",
                provided=languages,
                effective=None,
            )
        )
    return TranscriptionResult(
        text=composed,
        detected_language=languages[0] if len(languages) == 1 else None,
        duration=duration,
        segments=segments,
        words=words if any(chunk.words is not None for chunk in chunks) else None,
        diagnostics=diagnostics,
        extra={"windows": [chunk.extra for chunk in chunks]},
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


def create_engine(**kwargs: object) -> Qwen3ASREngine:
    """Construct the registered Qwen3-ASR 1.7B preset without loading models."""
    return Qwen3ASREngine(**kwargs)
