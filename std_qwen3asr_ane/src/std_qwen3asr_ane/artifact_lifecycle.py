"""Artifact status and explicit acquisition for the Qwen3-ASR target and draft.

``ArtifactManager`` is constructed alongside an engine but performs no I/O in
its constructor. Status reads local metadata and payload layouts only. Model
downloads, source hashing, conversion, compilation, and worker-environment
creation occur only through explicit acquisition.
"""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from standard_asr.contract.exceptions import ArtifactAcquisitionError
from standard_asr.engine import (
    ArtifactAction,
    ArtifactContext,
    ArtifactProgress,
    ArtifactRequirement,
    Diagnostic,
    allow_downloads,
)

from .acquisition import (
    acquire_in_worker,
    acquisition_lock,
    conversion_worker_root,
    inspect_conversion_worker,
)
from .artifact_validation import (
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    TARGET_MODEL_ID,
    ArtifactManifestError,
    DraftManifestInfo,
    TargetManifestInfo,
    resolve_manifest_path,
    validate_draft_manifest,
    validate_target_manifest,
)
from .batching import BatchHeadArtifact, validate_target_bound_batch_head
from .bundle import digest
from .conversion.build import SOURCE_REVISION, download_verified_source
from .conversion.toolchain import conversion_toolchain_available
from .profiles import PROFILES
from .source_validation import (
    SourceInspection,
    SourceValidationError,
    VerifiedSource,
    inspect_source_checkpoint,
    verify_source_checkpoint,
)

BUNDLE_ARTIFACT_ID = "qwen3-asr-1.7b-coreml"
DRAFT_ARTIFACT_ID = "qwen3-asr-0.6b-gpu-draft"
BATCH_HEAD_ARTIFACT_ID = "qwen3-asr-1.7b-batch-head"
BUILD_RECIPE = {"cache_length": 1024, "token_batch_size": 16, "layers_per_partition": 14}
COMPRESS_RECIPE = {"scheme": "palette", "bits": 8, "group_size": 32}

ArtifactState = str
ProgressCallback = Callable[[ArtifactProgress], None]
Emit = Callable[..., None]


class ArtifactConfig(Protocol):
    """Configuration fields consumed by the target/draft artifact manager."""

    profile: str
    model_dir: Path
    source_dir: Path
    draft_dir: Path | None
    draft_source_dir: Path
    use_batching: bool
    batch_head_dir: Path | None


@dataclass(frozen=True)
class TargetStatus:
    state: ArtifactState
    revision: str | None
    info: TargetManifestInfo | None = None


@dataclass(frozen=True)
class DraftStatus:
    state: ArtifactState
    revision: str | None
    info: DraftManifestInfo | None = None


@dataclass(frozen=True)
class BatchHeadStatus:
    state: ArtifactState
    revision: str | None
    artifact: BatchHeadArtifact | None = None


class ArtifactManager:
    """Own target/draft status, source validation, acquisition, and remediation."""

    def __init__(self, config: ArtifactConfig, model_key: str) -> None:
        self.config = config
        self.model_key = model_key
        self._digest_cache: dict[tuple[object, ...], str] = {}
        self._digest_lock = Lock()
        self._verified_sources: dict[tuple[Path, str, str], VerifiedSource] = {}

    def pull_command(self) -> str:
        """Return a shell-safe remedy preserving this configured target/draft."""
        command = f"standard-asr pull {self.model_key}"
        settings: dict[str, object] = {
            "profile": self.config.profile,
            "model_dir": self.config.model_dir.expanduser().absolute(),
            "source_dir": self.config.source_dir.expanduser().absolute(),
        }
        if self.config.draft_dir is not None:
            settings["draft_dir"] = self.config.draft_dir.expanduser().absolute()
            settings["draft_source_dir"] = self.config.draft_source_dir.expanduser().absolute()
        batch_head_dir = getattr(self.config, "batch_head_dir", None)
        if batch_head_dir is not None:
            settings["use_batching"] = "true"
            settings["batch_head_dir"] = batch_head_dir.expanduser().absolute()
        for key, value in settings.items():
            command += " --set " + shlex.quote(f"{key}={value}")
        return command

    def requirements(
        self, context: ArtifactContext
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        """Return context-specific target/draft requirements without side effects."""
        target_root = self.config.model_dir.expanduser().resolve()
        target = self._inspect_target(target_root)
        target_requirement = ArtifactRequirement(
            artifact_id=BUNDLE_ARTIFACT_ID,
            label="Qwen3-ASR 1.7B local Core ML bundle",
            state=target.state,
            required_for_inference=True,
            may_acquire_during_inference=False,
            source_is_mutable=False,
            location=target_root,
            artifact_version=target.revision,
            **self._acquisition_gate(target.state, target_revision=SOURCE_REVISION),
        )
        requirements: list[ArtifactRequirement] = [target_requirement]
        draft_needed = not (
            self.config.draft_dir is None
            or context.mode == "streaming"
            or context.params.candidate_languages
            or context.params.phrase_hints
            or getattr(context.params.provider_params, "disable_draft", False)
        )
        target_revision = target.info.source_revision if target.info is not None else SOURCE_REVISION
        if draft_needed:
            assert self.config.draft_dir is not None
            draft_root = self.config.draft_dir.expanduser().resolve()
            draft = self._inspect_draft(draft_root, target, target_root)
            draft_gate = self._dependent_gate(
                target,
                draft.state,
                needs_draft=True,
                target_revision=target_revision,
            )
            requirements.append(
                ArtifactRequirement(
                    artifact_id=DRAFT_ARTIFACT_ID,
                    label="Qwen3-ASR 0.6B draft checkpoint and verify head",
                    state=draft.state,
                    required_for_inference=True,
                    may_acquire_during_inference=False,
                    source_is_mutable=False,
                    location=draft_root,
                    artifact_version=draft.revision,
                    **draft_gate,
                )
            )

        batch_head_dir = getattr(self.config, "batch_head_dir", None)
        if batch_head_dir is not None and context.mode != "streaming":
            batch_root = batch_head_dir.expanduser().resolve()
            batch_head = self._inspect_batch_head(batch_root, target, target_root)
            if target.info is not None and target.info.token_batch_size < 2:
                batch_gate: dict[str, object] = {
                    "can_acquire_now": False,
                    "acquisition_blocker": "action_required",
                    "required_actions": (
                        ArtifactAction(
                            kind="provide_artifacts",
                            message=(
                                "Select a target bundle with token_batch_size above one "
                                "before acquiring its compact batch head."
                            ),
                        ),
                    ),
                }
            else:
                batch_gate = self._dependent_gate(
                    target,
                    batch_head.state,
                    needs_draft=False,
                    target_revision=target_revision,
                )
            requirements.append(
                ArtifactRequirement(
                    artifact_id=BATCH_HEAD_ARTIFACT_ID,
                    label="Qwen3-ASR target-bound compact batch head",
                    state=batch_head.state,
                    required_for_inference=False,
                    may_acquire_during_inference=False,
                    source_is_mutable=False,
                    location=batch_root,
                    artifact_version=batch_head.revision,
                    **batch_gate,
                )
            )
        return True, tuple(requirements), ()

    def acquire(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: ProgressCallback | None,
    ) -> None:
        """Acquire the runnable target closure selected by the Standard template."""
        del refresh  # Both checkpoint references are immutable revisions.
        requested = {requirement.artifact_id for requirement in requirements}

        def emit(phase: str, artifact_id: str | None = None, **counts: object) -> None:
            if progress is not None:
                progress(ArtifactProgress(phase=phase, artifact_id=artifact_id, **counts))

        batch_head_dir = getattr(self.config, "batch_head_dir", None)
        extra_paths = (
            (batch_head_dir,)
            if batch_head_dir is not None and BATCH_HEAD_ARTIFACT_ID in requested
            else ()
        )
        self._verified_sources.clear()
        try:
            with acquisition_lock(
                self.config,
                include_draft=DRAFT_ARTIFACT_ID in requested,
                extra_paths=extra_paths,
            ):
                current = {item.artifact_id: item for item in self.requirements(context)[1]}
                targets = {
                    artifact_id
                    for artifact_id in requested
                    if artifact_id in current
                    and current[artifact_id].state != "ready"
                    and current[artifact_id].can_acquire_now
                }
                if not targets:
                    return
                if not conversion_toolchain_available():
                    acquire_in_worker(self.config, targets, progress)
                    return
                with redirect_stdout(sys.stderr):
                    if BUNDLE_ARTIFACT_ID in targets:
                        self.acquire_bundle(emit)
                    if BATCH_HEAD_ARTIFACT_ID in targets:
                        self.acquire_batch_head(emit)
                    if DRAFT_ARTIFACT_ID in targets:
                        self.acquire_draft(emit)
        except ArtifactAcquisitionError:
            raise
        except Exception as error:
            raise ArtifactAcquisitionError(
                "Building the local model bundle failed; inspect the conversion log.",
                reason="failed",
                hint=f"After resolving the reported failure, retry: {self.pull_command()}",
            ) from error
        finally:
            self._verified_sources.clear()

    def ensure_source(
        self,
        emit: Emit,
        revision: str = SOURCE_REVISION,
    ) -> VerifiedSource:
        """Return one fully verified 1.7B source, downloading only when absent."""
        return self._ensure_checkpoint_source(
            self.config.source_dir,
            field="source_dir",
            model_id=TARGET_MODEL_ID,
            revision=revision,
            artifact_id=BUNDLE_ARTIFACT_ID,
            emit=emit,
        )

    def acquire_bundle(self, emit: Emit) -> None:
        """Build, validate, and atomically publish the configured target bundle."""
        from .compiled import compile_bundle
        from .conversion.build import build_bundle
        from .conversion.compress import compress_bundle

        target = self.config.model_dir.expanduser().resolve()
        if target.exists():
            raise ArtifactAcquisitionError(
                f"{target} already exists; move it away before pulling.",
                reason="action_required",
            )
        verified_source = self.ensure_source(emit)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as directory:
            work = Path(directory)
            recipe = dict(BUILD_RECIPE)
            if self.config.profile != "general":
                recipe.update(
                    profile=self.config.profile,
                    cache_length=PROFILES[self.config.profile].cache_length,
                )
            emit("converting", BUNDLE_ARTIFACT_ID)
            build_bundle(
                verified_source.root,
                work / "fp16",
                **recipe,
                verified_source=verified_source,
            )
            emit("converting", BUNDLE_ARTIFACT_ID)
            compress_bundle(work / "fp16", work / "lut8", **COMPRESS_RECIPE)
            emit("converting", BUNDLE_ARTIFACT_ID)
            compile_bundle(work / "lut8", work / "compiled")
            emit("verifying", BUNDLE_ARTIFACT_ID)
            staged = self._inspect_target(work / "compiled")
            if staged.state != "ready":
                raise ArtifactAcquisitionError(
                    "The converted target failed final artifact validation.",
                    reason="failed",
                )
            (work / "compiled").rename(target)

    def acquire_draft(self, emit: Emit) -> None:
        """Build, validate, and atomically publish a draft bound to the target."""
        from .conversion.draft import build_draft_bundle

        if self.config.draft_dir is None:
            raise ArtifactAcquisitionError(
                "No draft_dir is configured for this engine.", reason="unsupported"
            )
        target_root = self.config.model_dir.expanduser().resolve()
        target_status = self._inspect_target(target_root)
        if target_status.state != "ready" or target_status.info is None:
            raise ArtifactAcquisitionError(
                "The target bundle must be ready before building its draft.",
                reason="failed",
            )
        draft = self.config.draft_dir.expanduser().resolve()
        if draft.exists():
            raise ArtifactAcquisitionError(
                f"{draft} already exists; move it away before pulling.",
                reason="action_required",
            )
        verified_target_source = self.ensure_source(
            emit, revision=target_status.info.source_revision
        )
        verified_draft_source = self._ensure_draft_source(emit)
        draft.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{draft.name}-", dir=draft.parent) as directory:
            staged = Path(directory) / "bundle"
            emit("converting", DRAFT_ARTIFACT_ID)
            build_draft_bundle(
                target_root,
                verified_target_source.root,
                staged,
                draft_source=verified_draft_source.root,
                verified_target_source=verified_target_source,
                verified_draft_source=verified_draft_source,
            )
            emit("verifying", DRAFT_ARTIFACT_ID)
            staged_status = self._inspect_draft(staged, target_status, target_root)
            if staged_status.state != "ready":
                raise ArtifactAcquisitionError(
                    "The converted draft failed final target-binding validation.",
                    reason="failed",
                )
            staged.rename(draft)

    def acquire_batch_head(self, emit: Emit) -> None:
        """Build and atomically publish an optional head bound to the target."""
        from .conversion.batch_head import build_batch_head

        batch_head_dir = getattr(self.config, "batch_head_dir", None)
        if batch_head_dir is None:
            raise ArtifactAcquisitionError(
                "No batch_head_dir is configured for this engine.", reason="unsupported"
            )
        target_root = self.config.model_dir.expanduser().resolve()
        target_status = self._inspect_target(target_root)
        if target_status.state != "ready" or target_status.info is None:
            raise ArtifactAcquisitionError(
                "The target bundle must be ready before building its batch head.",
                reason="failed",
            )
        output = batch_head_dir.expanduser().resolve()
        if output.exists():
            raise ArtifactAcquisitionError(
                f"{output} already exists; move it away before pulling.",
                reason="action_required",
            )
        verified_source = self.ensure_source(
            emit, revision=target_status.info.source_revision
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
            staged = Path(directory) / "bundle"
            emit("converting", BATCH_HEAD_ARTIFACT_ID)
            build_batch_head(
                target_root,
                verified_source.root,
                staged,
                verified_source=verified_source,
            )
            emit("verifying", BATCH_HEAD_ARTIFACT_ID)
            staged_status = self._inspect_batch_head(
                staged, target_status, target_root
            )
            if staged_status.state != "ready":
                raise ArtifactAcquisitionError(
                    "The converted batch head failed final target-binding validation.",
                    reason="failed",
                )
            staged.rename(output)

    def _ensure_draft_source(self, emit: Emit) -> VerifiedSource:
        return self._ensure_checkpoint_source(
            self.config.draft_source_dir,
            field="draft_source_dir",
            model_id=DRAFT_MODEL_ID,
            revision=DRAFT_REVISION,
            artifact_id=DRAFT_ARTIFACT_ID,
            emit=emit,
        )

    def _ensure_checkpoint_source(
        self,
        source: Path,
        *,
        field: str,
        model_id: str,
        revision: str,
        artifact_id: str,
        emit: Emit,
    ) -> VerifiedSource:
        source = source.expanduser().resolve()
        cache_key = (source, model_id, revision)
        cached = self._verified_sources.get(cache_key)
        if cached is not None:
            return cached
        inspection = inspect_source_checkpoint(
            source,
            expected_model_id=model_id,
            expected_revision=revision,
        )
        if inspection.state == "missing":
            if not allow_downloads():
                raise ArtifactAcquisitionError(
                    "Checkpoint download is disabled by STANDARD_ASR_ALLOW_DOWNLOAD.",
                    reason="downloads_disabled",
                )
            emit("transferring", artifact_id)
            verified = download_verified_source(
                source,
                revision=revision,
                model_id=model_id,
                progress=lambda completed, total: emit(
                    "verifying",
                    artifact_id,
                    completed_units=completed,
                    total_units=total,
                    unit="bytes",
                ),
            )
            self._verified_sources[cache_key] = verified
            return verified
        if not inspection.usable:
            raise self._source_action_error(field, inspection)
        try:
            verified = verify_source_checkpoint(
                source,
                expected_model_id=model_id,
                expected_revision=revision,
                progress=lambda completed, total: emit(
                    "verifying",
                    artifact_id,
                    completed_units=completed,
                    total_units=total,
                    unit="bytes",
                ),
            )
            self._verified_sources[cache_key] = verified
            return verified
        except SourceValidationError as error:
            raise self._source_action_error(
                field, SourceInspection("corrupt", str(error))
            ) from error

    def _source_action_error(
        self, field: str, inspection: SourceInspection
    ) -> ArtifactAcquisitionError:
        return ArtifactAcquisitionError(
            f"{field} is not a complete matching checkpoint: {inspection.reason}",
            reason="action_required",
            hint=f"Move the directory away or select a valid {field}, then retry: {self.pull_command()}",
            required_actions=(
                ArtifactAction(
                    kind="provide_artifacts",
                    message=f"Provide a complete matching {field}.",
                ),
            ),
        )

    def _dependent_gate(
        self,
        target: TargetStatus,
        state: str,
        *,
        needs_draft: bool,
        target_revision: str,
    ) -> dict[str, object]:
        if target.state in {"incomplete", "corrupt"}:
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "action_required",
                "required_actions": (
                    ArtifactAction(
                        kind="provide_artifacts",
                        message="Repair or replace the target bundle before acquiring its auxiliary head.",
                    ),
                ),
            }
        return self._acquisition_gate(
            state,
            needs_draft=needs_draft,
            target_revision=target_revision,
        )

    def _acquisition_gate(
        self,
        state: str,
        *,
        needs_draft: bool = False,
        target_revision: str,
    ) -> dict[str, object]:
        if state == "ready":
            return {"can_acquire_now": False, "acquisition_blocker": None, "required_actions": ()}
        if state in {"incomplete", "corrupt"}:
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "action_required",
                "required_actions": (
                    ArtifactAction(
                        kind="provide_artifacts",
                        message=(
                            f"The directory is {state}; move it away, then run "
                            f"{self.pull_command()}"
                        ),
                    ),
                ),
            }

        source_checks = [
            (
                "source_dir",
                inspect_source_checkpoint(
                    self.config.source_dir,
                    expected_model_id=TARGET_MODEL_ID,
                    expected_revision=target_revision,
                ),
            )
        ]
        if needs_draft:
            source_checks.append(
                (
                    "draft_source_dir",
                    inspect_source_checkpoint(
                        self.config.draft_source_dir,
                        expected_model_id=DRAFT_MODEL_ID,
                        expected_revision=DRAFT_REVISION,
                    ),
                )
            )
        invalid = [(field, item) for field, item in source_checks if item.state in {"incomplete", "corrupt"}]
        if invalid:
            field, inspection = invalid[0]
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "action_required",
                "required_actions": (
                    ArtifactAction(
                        kind="provide_artifacts",
                        message=f"Provide a complete matching {field}: {inspection.reason}",
                    ),
                ),
            }
        sources_need_download = any(item.state == "missing" for _, item in source_checks)
        worker = None if conversion_toolchain_available() else inspect_conversion_worker(self.config)
        if worker is not None and not worker.ready and worker.reason != "missing":
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "action_required",
                "required_actions": (
                    ArtifactAction(
                        kind="install_external",
                        message=(
                            "Move the incompatible managed conversion environment at "
                            f"{conversion_worker_root(self.config)} away, then retry explicit acquisition."
                        ),
                    ),
                ),
            }
        worker_needs_download = worker is not None and not worker.ready
        if not allow_downloads() and (sources_need_download or worker_needs_download):
            return {
                "can_acquire_now": False,
                "acquisition_blocker": "downloads_disabled",
                "required_actions": (),
            }
        return {"can_acquire_now": True, "acquisition_blocker": None, "required_actions": ()}

    def _inspect_target(self, root: Path) -> TargetStatus:
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return TargetStatus("incomplete" if root.exists() else "missing", None)
        try:
            if not manifest_path.resolve().is_relative_to(root):
                raise ArtifactManifestError("target manifest escapes the bundle")
            manifest = json.loads(manifest_path.read_text())
            info = validate_target_manifest(manifest, profile=self.config.profile)
            for index, relative in enumerate(info.payload_paths):
                payload = resolve_manifest_path(root, relative, field=f"payload[{index}]")
                state = self._inspect_payload(payload)
                if state != "ready":
                    return TargetStatus(state, info.source_revision, info)
            return TargetStatus("ready", info.source_revision, info)
        except (json.JSONDecodeError, UnicodeError, ArtifactManifestError, TypeError, ValueError):
            return TargetStatus("corrupt", None)
        except FileNotFoundError:
            return TargetStatus("incomplete", None)

    def _inspect_draft(
        self,
        root: Path,
        target: TargetStatus,
        target_root: Path,
    ) -> DraftStatus:
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return DraftStatus("incomplete" if root.exists() else "missing", None)
        if target.info is None:
            return DraftStatus("unknown", None)
        try:
            if not manifest_path.resolve().is_relative_to(root):
                raise ArtifactManifestError("draft manifest escapes the bundle")
            manifest = json.loads(manifest_path.read_text())
            info = validate_draft_manifest(
                manifest,
                target=target.info,
                target_root=target_root,
            )
            head = resolve_manifest_path(root, info.verify_head_path, field="verify_head.path")
            state = self._inspect_payload(head)
            if state != "ready":
                return DraftStatus(state, info.revision, info)
            checkpoint = resolve_manifest_path(root, info.draft_path, field="draft.path")
            source = inspect_source_checkpoint(
                checkpoint,
                expected_model_id=DRAFT_MODEL_ID,
                expected_revision=DRAFT_REVISION,
                expected_content_sha256=info.draft_content_sha256,
            )
            if source.state != "ready":
                state = "incomplete" if source.state in {"missing", "incomplete"} else "corrupt"
                return DraftStatus(state, info.revision, info)
            if not self._draft_weights_match(info, root, target.info, target_root):
                return DraftStatus("corrupt", info.revision, info)
            return DraftStatus("ready", info.revision, info)
        except (json.JSONDecodeError, UnicodeError, ArtifactManifestError, TypeError, ValueError):
            return DraftStatus("corrupt", None)
        except FileNotFoundError:
            return DraftStatus("incomplete", None)

    def _inspect_batch_head(
        self,
        root: Path,
        target: TargetStatus,
        target_root: Path,
    ) -> BatchHeadStatus:
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return BatchHeadStatus("incomplete" if root.exists() else "missing", None)
        if target.info is None:
            return BatchHeadStatus("unknown", None)
        try:
            artifact = validate_target_bound_batch_head(
                root,
                target=target.info,
                target_root=target_root,
                weight_digest_resolver=self._cached_weight_digests,
            )
            state = self._inspect_payload(artifact.path)
            if state != "ready":
                return BatchHeadStatus(state, target.info.source_revision, artifact)
            return BatchHeadStatus("ready", target.info.source_revision, artifact)
        except (OSError, UnicodeError, ArtifactManifestError, TypeError, ValueError):
            return BatchHeadStatus("corrupt", target.info.source_revision)

    def _draft_weights_match(
        self,
        draft: DraftManifestInfo,
        draft_root: Path,
        target: TargetManifestInfo,
        target_root: Path,
    ) -> bool:
        head = resolve_manifest_path(draft_root, draft.verify_head_path, field="verify_head.path")
        target_head = resolve_manifest_path(
            target_root, target.files["lm_head"], field="files.lm_head"
        )
        draft_hashes = self._cached_weight_digests(head)
        target_hashes = self._cached_weight_digests(target_head)
        return bool(draft_hashes) and (
            draft_hashes == draft.declared_weight_sha256 == target_hashes
        )

    def _cached_weight_digests(self, package: Path) -> tuple[str, ...]:
        package = Path(package).resolve()
        weights = sorted(package.rglob("weight.bin"))
        values: list[str] = []
        with self._digest_lock:
            for path in weights:
                resolved = path.resolve()
                if not resolved.is_relative_to(package):
                    raise ArtifactManifestError("weight payload escapes its model package")
                stat = resolved.stat()
                key = (
                    str(resolved),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
                value = self._digest_cache.get(key)
                if value is None:
                    value = digest(resolved)
                    final_stat = resolved.stat()
                    final_key = (
                        str(resolved),
                        final_stat.st_dev,
                        final_stat.st_ino,
                        final_stat.st_size,
                        final_stat.st_mtime_ns,
                        final_stat.st_ctime_ns,
                    )
                    if final_key != key:
                        raise ArtifactManifestError(
                            "weight payload changed during binding verification"
                        )
                    self._digest_cache[key] = value
                values.append(value)
        return tuple(values)

    def _inspect_payload(self, payload: Path) -> str:
        if payload.suffix == ".mlpackage":
            return self._inspect_package(payload)
        if payload.suffix == ".mlmodelc":
            return self._inspect_compiled_package(payload)
        try:
            return "ready" if payload.is_file() and payload.stat().st_size > 0 else "incomplete"
        except FileNotFoundError:
            return "incomplete"

    @staticmethod
    def _inspect_package(package: Path) -> str:
        manifest_path = package / "Manifest.json"
        try:
            if not manifest_path.resolve().is_relative_to(package):
                return "corrupt"
            if not manifest_path.is_file():
                return "incomplete"
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, Mapping):
                return "corrupt"
            entries = manifest.get("itemInfoEntries")
            root_id = manifest.get("rootModelIdentifier")
            version = manifest.get("fileFormatVersion")
            if (
                not isinstance(entries, Mapping)
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
                if not isinstance(entry, Mapping):
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

    @staticmethod
    def _inspect_compiled_package(package: Path) -> str:
        if not package.is_dir():
            return "incomplete"
        try:
            for child in package.rglob("*"):
                if not child.resolve().is_relative_to(package):
                    return "corrupt"
            required = (
                package / "coremldata.bin",
                package / "model.mil",
                package / "weights/weight.bin",
            )
            if any(not path.is_file() or path.stat().st_size == 0 for path in required):
                return "incomplete"
        except FileNotFoundError:
            return "incomplete"
        return "ready"
