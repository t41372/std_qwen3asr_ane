"""Side-effect-free validation for target and speculative-draft artifacts.

The functions in this module inspect small manifests and, when a root is
provided, small binding files such as the tokenizer and manifest. They never
import Core ML or MLX, initialize a device, or contact a remote service. Large
weight hashing is an explicit separate operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import digest, language_head_output, lm_head_compression, offline_frontend_batch_size
from .embedding import embedding_quantization
from .profiles import ProfileName

TARGET_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DRAFT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
DRAFT_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
DRAFT_BUNDLE_KIND = "qwen3-asr-ane-draft"

_REQUIRED_FILE_ROLES = {
    "frontend",
    "encoder",
    "lm_head",
    "embedding",
    "tokenizer",
    "mel_filters",
}
_SHA256_LENGTH = 64


class ArtifactManifestError(ValueError):
    """A small artifact manifest is incompatible with this runtime."""


@dataclass(frozen=True)
class TargetManifestInfo:
    """Validated target metadata needed by status, conversion, and runtime."""

    manifest: Mapping[str, Any]
    schema_version: int
    model_id: str
    source_revision: str
    source_content_sha256: str | None
    files: Mapping[str, str]
    decoder_partitions: tuple[str, ...]
    cache_length: int
    token_batch_size: int
    max_audio_seconds: float
    residual_scale: float
    head_dim: int
    rope_theta: float
    frontend_batch_size: int

    @property
    def payload_paths(self) -> tuple[str, ...]:
        """Every manifest-relative payload consumed by the runtime."""
        return tuple(dict.fromkeys((*self.files.values(), *self.decoder_partitions)))


@dataclass(frozen=True)
class DraftManifestInfo:
    """Validated cheap binding between one draft and one target manifest."""

    manifest: Mapping[str, Any]
    draft_path: str
    verify_head_path: str
    revision: str
    draft_content_sha256: str | None
    declared_weight_sha256: tuple[str, ...]
    payload_binding_verified: bool = False


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactManifestError(f"{field} must be an object")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ArtifactManifestError(f"{field} must be a positive integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactManifestError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ArtifactManifestError(f"{field} must be a positive finite number")
    return result


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactManifestError(f"{field} must be a nonempty string")
    return value


def _sha256(value: object, field: str) -> str:
    text = _nonempty_string(value, field)
    if len(text) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in text):
        raise ArtifactManifestError(f"{field} must be a lowercase SHA-256 digest")
    return text


def validate_relative_manifest_path(value: object, field: str) -> str:
    """Validate a portable relative path before combining it with a bundle root."""
    relative = _nonempty_string(value, field)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ArtifactManifestError(f"{field} must stay inside the artifact root")
    return relative


def resolve_manifest_path(root: Path, relative: object, *, field: str) -> Path:
    """Resolve a validated manifest path and reject symlink escapes."""
    root = Path(root).expanduser().resolve()
    path = (root / validate_relative_manifest_path(relative, field)).resolve()
    if path == root or not path.is_relative_to(root):
        raise ArtifactManifestError(f"{field} escapes the artifact root")
    return path


def validate_target_manifest(
    manifest: object,
    *,
    profile: ProfileName | None = None,
) -> TargetManifestInfo:
    """Validate every cheap target field used before or during native loading."""
    data = _mapping(manifest, "target manifest")
    try:
        language_head_output(dict(data))
        frontend_batch_size = offline_frontend_batch_size(dict(data))
        quantization = embedding_quantization(dict(data))
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactManifestError(str(error)) from error

    schema_version = data.get("schema_version")
    if type(schema_version) is not int:
        raise ArtifactManifestError("schema_version must be an integer")
    if data.get("model_id") != TARGET_MODEL_ID:
        raise ArtifactManifestError(f"model_id must be {TARGET_MODEL_ID}")
    source_revision = _nonempty_string(data.get("source_revision"), "source_revision")
    source_content = data.get("source_content_sha256")
    if source_content is not None:
        source_content = _sha256(source_content, "source_content_sha256")

    files_value = _mapping(data.get("files"), "files")
    files = {
        _nonempty_string(role, "files role"): validate_relative_manifest_path(
            relative, f"files.{role}"
        )
        for role, relative in files_value.items()
    }
    required_roles = set(_REQUIRED_FILE_ROLES)
    if frontend_batch_size > 1:
        required_roles.add("frontend_batched")
    if quantization is not None:
        required_roles.add("embedding_scales")
    missing_roles = sorted(required_roles - files.keys())
    if missing_roles:
        raise ArtifactManifestError(f"files is missing required roles: {', '.join(missing_roles)}")

    partitions_value = data.get("decoder_partitions")
    if not isinstance(partitions_value, list) or not partitions_value:
        raise ArtifactManifestError("decoder_partitions must be a nonempty list")
    decoder_partitions = tuple(
        validate_relative_manifest_path(value, f"decoder_partitions[{index}]")
        for index, value in enumerate(partitions_value)
    )
    if len(decoder_partitions) != len(set(decoder_partitions)):
        raise ArtifactManifestError("decoder_partitions must not contain duplicate paths")

    cache_length = _positive_int(data.get("max_sequence_length"), "max_sequence_length")
    token_batch_size = _positive_int(data.get("token_batch_size", 1), "token_batch_size")
    if token_batch_size > cache_length:
        raise ArtifactManifestError("token_batch_size cannot exceed max_sequence_length")
    max_audio_seconds = _positive_float(data.get("max_audio_seconds"), "max_audio_seconds")
    residual_scale = _positive_float(data.get("residual_scale"), "residual_scale")
    head_dim = _positive_int(data.get("head_dim"), "head_dim")
    if head_dim < 2 or head_dim % 2:
        raise ArtifactManifestError("head_dim must be an even integer of at least 2")
    rope_theta = _positive_float(data.get("rope_theta"), "rope_theta")

    frontend = _mapping(data.get("frontend", {}), "frontend")
    if _positive_int(frontend.get("chunk_frames", 100), "frontend.chunk_frames") != 100:
        raise ArtifactManifestError("frontend.chunk_frames must be 100")
    encoder = _mapping(data.get("encoder", {}), "encoder")
    if _positive_int(encoder.get("window_tokens", 104), "encoder.window_tokens") != 104:
        raise ArtifactManifestError("encoder.window_tokens must be 104")
    if "prompt_template" in data:
        template = data["prompt_template"]
        if not isinstance(template, str):
            raise ArtifactManifestError("prompt_template must be a string")
        if template.count("<|audio_pad|>") != 1:
            raise ArtifactManifestError(
                "prompt_template must contain exactly one audio placeholder"
            )
        if template.count("<|im_start|>system\n<|im_end|>") != 1:
            raise ArtifactManifestError(
                "prompt_template must contain one unambiguous empty system slot"
            )

    if profile == "short-dictation" and (
        cache_length != 512 or max_audio_seconds > 12
    ):
        raise ArtifactManifestError(
            "short-dictation requires max_sequence_length=512 and max_audio_seconds<=12"
        )

    return TargetManifestInfo(
        manifest=data,
        schema_version=schema_version,
        model_id=TARGET_MODEL_ID,
        source_revision=source_revision,
        source_content_sha256=source_content,
        files=files,
        decoder_partitions=decoder_partitions,
        cache_length=cache_length,
        token_batch_size=token_batch_size,
        max_audio_seconds=max_audio_seconds,
        residual_scale=residual_scale,
        head_dim=head_dim,
        rope_theta=rope_theta,
        frontend_batch_size=frontend_batch_size,
    )


def validate_draft_manifest(
    manifest: object,
    *,
    target: TargetManifestInfo,
    target_root: Path | None = None,
) -> DraftManifestInfo:
    """Validate draft identity and its cheap binding to a selected target."""
    data = _mapping(manifest, "draft manifest")
    if data.get("schema_version") != 1 or data.get("kind") != DRAFT_BUNDLE_KIND:
        raise ArtifactManifestError("unsupported draft bundle identity or schema")

    draft = _mapping(data.get("draft"), "draft")
    if draft.get("model_id") != DRAFT_MODEL_ID:
        raise ArtifactManifestError(f"draft.model_id must be {DRAFT_MODEL_ID}")
    if draft.get("revision") != DRAFT_REVISION:
        raise ArtifactManifestError(f"draft.revision must be {DRAFT_REVISION}")
    draft_path = validate_relative_manifest_path(draft.get("path"), "draft.path")
    draft_content = draft.get("content_sha256")
    if draft_content is not None:
        draft_content = _sha256(draft_content, "draft.content_sha256")

    verify_head = _mapping(data.get("verify_head"), "verify_head")
    verify_head_path = validate_relative_manifest_path(verify_head.get("path"), "verify_head.path")
    width = _positive_int(verify_head.get("token_batch_size"), "verify_head.token_batch_size")
    if width != target.token_batch_size:
        raise ArtifactManifestError("verify_head.token_batch_size differs from the target")
    _positive_int(verify_head.get("vocabulary_chunk"), "verify_head.vocabulary_chunk")
    declared_hashes = verify_head.get("weight_sha256")
    if not isinstance(declared_hashes, list) or not declared_hashes:
        raise ArtifactManifestError("verify_head.weight_sha256 must be a nonempty list")
    weight_sha256 = tuple(
        _sha256(value, f"verify_head.weight_sha256[{index}]")
        for index, value in enumerate(declared_hashes)
    )

    expected = _mapping(data.get("target"), "target")
    comparisons = {
        "model_id": target.model_id,
        "source_revision": target.source_revision,
        "token_batch_size": target.token_batch_size,
        "weight_compression": lm_head_compression(dict(target.manifest)),
    }
    if target.source_content_sha256 is not None:
        comparisons["source_content_sha256"] = target.source_content_sha256
    for field, actual in comparisons.items():
        if expected.get(field) != actual:
            raise ArtifactManifestError(f"target.{field} differs from the selected target")

    if target_root is not None:
        tokenizer = resolve_manifest_path(
            target_root, target.files["tokenizer"], field="files.tokenizer"
        )
        if not tokenizer.is_file():
            raise ArtifactManifestError("the target tokenizer is missing")
        if expected.get("tokenizer_sha256") != digest(tokenizer):
            raise ArtifactManifestError("target.tokenizer_sha256 differs from the selected target")
        manifest_path = Path(target_root).expanduser().resolve() / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactManifestError("the target manifest is missing")
        # Builders record the bytes written to disk. Canonical fallback keeps the
        # pure manifest API useful when callers have not supplied a root.
        actual_manifest_digest = digest(manifest_path)
        if expected.get("manifest_sha256") != actual_manifest_digest:
            raise ArtifactManifestError("target.manifest_sha256 differs from the selected target")
    else:
        _sha256(expected.get("tokenizer_sha256"), "target.tokenizer_sha256")
        _sha256(expected.get("manifest_sha256"), "target.manifest_sha256")

    return DraftManifestInfo(
        manifest=data,
        draft_path=draft_path,
        verify_head_path=verify_head_path,
        revision=DRAFT_REVISION,
        draft_content_sha256=draft_content,
        declared_weight_sha256=weight_sha256,
    )


def _weight_digests(package: Path) -> tuple[str, ...]:
    paths = sorted(Path(package).rglob("weight.bin"))
    if not paths:
        raise ArtifactManifestError(f"{package} contains no weight.bin payload")
    return tuple(digest(path) for path in paths)


def verify_draft_weight_binding(
    draft: DraftManifestInfo,
    *,
    draft_root: Path,
    target: TargetManifestInfo,
    target_root: Path,
) -> DraftManifestInfo:
    """Hash both heads and prove the draft verifier uses target-identical weights."""
    verify_head = resolve_manifest_path(
        draft_root, draft.verify_head_path, field="verify_head.path"
    )
    target_head = resolve_manifest_path(
        target_root, target.files["lm_head"], field="files.lm_head"
    )
    draft_hashes = _weight_digests(verify_head)
    target_hashes = _weight_digests(target_head)
    if draft_hashes != draft.declared_weight_sha256:
        raise ArtifactManifestError("verify-head weights differ from their declared digests")
    if draft_hashes != target_hashes:
        raise ArtifactManifestError("verify-head weights differ from target lm_head weights")
    return DraftManifestInfo(
        manifest=draft.manifest,
        draft_path=draft.draft_path,
        verify_head_path=draft.verify_head_path,
        revision=draft.revision,
        draft_content_sha256=draft.draft_content_sha256,
        declared_weight_sha256=draft.declared_weight_sha256,
        payload_binding_verified=True,
    )
