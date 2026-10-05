"""Small manifest validation must agree with native target and draft requirements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from std_qwen3asr_ane.artifact_validation import (
    DRAFT_BUNDLE_KIND,
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    TARGET_MODEL_ID,
    ArtifactManifestError,
    validate_draft_manifest,
    validate_target_manifest,
    verify_draft_weight_binding,
)
from std_qwen3asr_ane.bundle import digest, lm_head_compression


def target_manifest(**changes) -> dict:
    manifest = {
        "schema_version": 1,
        "model_id": TARGET_MODEL_ID,
        "source_revision": "a" * 40,
        "files": {
            "frontend": "frontend.mlpackage",
            "encoder": "encoder.mlpackage",
            "lm_head": "lm_head.mlmodelc",
            "embedding": "embedding.npy",
            "tokenizer": "tokenizer.json",
            "mel_filters": "mel_filters.npy",
        },
        "decoder_partitions": ["decoder_00.mlmodelc"],
        "max_sequence_length": 1024,
        "token_batch_size": 16,
        "max_audio_seconds": 30,
        "residual_scale": 1,
        "head_dim": 128,
        "rope_theta": 1_000_000,
        "frontend": {"chunk_frames": 100},
        "encoder": {"window_tokens": 104},
    }
    manifest.update(changes)
    return manifest


def draft_manifest(target: dict, target_root: Path) -> dict:
    return {
        "schema_version": 1,
        "kind": DRAFT_BUNDLE_KIND,
        "draft": {
            "model_id": DRAFT_MODEL_ID,
            "revision": DRAFT_REVISION,
            "path": "Qwen3-ASR-0.6B",
        },
        "verify_head": {
            "path": "verify_head.mlmodelc",
            "token_batch_size": target["token_batch_size"],
            "vocabulary_chunk": 128,
            "weight_sha256": ["0" * 64],
        },
        "target": {
            "model_id": target["model_id"],
            "source_revision": target["source_revision"],
            "token_batch_size": target["token_batch_size"],
            "tokenizer_sha256": digest(target_root / target["files"]["tokenizer"]),
            "weight_compression": lm_head_compression(target),
            "manifest_sha256": digest(target_root / "manifest.json"),
        },
    }


@pytest.mark.parametrize(
    "field",
    [
        "decoder_partitions",
        "max_sequence_length",
        "max_audio_seconds",
        "residual_scale",
        "head_dim",
        "rope_theta",
    ],
)
def test_target_rejects_missing_runtime_fields(field: str):
    manifest = target_manifest()
    del manifest[field]
    with pytest.raises(ArtifactManifestError, match=field):
        validate_target_manifest(manifest)


def test_target_validates_partition_closure_and_profile():
    with pytest.raises(ArtifactManifestError, match="nonempty"):
        validate_target_manifest(target_manifest(decoder_partitions=[]))
    with pytest.raises(ArtifactManifestError, match="duplicate"):
        validate_target_manifest(
            target_manifest(decoder_partitions=["decoder.mlmodelc", "decoder.mlmodelc"])
        )
    with pytest.raises(ArtifactManifestError, match="short-dictation"):
        validate_target_manifest(target_manifest(), profile="short-dictation")
    short = validate_target_manifest(
        target_manifest(max_sequence_length=512, max_audio_seconds=12),
        profile="short-dictation",
    )
    assert short.cache_length == 512 and short.max_audio_seconds == 12


def test_draft_rejects_known_identity_and_target_mismatches(tmp_path: Path):
    target = target_manifest()
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps(target))
    target_info = validate_target_manifest(target)
    good = draft_manifest(target, tmp_path)
    assert validate_draft_manifest(good, target=target_info, target_root=tmp_path).revision

    wrong_model = json.loads(json.dumps(good))
    wrong_model["draft"]["model_id"] = "other/model"
    with pytest.raises(ArtifactManifestError, match="draft.model_id"):
        validate_draft_manifest(wrong_model, target=target_info, target_root=tmp_path)

    wrong_target = json.loads(json.dumps(good))
    wrong_target["target"]["token_batch_size"] = 8
    with pytest.raises(ArtifactManifestError, match="target.token_batch_size"):
        validate_draft_manifest(wrong_target, target=target_info, target_root=tmp_path)


def test_explicit_weight_binding_hashes_both_heads(tmp_path: Path):
    target_root, draft_root = tmp_path / "target", tmp_path / "draft"
    target_root.mkdir()
    draft_root.mkdir()
    target = target_manifest()
    (target_root / "tokenizer.json").write_text("{}")
    (target_root / "manifest.json").write_text(json.dumps(target))
    target_weight = target_root / "lm_head.mlmodelc/weights/weight.bin"
    target_weight.parent.mkdir(parents=True)
    target_weight.write_bytes(b"shared")

    draft_weight = draft_root / "verify_head.mlmodelc/weights/weight.bin"
    draft_weight.parent.mkdir(parents=True)
    draft_weight.write_bytes(b"shared")
    declared = digest(target_weight)
    manifest = draft_manifest(target, target_root)
    manifest["verify_head"]["weight_sha256"] = [declared]
    target_info = validate_target_manifest(target)
    draft_info = validate_draft_manifest(
        manifest, target=target_info, target_root=target_root
    )
    verified = verify_draft_weight_binding(
        draft_info,
        draft_root=draft_root,
        target=target_info,
        target_root=target_root,
    )
    assert verified.payload_binding_verified

    draft_weight.write_bytes(b"changed")
    with pytest.raises(ArtifactManifestError, match="declared"):
        verify_draft_weight_binding(
            draft_info,
            draft_root=draft_root,
            target=target_info,
            target_root=target_root,
        )
