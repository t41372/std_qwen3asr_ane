"""The target/draft artifact manager is pure at construction and honest at status."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from standard_asr.engine import ArtifactContext, ProviderParams, RuntimeParams

from std_qwen3asr_ane.artifact_lifecycle import (
    BATCH_HEAD_ARTIFACT_ID,
    BUNDLE_ARTIFACT_ID,
    DRAFT_ARTIFACT_ID,
    ArtifactManager,
)
from std_qwen3asr_ane.artifact_validation import DRAFT_MODEL_ID, DRAFT_REVISION, TARGET_MODEL_ID
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.build import SOURCE_REVISION


def config(root: Path, *, draft: bool = False, batching: bool = False):
    return SimpleNamespace(
        profile="general",
        model_dir=root / "target",
        source_dir=root / "source-1.7b",
        draft_dir=root / "draft" if draft else None,
        draft_source_dir=root / "source-0.6b",
        use_batching=batching,
        batch_head_dir=root / "batch-head" if batching else None,
    )


def compiled(path: Path, weight: bytes = b"weight") -> None:
    (path / "weights").mkdir(parents=True)
    (path / "coremldata.bin").write_bytes(b"metadata")
    (path / "model.mil").write_bytes(b"program")
    (path / "weights/weight.bin").write_bytes(weight)


def target(root: Path, *, head_weight: bytes = b"shared") -> dict:
    root.mkdir()
    files = {
        "frontend": "frontend.mlmodelc",
        "encoder": "encoder.mlmodelc",
        "lm_head": "lm_head.mlmodelc",
        "embedding": "embedding.npy",
        "tokenizer": "tokenizer.json",
        "mel_filters": "mel_filters.npy",
    }
    compiled(root / files["frontend"])
    compiled(root / files["encoder"])
    compiled(root / files["lm_head"], head_weight)
    compiled(root / "decoder.mlmodelc")
    for role in ("embedding", "tokenizer", "mel_filters"):
        (root / files[role]).write_bytes(b"{}")
    manifest = {
        "schema_version": 1,
        "model_id": TARGET_MODEL_ID,
        "source_revision": SOURCE_REVISION,
        "files": files,
        "decoder_partitions": ["decoder.mlmodelc"],
        "max_sequence_length": 1024,
        "token_batch_size": 16,
        "max_audio_seconds": 30,
        "residual_scale": 1,
        "head_dim": 128,
        "rope_theta": 1_000_000,
        "frontend": {"chunk_frames": 100},
        "encoder": {"window_tokens": 104},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def source(root: Path, model_id: str, revision: str) -> None:
    root.mkdir()
    for name in (
        "config.json",
        "chat_template.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (root / name).write_text("{}")
    (root / "merges.txt").write_text("merge")
    (root / "model.safetensors").write_bytes(b"checkpoint")
    (root / "source.json").write_text(
        json.dumps({"model_id": model_id, "revision": revision})
    )


def draft(root: Path, target_root: Path, manifest: dict, *, head_weight: bytes = b"shared") -> None:
    root.mkdir()
    compiled(root / "verify_head.mlmodelc", head_weight)
    source(root / "Qwen3-ASR-0.6B", DRAFT_MODEL_ID, DRAFT_REVISION)
    sha256 = hashlib.sha256(head_weight).hexdigest()
    record = {
        "schema_version": 1,
        "kind": "qwen3-asr-ane-draft",
        "draft": {
            "model_id": DRAFT_MODEL_ID,
            "revision": DRAFT_REVISION,
            "path": "Qwen3-ASR-0.6B",
        },
        "verify_head": {
            "path": "verify_head.mlmodelc",
            "token_batch_size": 16,
            "vocabulary_chunk": 8192,
            "weight_sha256": [sha256],
        },
        "target": {
            "model_id": TARGET_MODEL_ID,
            "source_revision": SOURCE_REVISION,
            "token_batch_size": 16,
            "tokenizer_sha256": digest(target_root / "tokenizer.json"),
            "weight_compression": {"scheme": None, "bits": None, "group_size": None},
            "manifest_sha256": digest(target_root / "manifest.json"),
        },
    }
    (root / "manifest.json").write_text(json.dumps(record))


def batch_head(
    root: Path,
    target_root: Path,
    manifest: dict,
    *,
    head_weight: bytes = b"shared",
    target_width: int | None = None,
) -> None:
    root.mkdir()
    compiled(root / "batch_head.mlmodelc", head_weight)
    record = {
        "schema_version": 1,
        "kind": "qwen3-asr-ane-batch-head",
        "head": {
            "path": "batch_head.mlmodelc",
            "token_batch_size": manifest["token_batch_size"],
            "vocabulary_chunk": 8192,
            "weight_sha256": [hashlib.sha256(head_weight).hexdigest()],
        },
        "target": {
            "model_id": manifest["model_id"],
            "source_revision": manifest["source_revision"],
            "token_batch_size": (
                manifest["token_batch_size"] if target_width is None else target_width
            ),
            "tokenizer_sha256": digest(target_root / manifest["files"]["tokenizer"]),
            "weight_compression": {"scheme": None, "bits": None, "group_size": None},
            "manifest_sha256": digest(target_root / "manifest.json"),
        },
    }
    (root / "manifest.json").write_text(json.dumps(record))


def test_constructor_is_pure(monkeypatch, tmp_path: Path):
    def forbidden(*args, **kwargs):
        raise AssertionError("ArtifactManager constructor touched the filesystem")

    with monkeypatch.context() as scoped:
        for name in ("stat", "exists", "is_file", "mkdir", "open", "resolve"):
            scoped.setattr(Path, name, forbidden)
        ArtifactManager(config(tmp_path, draft=True), "std-qwen3asr-ane/1.7b")


def test_target_status_rejects_known_manifest_incompatibility(tmp_path: Path):
    settings = config(tmp_path, draft=True)
    manifest = target(settings.model_dir)
    del manifest["decoder_partitions"]
    (settings.model_dir / "manifest.json").write_text(json.dumps(manifest))
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    requirements = manager.requirements(ArtifactContext(mode="batch"))[1]
    requirement = requirements[0]
    assert requirement.state == "corrupt"
    assert not requirement.can_acquire_now
    assert requirement.acquisition_blocker == "action_required"
    assert requirements[1].state == "missing"
    assert not requirements[1].can_acquire_now
    assert requirements[1].acquisition_blocker == "action_required"


def test_ready_draft_is_bound_and_weight_hashes_are_cached(monkeypatch, tmp_path: Path):
    settings = config(tmp_path, draft=True)
    manifest = target(settings.model_dir)
    draft(settings.draft_dir, settings.model_dir, manifest)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")

    import std_qwen3asr_ane.artifact_lifecycle as lifecycle

    calls = 0
    real_digest = lifecycle.digest

    def counted(path):
        nonlocal calls
        calls += 1
        return real_digest(path)

    monkeypatch.setattr(lifecycle, "digest", counted)
    first = manager.requirements(ArtifactContext(mode="batch"))[1]
    assert {item.artifact_id: item.state for item in first} == {
        BUNDLE_ARTIFACT_ID: "ready",
        DRAFT_ARTIFACT_ID: "ready",
    }
    assert calls == 2
    manager.requirements(ArtifactContext(mode="batch"))
    assert calls == 2

    (settings.draft_dir / "verify_head.mlmodelc/weights/weight.bin").write_bytes(b"changed")
    second = manager.requirements(ArtifactContext(mode="batch"))[1]
    assert second[1].state == "corrupt"
    assert calls == 3


def test_guided_batch_context_omits_the_unused_draft(tmp_path: Path):
    settings = config(tmp_path, draft=True)
    target(settings.model_dir)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    for params in (
        RuntimeParams(candidate_languages=["en", "zh"]),
        RuntimeParams(phrase_hints=["OpenAI"]),
    ):
        requirements = manager.requirements(ArtifactContext(mode="batch", params=params))[1]
        assert [item.artifact_id for item in requirements] == [BUNDLE_ARTIFACT_ID]


def test_request_can_explicitly_disable_the_configured_draft(tmp_path: Path):
    class Params(ProviderParams):
        disable_draft: bool = False

    settings = config(tmp_path, draft=True)
    target(settings.model_dir)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    context = ArtifactContext(
        mode="batch", params=RuntimeParams(provider_params=Params(disable_draft=True))
    )
    assert [item.artifact_id for item in manager.requirements(context)[1]] == [
        BUNDLE_ARTIFACT_ID
    ]


def test_optional_batch_head_reports_ready_and_bad_binding(monkeypatch, tmp_path: Path):
    settings = config(tmp_path, batching=True)
    manifest = target(settings.model_dir)
    batch_head(settings.batch_head_dir, settings.model_dir, manifest)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    import std_qwen3asr_ane.artifact_lifecycle as lifecycle

    calls = 0
    real_digest = lifecycle.digest

    def counted(path):
        nonlocal calls
        calls += 1
        return real_digest(path)

    monkeypatch.setattr(lifecycle, "digest", counted)
    requirements = manager.requirements(ArtifactContext(mode="batch"))[1]
    head = requirements[-1]
    assert head.artifact_id == BATCH_HEAD_ARTIFACT_ID
    assert head.state == "ready" and not head.required_for_inference
    assert calls == 2
    manager.requirements(ArtifactContext(mode="batch"))
    assert calls == 2

    # Rebuild the fixture with a target-width mismatch while preserving payloads.
    shutil.rmtree(settings.batch_head_dir)
    batch_head(settings.batch_head_dir, settings.model_dir, manifest, target_width=8)
    head = manager.requirements(ArtifactContext(mode="batch"))[1][-1]
    assert head.state == "corrupt"
    assert not head.can_acquire_now and head.acquisition_blocker == "action_required"


def test_missing_optional_batch_head_is_truthful_offline(monkeypatch, tmp_path: Path):
    settings = config(tmp_path, batching=True)
    target(settings.model_dir)
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    head = manager.requirements(ArtifactContext(mode="batch"))[1][-1]
    assert head.state == "missing" and not head.required_for_inference
    assert not head.can_acquire_now
    assert head.acquisition_blocker == "downloads_disabled"


def test_target_is_acquired_before_its_optional_batch_head(monkeypatch, tmp_path: Path):
    # Transfers are replaced below; this case checks acquisition ordering.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    settings = config(tmp_path, batching=True)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    context = ArtifactContext(mode="batch")
    requirements = manager.requirements(context)[1]
    order = []

    def acquire_target(emit):
        order.append("target")
        target(settings.model_dir)

    def acquire_head(emit):
        order.append("batch_head")

    monkeypatch.setattr(manager, "acquire_bundle", acquire_target)
    monkeypatch.setattr(manager, "acquire_batch_head", acquire_head)
    monkeypatch.setattr(
        "std_qwen3asr_ane.artifact_lifecycle.conversion_toolchain_available", lambda: True
    )
    manager.acquire(context, requirements, False, None)
    assert order == ["target", "batch_head"]


def test_batch_head_staging_is_atomic(monkeypatch, tmp_path: Path):
    settings = config(tmp_path, batching=True)
    manifest = target(settings.model_dir)
    source(settings.source_dir, TARGET_MODEL_ID, SOURCE_REVISION)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")

    import std_qwen3asr_ane.conversion.batch_head as builder

    def fail(target_root, source_root, output, *, verified_source):
        output.mkdir(parents=True)
        (output / "partial").write_text("partial")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(builder, "build_batch_head", fail)
    try:
        manager.acquire_batch_head(lambda *args, **kwargs: None)
    except RuntimeError as error:
        assert "failed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("failed conversion unexpectedly published a batch head")
    assert not settings.batch_head_dir.exists()

    def succeed(target_root, source_root, output, *, verified_source):
        batch_head(output, target_root, manifest)

    monkeypatch.setattr(builder, "build_batch_head", succeed)
    manager.acquire_batch_head(lambda *args, **kwargs: None)
    assert manager.requirements(ArtifactContext(mode="batch"))[1][-1].state == "ready"


def test_invalid_local_source_is_actionable_not_runnable(monkeypatch, tmp_path: Path):
    settings = config(tmp_path)
    settings.source_dir.mkdir()
    (settings.source_dir / "source.json").write_text("{}")
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    requirement = manager.requirements(ArtifactContext(mode="batch"))[1][0]
    assert requirement.state == "missing"
    assert not requirement.can_acquire_now
    assert requirement.acquisition_blocker == "action_required"
    assert requirement.required_actions[0].kind == "provide_artifacts"


def test_offline_source_hash_progress_is_exact(tmp_path: Path):
    settings = config(tmp_path)
    source(settings.source_dir, TARGET_MODEL_ID, SOURCE_REVISION)
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    events = []

    def emit(phase, artifact_id, **counts):
        events.append((phase, artifact_id, counts))

    verified = manager.ensure_source(emit)
    byte_events = [item for item in events if item[2].get("unit") == "bytes"]
    assert verified.content_sha256
    assert byte_events[0][2]["completed_units"] == 0
    assert byte_events[-1][2]["completed_units"] == byte_events[-1][2]["total_units"]


def test_offline_gate_requires_a_prepared_converter(monkeypatch, tmp_path: Path):
    settings = config(tmp_path)
    source(settings.source_dir, TARGET_MODEL_ID, SOURCE_REVISION)
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")

    import std_qwen3asr_ane.artifact_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "conversion_toolchain_available", lambda: False)
    monkeypatch.setattr(
        lifecycle,
        "inspect_conversion_worker",
        lambda config: SimpleNamespace(ready=False, reason="missing"),
    )
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    requirement = manager.requirements(ArtifactContext(mode="batch"))[1][0]
    assert not requirement.can_acquire_now
    assert requirement.acquisition_blocker == "downloads_disabled"


def test_alternate_target_revision_drives_source_validation(monkeypatch, tmp_path: Path):
    settings = config(tmp_path, draft=True)
    manifest = target(settings.model_dir)
    alternate = "b" * 40
    manifest["source_revision"] = alternate
    (settings.model_dir / "manifest.json").write_text(json.dumps(manifest))
    source(settings.source_dir, TARGET_MODEL_ID, alternate)
    source(settings.draft_source_dir, DRAFT_MODEL_ID, DRAFT_REVISION)
    monkeypatch.setattr(
        "std_qwen3asr_ane.artifact_lifecycle.conversion_toolchain_available", lambda: True
    )
    manager = ArtifactManager(settings, "std-qwen3asr-ane/1.7b")
    verified = manager.ensure_source(lambda *args, **kwargs: None, revision=alternate)
    assert verified.revision == alternate
    draft_requirement = manager.requirements(ArtifactContext(mode="batch"))[1][1]
    assert draft_requirement.state == "missing"
    assert draft_requirement.can_acquire_now
