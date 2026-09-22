"""Fake-only reproductions for the artifact lifecycle audit.

Run with::

    .venv/bin/python research/standard-asr-audit-2026-09-22/probe_artifacts.py

The probe writes only minimal temporary JSON and placeholder payloads. It does
not load Core ML or MLX, contact a model service, download weights, convert a
bundle, or claim that its placeholder files are executable models.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from standard_asr.contract.exceptions import ArtifactAcquisitionError, ConfigError

from std_qwen3asr_ane import plugin
from std_qwen3asr_ane.plugin import MODEL_ID, create_engine

RUNTIME_REQUIRED_FIELDS = {
    "decoder_partitions",
    "head_dim",
    "max_audio_seconds",
    "max_sequence_length",
    "residual_scale",
    "rope_theta",
}


def _write_package(path: Path) -> None:
    """Write the smallest Core ML package layout accepted by status inspection."""
    data = path / "Data/com.apple.CoreML"
    (data / "weights").mkdir(parents=True)
    (data / "model.mlmodel").write_bytes(b"fake model specification")
    (data / "weights/weight.bin").write_bytes(b"fake weight payload")
    (path / "Manifest.json").write_text(
        json.dumps(
            {
                "fileFormatVersion": "1.0.0",
                "itemInfoEntries": {
                    "model": {"path": "com.apple.CoreML/model.mlmodel"},
                    "weights": {"path": "com.apple.CoreML/weights"},
                },
                "rootModelIdentifier": "model",
            }
        )
    )


def _write_target(path: Path) -> dict[str, object]:
    """Write a layout-complete target whose manifest is knowingly non-loadable."""
    path.mkdir()
    files = {
        "frontend": "frontend.mlpackage",
        "encoder": "encoder.mlpackage",
        "decoder": "decoder.mlpackage",
        "lm_head": "lm_head.mlpackage",
        "embedding": "embedding.npy",
        "tokenizer": "tokenizer.json",
        "mel_filters": "mel_filters.npy",
    }
    for relative in files.values():
        payload = path / relative
        if payload.suffix == ".mlpackage":
            _write_package(payload)
        else:
            payload.write_bytes(b"{}")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "source_revision": "a" * 40,
        "files": files,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _write_draft(path: Path) -> None:
    """Write a layout-complete draft with a knowingly invalid target binding."""
    head = path / "verify_head.mlmodelc"
    (head / "weights").mkdir(parents=True)
    for relative in ("coremldata.bin", "model.mil", "weights/weight.bin"):
        (head / relative).write_bytes(b"fake compiled payload")
    checkpoint = path / "Qwen3-ASR-0.6B"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"fake checkpoint payload")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "qwen3-asr-ane-draft",
                "draft": {
                    "model_id": "not-the-draft-model",
                    "revision": "not-the-pinned-revision",
                    "path": checkpoint.name,
                },
                "verify_head": {
                    "path": head.name,
                    "token_batch_size": 16,
                },
                "target": {},
            }
        )
    )


def probe_false_ready_target_and_profile(root: Path) -> dict[str, object]:
    target = root / "target"
    manifest = _write_target(target)
    missing = sorted(RUNTIME_REQUIRED_FIELDS - manifest.keys())

    general = create_engine(download_root=root, model_dir=target)
    general_readiness = general.artifact_status().readiness

    short = create_engine(download_root=root, model_dir=target, profile="short-dictation")
    short_readiness = short.artifact_status().readiness
    try:
        short.prepare()
    except ConfigError as error:
        short_prepare = type(error).__name__
    else:  # pragma: no cover - the current contradiction is the audit evidence
        raise AssertionError("the incompatible short-profile fixture unexpectedly prepared")

    assert missing == sorted(RUNTIME_REQUIRED_FIELDS)
    assert general_readiness == "ready"
    assert short_readiness == "ready"
    assert short_prepare == "ConfigError"
    return {
        "general_status": general_readiness,
        "missing_runtime_fields": missing,
        "short_prepare": short_prepare,
        "short_status": short_readiness,
    }


def probe_false_ready_draft_binding(root: Path) -> dict[str, object]:
    target = root / "draft-target"
    _write_target(target)
    draft = root / "draft"
    draft.mkdir()
    _write_draft(draft)

    report = create_engine(download_root=root, model_dir=target, draft_dir=draft).artifact_status()
    states = {requirement.artifact_id: requirement.state for requirement in report.requirements}
    draft_manifest = json.loads((draft / "manifest.json").read_text())

    assert report.readiness == "ready"
    assert all(state == "ready" for state in states.values())
    assert draft_manifest["target"] == {}
    return {
        "aggregate_status": report.readiness,
        "draft_model_id": draft_manifest["draft"]["model_id"],
        "draft_revision": draft_manifest["draft"]["revision"],
        "requirement_states": states,
        "target_binding": draft_manifest["target"],
    }


def probe_misleading_source_gate(root: Path) -> dict[str, object]:
    source = root / "invalid-source"
    source.mkdir()
    (source / "source.json").write_text("{}")
    target = root / "missing-target"

    previous_policy = os.environ.get("STANDARD_ASR_ALLOW_DOWNLOAD")
    original_toolchain_check = plugin._conversion_toolchain_available
    os.environ["STANDARD_ASR_ALLOW_DOWNLOAD"] = "0"
    plugin._conversion_toolchain_available = lambda: True
    try:
        engine = create_engine(download_root=root, model_dir=target, source_dir=source)
        requirement = engine.artifact_status().requirements[0]
        try:
            engine.acquire_artifacts()
        except ArtifactAcquisitionError as error:
            acquire_reason = error.reason
        else:  # pragma: no cover - malformed provenance cannot build this target
            raise AssertionError("acquisition unexpectedly accepted malformed provenance")
    finally:
        plugin._conversion_toolchain_available = original_toolchain_check
        if previous_policy is None:
            os.environ.pop("STANDARD_ASR_ALLOW_DOWNLOAD", None)
        else:
            os.environ["STANDARD_ASR_ALLOW_DOWNLOAD"] = previous_policy

    assert requirement.can_acquire_now is True
    assert requirement.acquisition_blocker is None
    assert acquire_reason == "action_required"
    assert not target.exists()
    return {
        "acquire_reason": acquire_reason,
        "can_acquire_now": requirement.can_acquire_now,
        "downloads_disabled": True,
        "preflight_blocker": requirement.acquisition_blocker,
        "source_metadata": {},
        "target_created": target.exists(),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="standard-asr-artifact-probe-") as directory:
        root = Path(directory)
        result = {
            "a1_target_and_profile": probe_false_ready_target_and_profile(root),
            "a2_draft_binding": probe_false_ready_draft_binding(root),
            "a3_source_gate": probe_misleading_source_gate(root),
            "boundary": "fake metadata and placeholder payloads only; no native model validation",
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
