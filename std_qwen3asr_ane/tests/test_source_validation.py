"""Checkpoint sources expose cheap closure and explicit content provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from std_qwen3asr_ane.source_validation import (
    SourceValidationError,
    inspect_source_checkpoint,
    verify_source_checkpoint,
    write_source_provenance,
)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
REVISION = "a" * 40


def source(root: Path, *, sharded: bool = True) -> Path:
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
    if sharded:
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for name in shards:
            (root / name).write_bytes(name.encode())
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
        )
    else:
        (root / "model.safetensors").write_bytes(b"weights")
    (root / "source.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": REVISION})
    )
    return root


def test_cheap_inspection_checks_identity_and_shard_closure(tmp_path: Path):
    root = source(tmp_path / "source")
    inspection = inspect_source_checkpoint(
        root, expected_model_id=MODEL_ID, expected_revision=REVISION
    )
    assert inspection.usable and not inspection.content_recorded

    (root / "model-00002-of-00002.safetensors").unlink()
    assert inspect_source_checkpoint(root).state == "corrupt"


def test_content_provenance_round_trip_and_tamper_detection(tmp_path: Path):
    root = source(tmp_path / "source", sharded=False)
    progress = []
    written = write_source_provenance(
        root,
        model_id=MODEL_ID,
        revision=REVISION,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    assert progress[0][0] == 0
    assert progress[-1][0] == progress[-1][1]
    assert written.content_sha256
    inspection = inspect_source_checkpoint(root)
    assert inspection.usable and inspection.content_recorded
    assert inspection.content_sha256 == written.content_sha256
    (root / "operator-note.md").write_text("not consumed by conversion")
    assert verify_source_checkpoint(root).content_sha256 == written.content_sha256

    weights = root / "model.safetensors"
    weights.write_bytes(b"changed-size")
    assert inspect_source_checkpoint(root).state == "corrupt"
    with pytest.raises(SourceValidationError):
        verify_source_checkpoint(root)


def test_same_size_tamper_is_found_by_explicit_verification(tmp_path: Path):
    root = source(tmp_path / "source", sharded=False)
    write_source_provenance(root, model_id=MODEL_ID, revision=REVISION)
    weights = root / "model.safetensors"
    weights.write_bytes(b"changed")  # same length as b"weights"
    assert inspect_source_checkpoint(root).state == "ready"
    with pytest.raises(SourceValidationError, match="digest"):
        verify_source_checkpoint(root)
