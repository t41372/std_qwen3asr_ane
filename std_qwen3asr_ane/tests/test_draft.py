"""Draft bundle binding, the target cursor and the runtime's speculative path.

MLX is never imported here; a fake draft implements the TokenDecoder protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_runtime import bundle, fake_coreml  # noqa: F401 — shared toy runtime fixtures

from std_qwen3asr_ane.bundle import SUPPORTED_SCHEMA_VERSIONS, lm_head_compression
from std_qwen3asr_ane.draft import (
    DRAFT_BUNDLE_KIND,
    DRAFT_REVISION,
    DraftDependencyError,
    MLXDraft,
    check_draft_target,
    load_draft_manifest,
)
from std_qwen3asr_ane.runtime import CoreMLRuntime, TargetCursor, VerifyHead


def draft_manifest(target: CoreMLRuntime, **overrides) -> dict:
    manifest = {
        "schema_version": 1,
        "kind": DRAFT_BUNDLE_KIND,
        "draft": {
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "revision": DRAFT_REVISION,
            "path": "Qwen3-ASR-0.6B",
        },
        "verify_head": {
            "path": "verify_head.mlmodelc",
            "token_batch_size": target.token_batch_size,
        },
        "target": {
            "model_id": target.manifest["model_id"],
            "source_revision": target.manifest["source_revision"],
            "token_batch_size": target.token_batch_size,
            "tokenizer_sha256": target.tokenizer_sha256,
            "weight_compression": lm_head_compression(target.manifest),
        },
    }
    manifest["target"].update(overrides)
    return manifest


class FakeDraft:
    """Proposes a fixed script of tokens; wrong proposals exercise rejection."""

    def __init__(self, script):
        self.script = list(script)
        self.prepared = None
        self.positions = []

    def prepare(self, samples, token_ids):
        self.prepared = (len(samples), list(token_ids))
        return {"draft_encoder_seconds": 0.0, "draft_prefill_seconds": 0.0}

    def step(self, tokens, position):
        self.positions.append((position, tuple(tokens)))
        index = position - len(self.prepared[1]) + len(tokens) - 1
        return [self.script[index + 1] if index + 1 < len(self.script) else 0]

    def choose(self, hidden):
        return int(hidden)


def test_manifest_and_target_binding(bundle: Path, fake_coreml) -> None:  # noqa: F811
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["source_revision"] = "a" * 40
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    target = CoreMLRuntime(bundle)
    good = draft_manifest(target)
    check_draft_target(good, target)
    for key, value in (
        ("source_revision", "c" * 40),
        ("token_batch_size", 64),
        ("tokenizer_sha256", "0" * 64),
        ("weight_compression", {"scheme": "palette", "bits": 4, "group_size": 16}),
    ):
        with pytest.raises(ValueError, match=key):
            check_draft_target(draft_manifest(target, **{key: value}), target)
    (bundle / "draft-manifest.json").write_text(json.dumps({"schema_version": 1, "kind": "x"}))
    with pytest.raises(FileNotFoundError):
        load_draft_manifest(bundle / "missing")
    (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1, "kind": "x"}))
    with pytest.raises(ValueError, match="draft bundle"):
        load_draft_manifest(bundle)


def test_target_binding_uses_head_compression(bundle: Path, fake_coreml) -> None:  # noqa: F811
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["source_revision"] = "a" * 40
    shared = {"scheme": "palette", "bits": 8, "group_size": 32}
    manifest["weight_compression"] = {**shared, "roles": ["decoder", "encoder"]}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    target = CoreMLRuntime(bundle)
    assert lm_head_compression(target.manifest) == {key: None for key in shared}
    check_draft_target(draft_manifest(target), target)
    with pytest.raises(ValueError, match="weight_compression"):
        check_draft_target(draft_manifest(target, weight_compression=shared), target)
    manifest["weight_compression"]["roles"].append("lm_head")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    target = CoreMLRuntime(bundle)
    assert lm_head_compression(target.manifest) == shared
    check_draft_target(draft_manifest(target, weight_compression=shared), target)


def test_draft_builder_admits_every_supported_bundle_schema(tmp_path: Path) -> None:
    from std_qwen3asr_ane.conversion.draft import build_draft_bundle

    target = tmp_path / "target"
    target.mkdir()
    identity = {"model_id": "Qwen/Qwen3-ASR-1.7B", "source_revision": "a" * 40}
    for version in SUPPORTED_SCHEMA_VERSIONS:
        (target / "manifest.json").write_text(json.dumps({"schema_version": version, **identity}))
        # Past the identity gate, the builder reads the source checkpoint next.
        with pytest.raises(FileNotFoundError):
            build_draft_bundle(target, tmp_path / "missing-source", tmp_path / f"out-{version}")
    (target / "manifest.json").write_text(json.dumps({"schema_version": 4, **identity}))
    with pytest.raises(ValueError, match="1.7B bundle"):
        build_draft_bundle(target, tmp_path / "missing-source", tmp_path / "out-4")


def test_mlx_absence_is_a_dependency_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import builtins

    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith(("mlx", "mlx_audio")):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DraftDependencyError, match="gpu-draft"):
        MLXDraft(tmp_path)


def test_target_cursor_rows_and_choice(bundle: Path, fake_coreml) -> None:  # noqa: F811
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest.update(source_revision="a" * 40, token_batch_size=4)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    target = CoreMLRuntime(bundle)
    states = [model.make_state() for model in target.decoders]
    cursor = TargetCursor(target, states)
    rows = cursor.step([1, 2, 3], 5)
    assert len(rows) == 3 and all(row.shape == (1, 8, 1, 1) for row in rows)
    assert cursor.choose(5) == 5
    head = SimpleNamespace(choose_rows=lambda hidden, count: list(range(count)))
    assert TargetCursor(target, states, head).step([1, 2], 0) == [0, 1]
    with pytest.raises(ValueError):
        cursor.step([1, 2, 3, 4, 5], 0)


def test_speculative_path_matches_serial(bundle: Path, fake_coreml) -> None:  # noqa: F811
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest.update(source_revision="a" * 40, token_batch_size=4)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    target = CoreMLRuntime(bundle)
    eos = target.tokenizer.token_to_id("<|im_end|>")
    audio = np.zeros(16000, dtype=np.float32)
    script = [1, 1, 1, eos]
    for model in fake_coreml:
        if model.role == "lm_head":
            model.tokens = list(script)
    serial = target.transcribe(audio, language=None, max_new_tokens=8)
    for model in fake_coreml:
        if model.role == "lm_head":
            model.tokens = list(script)
    draft = SimpleNamespace(model=FakeDraft([1, 1, 5, eos]), head=None)
    result = target.transcribe_speculative(
        audio, draft, language=None, max_new_tokens=8, lookahead=3
    )
    assert result.token_ids == serial.token_ids == (1, 1, 1)
    assert result.text == serial.text
    assert draft.model.prepared[0] == audio.size and draft.model.positions
    assert result.timings["verifier_calls"] >= 1 and "draft_prefill_seconds" in result.timings
    with pytest.raises(ValueError, match="lookahead"):
        target.transcribe_speculative(audio, draft, language=None, max_new_tokens=8, lookahead=4)


def test_verify_head_maps_chunk_winners_to_global_ids(bundle: Path, fake_coreml) -> None:  # noqa: F811
    import sys

    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest.update(source_revision="a" * 40, token_batch_size=4)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    target = CoreMLRuntime(bundle)
    module = sys.modules["coremltools"]
    real_model = module.models.MLModel

    class Head:
        def __init__(self, path, *, compute_units):
            self.width = 4

        def predict(self, data):
            width = data["hidden_states"].shape[-1]
            # Two chunks (the toy lm_head splits its 9-token vocabulary 4 + 5):
            # column i wins in chunk i % 2 with local index i.
            values = np.full((1, 2, width), -1.0, np.float32)
            indices = np.zeros((1, 2, width), np.int32)
            for column in range(width):
                values[0, column % 2, column] = 1.0
                indices[0, column % 2, column] = column
            return {"max_values": values, "max_indices": indices}

        def close(self, *, timeout=5):
            pass

    module.models.MLModel = Head
    try:
        head = VerifyHead(bundle / "verify_head.mlpackage", target, compute_units="cpu_and_ne")
        assert head.chunk_size == 4 and head.width == 4
        hidden = np.zeros((1, 8, 1, 3), np.float16)
        assert head.choose_rows(hidden, 3) == [0, 4 + 1, 2]
        with pytest.raises(ValueError, match="width"):
            head.choose_rows(np.zeros((1, 8, 1, 5), np.float16), 5)
        with pytest.raises(ValueError, match="chunk"):
            VerifyHead(
                bundle / "verify_head.mlpackage",
                target,
                compute_units="cpu_and_ne",
                vocabulary_chunk=8192,
            )
    finally:
        module.models.MLModel = real_model
