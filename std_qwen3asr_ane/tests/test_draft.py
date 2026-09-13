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

from std_qwen3asr_ane.draft import (
    DRAFT_BUNDLE_KIND,
    DraftDependencyError,
    MLXDraft,
    check_draft_target,
    load_draft_manifest,
)
from std_qwen3asr_ane.runtime import CoreMLRuntime, TargetCursor


def draft_manifest(target: CoreMLRuntime, **overrides) -> dict:
    compression = target.manifest.get("weight_compression") or {}
    manifest = {
        "schema_version": 1,
        "kind": DRAFT_BUNDLE_KIND,
        "draft": {
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "revision": "b" * 40,
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
            "weight_compression": {
                key: compression.get(key) for key in ("scheme", "bits", "group_size")
            },
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
