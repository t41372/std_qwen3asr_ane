"""Independent-sequence packing tests over the real runtime decoder interface."""

# ruff: noqa: F811

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from test_runtime import bundle, fake_coreml  # noqa: F401 — shared fixtures

from std_qwen3asr_ane.artifact_validation import validate_target_manifest
from std_qwen3asr_ane.batching import (
    BATCH_HEAD_BUNDLE_KIND,
    OfflineRecognitionRequest,
    PackedDecoderRow,
    load_target_bound_batch_head,
    validate_target_bound_batch_head,
)
from std_qwen3asr_ane.bundle import digest, lm_head_compression
from std_qwen3asr_ane.runtime import CoreMLRuntime


class ScriptedBatchHead:
    """A compact-head double whose rows expose scheduler grouping in its script."""

    def __init__(self, width: int, eos: int) -> None:
        self.width = width
        self.eos = eos
        self.calls: list[int] = []

    def choose_rows(self, hidden, count, *, cancel=None):
        del hidden, cancel
        self.calls.append(count)
        if len(self.calls) == 1:
            assert count == 2
            return [1, self.eos]
        assert count == 1
        return [self.eos]


def _set_width(bundle: Path, width: int = 4) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["token_batch_size"] = width
    manifest_path.write_text(json.dumps(manifest))


def test_packed_rows_have_disjoint_update_slots_and_block_diagonal_masks(bundle, fake_coreml) -> None:
    _set_width(bundle)
    runtime = CoreMLRuntime(bundle)
    states = runtime._new_packed_states()
    width = runtime.embeddings.shape[1]
    runtime._decode_packed_rows(
        (
            PackedDecoderRow(0, np.ones((1, width, 1, 1), np.float16), 0, 1),
            PackedDecoderRow(1, np.full((1, width, 1, 1), 2, np.float16), 4, 0),
        ),
        states,
    )

    decoder = next(model for model in fake_coreml if model.role == "decoder_00")
    inputs, state = decoder.calls[-1]
    update = inputs["update_mask"][0, 0]
    mask = inputs["attention_mask"][0, 0]
    assert state is states[0]
    assert np.flatnonzero(update[0]).tolist() == [1]
    assert np.flatnonzero(update[1]).tolist() == [4]
    assert np.flatnonzero(mask[0] == 0).tolist() == [0, 1]
    assert np.flatnonzero(mask[1] == 0).tolist() == [4]
    np.testing.assert_array_equal(inputs["hidden_states"][..., 0], 1)
    np.testing.assert_array_equal(inputs["hidden_states"][..., 1], 2)
    # The second row writes global cache slot 4, but its first token must use
    # the same local RoPE input as a standalone serial request at position 0.
    np.testing.assert_allclose(
        inputs["cosine"][0, 0, 0, :2], [runtime.cosine[1, 0], runtime.cosine[0, 0]], atol=5e-4
    )


def test_transcribe_many_packs_independent_prompts_and_isolates_each_result(bundle, fake_coreml) -> None:
    _set_width(bundle)
    runtime = CoreMLRuntime(bundle)
    eos = runtime.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    head = ScriptedBatchHead(runtime.token_batch_size, eos)
    outcomes = runtime.transcribe_many(
        (
            OfflineRecognitionRequest(np.zeros(8000, np.float32), "en", 4, context="first"),
            OfflineRecognitionRequest(np.zeros(8000, np.float32), "en", 4, context="first"),
        ),
        batch_head=head,
    )

    assert [outcome.execution for outcome in outcomes] == ["packed", "packed"]
    assert [outcome.error for outcome in outcomes] == [None, None]
    assert [outcome.result.text for outcome in outcomes] == ["hello", ""]
    assert [outcome.result.language for outcome in outcomes] == ["en", "en"]
    assert head.calls == [2, 1]
    stats = outcomes[0].stats
    assert stats is not None and stats.lane_count == 2 and stats.decoder_calls > 1
    assert outcomes[1].stats == stats

    decoder = next(model for model in fake_coreml if model.role == "decoder_00")
    # At least one invocation carries two independent requests in the same
    # native graph call; this is not an outer executor around serial calls.
    assert any(call[0]["update_mask"][0, 0].sum() > 1 for call in decoder.calls)


def test_missing_batch_head_uses_the_unchanged_serial_path(bundle, fake_coreml) -> None:
    _set_width(bundle)
    runtime = CoreMLRuntime(bundle)
    lm_head = next(model for model in fake_coreml if model.role == "lm_head")
    eos = runtime.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    lm_head.tokens[:] = [1, eos]
    direct = runtime.transcribe(np.zeros(8000, np.float32), language=None, max_new_tokens=4)
    lm_head.tokens[:] = [1, eos]
    outcome = runtime.transcribe_many(
        (OfflineRecognitionRequest(np.zeros(8000, np.float32), None, 4),)
    )[0]

    assert outcome.execution == "serial"
    assert outcome.fallback_reason == "no target-bound compact batch head was supplied"
    assert outcome.error is None
    assert outcome.result.text == direct.text
    assert outcome.result.token_ids == direct.token_ids


def test_one_invalid_input_does_not_abort_its_packable_peers(bundle, fake_coreml) -> None:
    _set_width(bundle)
    runtime = CoreMLRuntime(bundle)
    eos = runtime.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    head = ScriptedBatchHead(runtime.token_batch_size, eos)
    outcomes = runtime.transcribe_many(
        (
            OfflineRecognitionRequest(np.zeros(8000, np.float32), "en", 4),
            OfflineRecognitionRequest(np.array([], np.float32), "en", 4),
            OfflineRecognitionRequest(np.zeros(8000, np.float32), "en", 4),
        ),
        batch_head=head,
    )

    assert outcomes[0].execution == outcomes[2].execution == "packed"
    assert outcomes[0].result.text == "hello"
    assert outcomes[2].result.text == ""
    assert outcomes[1].execution == "serial"
    assert outcomes[1].result is None
    assert isinstance(outcomes[1].error, ValueError)
    assert "nonempty" in str(outcomes[1].error)


def test_target_bound_batch_head_rejects_a_same_shaped_unbound_payload(bundle, fake_coreml, tmp_path) -> None:
    _set_width(bundle)
    target_info = validate_target_manifest(json.loads((bundle / "manifest.json").read_text()))
    target_weight = bundle / "lm_head.mlpackage/weights/weight.bin"
    target_weight.parent.mkdir(parents=True)
    target_weight.write_bytes(b"target-head")
    root = tmp_path / "batch-head"
    head_weight = root / "batch_head.mlmodelc/weights/weight.bin"
    head_weight.parent.mkdir(parents=True)
    head_weight.write_bytes(b"target-head")
    record = {
        "schema_version": 1,
        "kind": BATCH_HEAD_BUNDLE_KIND,
        "head": {
            "path": "batch_head.mlmodelc",
            "token_batch_size": 4,
            "vocabulary_chunk": 4,
            "weight_sha256": [digest(head_weight)],
        },
        "target": {
            "model_id": target_info.model_id,
            "source_revision": target_info.source_revision,
            "token_batch_size": target_info.token_batch_size,
            "tokenizer_sha256": digest(bundle / target_info.files["tokenizer"]),
            "manifest_sha256": digest(bundle / "manifest.json"),
            "weight_compression": lm_head_compression(dict(target_info.manifest)),
        },
    }
    (root / "manifest.json").write_text(json.dumps(record))
    artifact = validate_target_bound_batch_head(
        root, target=target_info, target_root=bundle
    )
    assert not fake_coreml  # Artifact validation must not construct a Core ML model.
    assert artifact.path == (root / "batch_head.mlmodelc").resolve()
    runtime = CoreMLRuntime(bundle)
    assert load_target_bound_batch_head(root, runtime) == artifact
    head_weight.write_bytes(b"wrong-head")
    try:
        load_target_bound_batch_head(root, runtime)
    except ValueError as error:
        assert "weights" in str(error)
    else:  # pragma: no cover - regression guard for a binding bypass
        raise AssertionError("An altered batch head must be rejected")
