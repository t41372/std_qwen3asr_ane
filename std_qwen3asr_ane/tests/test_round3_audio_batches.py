"""Batch grouping preserves order, per-window masks, tails and streaming delegation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from round3_runtime import ExperimentalRuntime

from std_qwen3asr_ane.runtime import CoreMLRuntime


class Frontend:
    def predict(self, data):
        # Expose input order and both convolution masks in every output value.
        values = data["mel_features"][:, :, :1, ::8]
        mask_sum = data["conv1_mask"].sum(axis=-1, keepdims=True)
        mask_sum += data["conv2_mask"].sum(axis=-1, keepdims=True)
        return {"chunk_embeddings": values + mask_sum}


class Encoder:
    def predict(self, data):
        invalid = (data["key_mask"] < 0).sum(axis=1, keepdims=True)
        return {"audio_embeddings": data["hidden_states"] + invalid}


def runtime(kind, front=1, encoder=1):
    result = object.__new__(kind)
    result.chunk_frames, result.window_tokens = 100, 104
    result.embeddings = np.zeros((10, 1), np.float16)
    result.frontend, result.encoder = Frontend(), Encoder()
    if kind is ExperimentalRuntime:
        result.audio_batches = {"frontend": front, "encoder": encoder}
        result.batched_models = {"frontend": Frontend(), "encoder": Encoder()}
    return result


@pytest.mark.parametrize(
    "frames", [1, 49, 99, 100, 101, 399, 400, 401, 799, 800, 801, 1599, 1600, 1601, 3000]
)
@pytest.mark.parametrize("front,encoder", [(4, 1), (8, 1), (1, 2), (4, 2)])
def test_batched_grouping_matches_serial_audio(frames, front, encoder):
    features = np.broadcast_to(np.arange(frames, dtype=np.float32)[None], (128, frames))
    baseline, candidate = runtime(CoreMLRuntime), runtime(ExperimentalRuntime, front, encoder)
    expected = baseline._encode_audio(features)
    actual = candidate._encode_audio(features)
    np.testing.assert_array_equal(actual, expected)
    chunks = (frames + 99) // 100
    windows = (actual.shape[-1] + 103) // 104
    assert candidate.audio_call_counts == {
        "frontend_calls": chunks // front + chunks % front,
        "encoder_calls": windows // encoder + windows % encoder,
    }


def test_streaming_keeps_context_owned_b1_path():
    candidate = runtime(ExperimentalRuntime, 4, 2)
    features = np.ones((128, 500), np.float32)
    sentinel = np.ones((1, 1, 1, 65), np.float32)

    class Context:
        def encode(self, actual, *, owner, frontend, encoder):
            assert actual is features and owner is candidate
            assert frontend is candidate.frontend and encoder is candidate.encoder
            return sentinel

    assert candidate._encode_audio(features, audio_context=Context()) is sentinel
    assert candidate.audio_call_counts == {}
