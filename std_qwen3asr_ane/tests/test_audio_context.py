"""Exact cached encodings against the unchanged batch orchestration oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from std_qwen3asr_ane.audio import log_mel_spectrogram
from std_qwen3asr_ane.audio_context import AudioPrefixContext
from std_qwen3asr_ane.runtime import CoreMLRuntime


class Frontend:
    def __init__(self):
        self.calls = 0

    def predict(self, data):
        self.calls += 1
        frames = data["mel_features"][0, 0].sum(axis=0)
        values = np.pad(frames, (0, 4)).reshape(13, 8).sum(axis=1)
        values += data["conv1_mask"].sum() + data["conv2_mask"].sum()
        return {"chunk_embeddings": np.tile(values[None, None, None], (1, 4, 1, 1))}


class Encoder:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def predict(self, data):
        self.calls += 1
        if self.fail:
            raise RuntimeError("window failed")
        valid = data["key_mask"][0, :, 0, 0] == 0
        hidden = data["hidden_states"]
        mixed = hidden + hidden[..., valid].mean(axis=-1, keepdims=True)
        return {"audio_embeddings": np.tile(mixed, (1, 2, 1, 1))}


def batch(features):
    runtime = CoreMLRuntime.__new__(CoreMLRuntime)
    runtime.frontend, runtime.encoder = Frontend(), Encoder()
    runtime.chunk_frames, runtime.window_tokens = 100, 104
    runtime.embeddings = np.zeros((1, 8), np.float32)
    return runtime._encode_audio(features)


@pytest.fixture
def cached():
    owner, frontend, encoder = object(), Frontend(), Encoder()
    context = AudioPrefixContext(owner=owner, hidden_width=8)
    return SimpleNamespace(
        context=context,
        frontend=frontend,
        encoder=encoder,
        encode=lambda features: context.encode(
            features, owner=owner, frontend=frontend, encoder=encoder
        ),
    )


def test_growth_shrink_and_chunk_window_boundaries_match_batch(cached):
    features = np.random.default_rng(9).normal(size=(128, 1700)).astype(np.float32)
    for count in (50, 99, 100, 101, 199, 200, 799, 800, 801, 899, 1600, 1651, 800, 50):
        actual = cached.encode(features[:, :count])
        np.testing.assert_array_equal(actual, batch(features[:, :count]))


def test_completed_windows_are_reused_but_growing_tail_is_recomputed(cached):
    features = np.ones((128, 1000), np.float32)
    cached.encode(features[:, :800])
    cached.encode(features[:, :900])
    assert cached.context.timings == {
        "frontend_calls": 1,
        "encoder_calls": 1,
        "reused_frontend_chunks": 8,
        "reused_encoder_windows": 1,
    }
    cached.encode(features)
    assert cached.context.timings["frontend_calls"] == 1
    assert cached.context.timings["encoder_calls"] == 1
    actual = cached.encode(features)
    assert cached.context.timings["frontend_calls"] == 0
    assert cached.context.timings["encoder_calls"] == 0
    actual.fill(123)  # Returned arrays cannot corrupt cached model outputs.
    np.testing.assert_array_equal(cached.encode(features), batch(features))


def test_revising_old_features_invalidates_their_window_only(cached):
    features = np.zeros((128, 1600), np.float32)
    cached.encode(features)
    features[0, 1] = 1.0
    np.testing.assert_array_equal(cached.encode(features), batch(features))
    assert cached.context.timings["frontend_calls"] == 1
    assert cached.context.timings["encoder_calls"] == 1
    assert cached.context.timings["reused_encoder_windows"] == 1


def test_short_clip_mask_change_prevents_reuse_even_with_identical_padding(cached):
    cached.encode(np.zeros((128, 50), np.float32))
    features = np.zeros((128, 100), np.float32)
    np.testing.assert_array_equal(cached.encode(features), batch(features))
    assert cached.context.timings["frontend_calls"] == 1


def test_louder_audio_changes_old_mel_floor_and_invalidates_completed_chunks(cached):
    filters = np.ones((201, 128), np.float32)
    silence = np.zeros(16000, np.float32)
    quiet = log_mel_spectrogram(silence, filters)
    cached.encode(quiet)
    loud = np.random.default_rng(19).normal(size=16000).astype(np.float32)
    features = log_mel_spectrogram(np.concatenate((silence, loud)), filters)
    assert not np.array_equal(features[:, :50], quiet[:, :50])
    np.testing.assert_array_equal(cached.encode(features), batch(features))
    assert cached.context.timings["reused_frontend_chunks"] == 0


def test_failure_discards_reuse_and_retry_matches_fresh_state(cached):
    features = np.zeros((128, 900), np.float32)
    cached.encode(features[:, :800])
    cached.encoder.fail = True
    with pytest.raises(RuntimeError, match="window failed"):
        cached.encode(features)
    assert cached.context.timings == {}
    cached.encoder.fail = False
    np.testing.assert_array_equal(cached.encode(features), batch(features))
    assert cached.context.timings["frontend_calls"] == 9
    assert cached.context.timings["encoder_calls"] == 2


def test_context_rejects_a_different_runtime(cached):
    with pytest.raises(ValueError, match="different runtime"):
        cached.context.encode(
            np.zeros((128, 100), np.float32),
            owner=object(),
            frontend=cached.frontend,
            encoder=cached.encoder,
        )
