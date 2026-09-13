"""Decoder rewind correctness, including stale generated suffixes and failures."""

from copy import deepcopy

import numpy as np
import pytest

from std_qwen3asr_ane.streaming_context import DecoderPrefixContext


class CausalDecoder:
    """A small two-partition causal recurrence with inspectable persistent state."""

    def __init__(self, *, batch_size=16, cache_length=128):
        self.batch_size = batch_size
        self.cache_length = cache_length
        self.calls = []
        self.fail_at = None

    def context(self):
        def make_states():
            return [np.zeros((1, 4, 1, self.cache_length)) for _ in range(2)]

        return DecoderPrefixContext(
            owner=self,
            states=make_states(),
            token_batch_size=self.batch_size,
            cache_length=self.cache_length,
            make_states=make_states,
        )

    def step(self, hidden, position, states):
        self.calls.append((position, hidden.shape[-1]))
        hidden = hidden.copy()
        for state in states:
            for row in range(hidden.shape[-1]):
                index = position + row
                # Overwrite before attending; future cache slots remain stale
                # but are never visible to the current causal query.
                state[..., index] = hidden[..., row]
                hidden[..., row] += state[..., : index + 1].sum(axis=-1) / (index + 1)
            if self.fail_at == position:
                raise RuntimeError("simulated partial partition failure")
        return hidden[..., -1:]

    def prefill(self, context, prompt):
        return context.prefill(prompt, decode_step=self.step, owner=self)


def embeddings(length, *, seed=10):
    return np.random.default_rng(seed).normal(size=(1, 4, 1, length)).astype(np.float32)


@pytest.mark.parametrize(
    "changed_token, expected_reused",
    [(0, 0), (15, 0), (16, 16), (31, 16), (32, 32), (52, 48), (None, 48)],
)
def test_exact_changed_prefix_replays_from_block_boundary(changed_token, expected_reused):
    decoder = CausalDecoder()
    context = decoder.context()
    original = embeddings(53)
    first = decoder.prefill(context, original)
    assert first.reused_tokens == 0 and first.decoded_tokens == 53
    # Simulate generation after the old prompt. A later prefill must never
    # mistake these populated positions for evidence about its new prompt.
    decoder.step(embeddings(9, seed=2), 53, context.states)
    prompt = original.copy()
    if changed_token is not None:
        # This is below float16 resolution: compare the original embeddings,
        # not the rounded decoder inputs and not an allclose tolerance.
        prompt[..., changed_token] = np.nextafter(prompt[..., changed_token], np.float32(np.inf))
    decoder.calls.clear()
    actual = decoder.prefill(context, prompt)
    calls = decoder.calls.copy()
    expected = decoder.prefill(decoder.context(), prompt)
    assert actual.reused_tokens == expected_reused
    assert actual.decoded_tokens == 53 - expected_reused
    assert calls[0][0] == expected_reused
    np.testing.assert_array_equal(actual.hidden, expected.hidden)
    for state, fresh in zip(actual.states, expected.states, strict=True):
        np.testing.assert_array_equal(state[..., :53], fresh[..., :53])
    assert actual.states is first.states


@pytest.mark.parametrize("old_length, new_length", [(32, 32), (33, 58), (53, 32), (53, 17), (7, 3)])
def test_growing_and_shortening_prompts_match_fresh_decoder(old_length, new_length):
    decoder = CausalDecoder()
    context = decoder.context()
    full = embeddings(max(old_length, new_length))
    decoder.prefill(context, full[..., :old_length])
    decoder.step(embeddings(11, seed=7), old_length, context.states)
    actual = decoder.prefill(context, full[..., :new_length])
    expected = decoder.prefill(decoder.context(), full[..., :new_length])
    reusable = min(old_length, new_length - 1) // 16 * 16
    assert actual.reused_tokens == reusable
    np.testing.assert_array_equal(actual.hidden, expected.hidden)
    # Continue generation after rewind, which overwrites a stale prompt or
    # generated suffix while retaining the same position and causal semantics.
    token = embeddings(1, seed=19)
    continued = decoder.step(token, new_length, actual.states)
    fresh_continued = decoder.step(token, new_length, expected.states)
    np.testing.assert_array_equal(continued, fresh_continued)


def test_repeated_random_revisions_match_fresh_state_and_bound_snapshot_storage():
    decoder = CausalDecoder()
    context = decoder.context()
    rng = np.random.default_rng(25)
    prompt = embeddings(97)
    state_ids = [id(state) for state in context.states]
    for _ in range(50):
        length = int(rng.integers(1, 110))
        change = int(rng.integers(0, length))
        revised = embeddings(length, seed=int(rng.integers(1000)))
        preserved = min(change, prompt.shape[-1])
        revised[..., :preserved] = prompt[..., :preserved]
        actual = decoder.prefill(context, revised)
        expected = decoder.prefill(decoder.context(), revised)
        np.testing.assert_array_equal(actual.hidden, expected.hidden)
        for position in range(length, length + 5):
            token = embeddings(1, seed=position)
            np.testing.assert_array_equal(
                decoder.step(token, position, actual.states),
                decoder.step(token, position, expected.states),
            )
        assert [id(state) for state in context.states] == state_ids
        assert context._prompt.nbytes == revised.nbytes
        prompt = revised


def test_snapshot_does_not_borrow_caller_or_decoder_output_storage():
    decoder = CausalDecoder()
    context = decoder.context()
    prompt = embeddings(49)
    original = prompt.copy()
    decoder.prefill(context, prompt)
    prompt[..., 17] += 1
    actual = decoder.prefill(context, prompt)
    assert actual.reused_tokens == 16
    expected = decoder.prefill(decoder.context(), prompt)
    np.testing.assert_array_equal(actual.hidden, expected.hidden)
    assert not np.shares_memory(context._prompt, prompt)
    assert actual.hidden.flags.owndata
    assert not np.array_equal(original, context._prompt)


def test_failed_partition_update_invalidates_reuse_and_retry_overwrites_from_zero():
    decoder = CausalDecoder()
    context = decoder.context()
    original = embeddings(53)
    decoder.prefill(context, original)
    revised = original.copy()
    revised[..., 19:] += 1
    decoder.fail_at = 32
    with pytest.raises(RuntimeError, match="partition failure"):
        decoder.prefill(context, revised)
    assert context._prompt is None
    decoder.fail_at = None
    actual = decoder.prefill(context, revised)
    assert actual.reused_tokens == 0
    expected = decoder.prefill(decoder.context(), revised)
    np.testing.assert_array_equal(actual.hidden, expected.hidden)


def test_reset_owner_checks_and_dtype_changes():
    decoder = CausalDecoder()
    context = decoder.context()
    prompt = embeddings(64)
    decoder.prefill(context, prompt)
    with pytest.raises(ValueError, match="different runtime"):
        context.prefill(prompt, decode_step=decoder.step, owner=object())
    assert decoder.prefill(context, prompt).reused_tokens == 48
    assert decoder.prefill(context, prompt.astype(np.float64)).reused_tokens == 0
    states = context.states
    context.reset()
    assert context.states is states
    assert decoder.prefill(context, prompt).reused_tokens == 0
    assert context.states is not states


def test_failure_replaces_poisoned_future_cache_before_retry():
    decoder = CausalDecoder()
    context = decoder.context()
    prompt = embeddings(53)
    decoder.prefill(context, prompt)
    previous_states = context.states
    for state in previous_states:
        state[..., 91] = np.nan
    context.reset()
    actual = decoder.prefill(context, prompt)
    assert actual.reused_tokens == 0
    assert all(np.isfinite(state).all() for state in actual.states)
    assert not any(state is old for state in actual.states for old in previous_states)
    expected = decoder.prefill(decoder.context(), prompt)
    np.testing.assert_array_equal(actual.hidden, expected.hidden)


def test_reset_without_state_factory_requires_new_context():
    context = DecoderPrefixContext(
        owner=None, states=[object()], token_batch_size=16, cache_length=128
    )
    context.reset()
    with pytest.raises(RuntimeError, match="fresh context"):
        context.prefill(embeddings(3), decode_step=lambda *args: None, owner=None)


@pytest.mark.parametrize(
    "prompt",
    [
        np.empty((1, 4, 1, 0)),
        np.zeros((1, 4, 1, 129)),
        np.zeros((2, 4, 1, 5)),
        np.zeros((1, 4, 2, 5)),
        np.zeros((1, 4, 1, 5), dtype=np.int32),
        np.full((1, 4, 1, 5), np.nan),
        np.full((1, 4, 1, 5), np.inf),
    ],
)
def test_invalid_prompts_do_not_execute(prompt):
    decoder = CausalDecoder()
    with pytest.raises(ValueError, match="Prompt"):
        decoder.prefill(decoder.context(), prompt)
    assert not decoder.calls


def test_invalid_hidden_invalidates_cache_proof():
    decoder = CausalDecoder()
    context = decoder.context()
    prompt = embeddings(33)
    decoder.prefill(context, prompt)
    with pytest.raises(RuntimeError, match="invalid final hidden"):
        context.prefill(
            prompt,
            decode_step=lambda *args: np.full((1, 4, 1, 1), np.nan),
            owner=decoder,
        )
    assert decoder.prefill(context, prompt).reused_tokens == 0


@pytest.mark.parametrize("batch_size, cache_length", [(0, 128), (129, 128), (True, 128), (16, 0)])
def test_invalid_context_dimensions(batch_size, cache_length):
    with pytest.raises(ValueError):
        DecoderPrefixContext(
            owner=object(),
            states=[object()],
            token_batch_size=batch_size,
            cache_length=cache_length,
        )


def test_rewind_matches_fresh_tiny_torch_decoder_with_runtime_masks():
    """Exercise the actual T16 masks and converted architecture without Core ML."""
    torch = pytest.importorskip("torch")
    from std_qwen3asr_ane.conversion.decoder import DecoderPartition
    from std_qwen3asr_ane.runtime import CoreMLRuntime

    torch.manual_seed(12)
    config = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
    }
    template = DecoderPartition(config, 1, 128, token_batch_size=16).eval()

    class TorchModel:
        def make_state(self):
            return deepcopy(template)

        def predict(self, data, *, state):
            with torch.inference_mode():
                output = state(
                    *(
                        torch.from_numpy(data[key].astype(np.float32))
                        for key in (
                            "hidden_states",
                            "cosine",
                            "sine",
                            "attention_mask",
                            "update_mask",
                        )
                    )
                )
            return {"output_hidden_states": output.numpy().copy()}

    runtime = CoreMLRuntime.__new__(CoreMLRuntime)
    runtime.embeddings = np.zeros((2, 8))
    runtime.cache_length = 128
    runtime.token_batch_size = 16
    runtime.decoders = [TorchModel(), TorchModel()]
    phases = np.arange(128, dtype=np.float32)[:, None] / np.array([1, 1000])[None]
    runtime.cosine = np.cos(phases).astype(np.float32)
    runtime.sine = np.sin(phases).astype(np.float32)

    def make_context():
        return DecoderPrefixContext(
            owner=runtime,
            states=[model.make_state() for model in runtime.decoders],
            token_batch_size=16,
            cache_length=128,
        )

    prompt = np.random.default_rng(12).normal(size=(1, 8, 1, 57)).astype(np.float32)
    reused = make_context()
    for length, changed in [(37, 0), (57, 29), (57, 48), (32, 16), (53, 31)]:
        revised = prompt[..., :length].copy()
        revised[..., changed:] += 0.5
        actual = reused.prefill(revised, decode_step=runtime._decode_step, owner=runtime)
        fresh = make_context()
        expected = fresh.prefill(revised, decode_step=runtime._decode_step, owner=runtime)
        np.testing.assert_array_equal(actual.hidden, expected.hidden)
        for position in range(length, length + 3):
            token = prompt[..., position % 57 : position % 57 + 1]
            np.testing.assert_array_equal(
                runtime._decode_step(token, position, actual.states),
                runtime._decode_step(token, position, expected.states),
            )
        for state, reference in zip(actual.states, expected.states, strict=True):
            for name, value in state.named_buffers():
                torch.testing.assert_close(
                    value[..., : length + 3],
                    dict(reference.named_buffers())[name][..., : length + 3],
                    atol=0,
                    rtol=0,
                )
