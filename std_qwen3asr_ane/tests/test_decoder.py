"""Numerical invariants for the static decoder representation."""

import copy

import pytest
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerTextDecoderLayer
from transformers.cache_utils import DynamicCache

from std_qwen3asr_ane.conversion.decoder import DecoderPartition, StableRMSNorm, stable_silu


@pytest.mark.parametrize("width", [1, 4])
def test_grouped_attention_preserves_causal_outputs_and_kv_states(width):
    torch.manual_seed(17)
    config = Qwen3ASRTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
    ).to_dict()
    reference = DecoderPartition(config, 2, 32, token_batch_size=width).eval()
    grouped = copy.deepcopy(reference)
    for layer in grouped.layers:
        layer.enable_grouped_attention()
    with torch.inference_mode():
        for position in (0, width, 2 * width, width):
            x = torch.randn(1, 64, 1, width)
            row_positions = torch.arange(position, position + width)
            angles = row_positions / 1000000 ** (torch.arange(4).float()[:, None] / 4)
            visible = torch.arange(32)[None] <= row_positions[:, None]
            mask = torch.where(visible, 0.0, -10000.0)[None, None]
            update = torch.nn.functional.one_hot(row_positions, 32).float()[None, None]
            inputs = (x, angles.cos()[None, :, None], angles.sin()[None, :, None], mask, update)
            torch.testing.assert_close(grouped(*inputs), reference(*inputs), atol=2e-6, rtol=2e-6)
            for (_, expected), (_, actual) in zip(
                reference.named_buffers(), grouped.named_buffers(), strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_stable_silu_preserves_reference_across_activation_range():
    x = torch.linspace(-200, 200, 40001)
    expected = torch.nn.functional.silu(x)
    torch.testing.assert_close(stable_silu(x), expected, atol=2e-5, rtol=2e-6)
    half = stable_silu(x.half())
    assert torch.isfinite(half).all()
    torch.testing.assert_close(half.float(), expected, atol=0.1, rtol=0.003)


@pytest.mark.parametrize("magnitude", [0.0, 1e-5, 0.01, 1, 100, 10000])
def test_stable_normalization_preserves_extremes(magnitude):
    torch.manual_seed(4)
    x = (torch.randn(2, 128, 1, 1) * magnitude).half()
    norm = StableRMSNorm(128).half()
    expected = x.float() * torch.rsqrt(x.float().square().mean(1, keepdim=True) + 1e-6)
    actual = norm(x).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.006)


def test_stateful_partition_matches_official_causal_layer():
    torch.manual_seed(12)
    config = Qwen3ASRTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    official = Qwen3ASRThinkerTextDecoderLayer(config, 0).eval()
    partition = DecoderPartition(config.to_dict(), 1, 8).eval()
    layer = partition.layers[0]
    state = official.state_dict()
    for name, parameter in layer.named_parameters():
        if name.startswith(("input_layernorm", "post_attention_layernorm")):
            source_name = name
        elif name.startswith(("gate_proj", "up_proj", "down_proj")):
            source_name = "mlp." + name
        else:
            source_name = "self_attn." + name
        parameter.data.copy_(state[source_name].reshape(parameter.shape))
    cache = DynamicCache()
    with torch.inference_mode():
        for position in range(5):
            x = torch.randn(1, 1, 32)
            phase = position / 1000000 ** (torch.arange(4).float() / 4)
            cos, sin = phase.cos(), phase.sin()
            expected = official(
                x,
                (cos.repeat(2).reshape(1, 1, 8), sin.repeat(2).reshape(1, 1, 8)),
                past_key_values=cache,
            )
            mask = torch.full((1, 1, 1, 8), -10000.0)
            mask[..., : position + 1] = 0
            update = torch.zeros_like(mask)
            update[..., position] = 1
            actual = (
                partition(
                    x.transpose(1, 2).unsqueeze(2),
                    cos.reshape(1, 4, 1, 1),
                    sin.reshape(1, 4, 1, 1),
                    mask,
                    update,
                )
                .squeeze(2)
                .transpose(1, 2)
            )
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_chunked_prefill_preserves_causal_attention_and_unused_cache():
    torch.manual_seed(9)
    config = Qwen3ASRTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    ).to_dict()
    sequential = DecoderPartition(config, 1, 16).eval()
    batched = DecoderPartition(config, 1, 16, token_batch_size=4).eval()
    batched.load_state_dict(sequential.state_dict())
    inputs = torch.randn(1, 32, 1, 7)
    expected = []
    with torch.inference_mode():
        for position in range(7):
            phase = position / 1000000 ** (torch.arange(4).float() / 4)
            mask = torch.full((1, 1, 1, 16), -10000.0)
            mask[..., : position + 1] = 0
            update = torch.zeros_like(mask)
            update[..., position] = 1
            expected.append(
                sequential(
                    inputs[..., position : position + 1],
                    phase.cos().reshape(1, 4, 1, 1),
                    phase.sin().reshape(1, 4, 1, 1),
                    mask,
                    update,
                )
            )
        actual = []
        for start, valid in [(0, 4), (4, 2), (6, 1)]:
            hidden = torch.zeros(1, 32, 1, 4)
            hidden[..., :valid] = inputs[..., start : start + valid]
            phase = (
                torch.arange(start, start + 4)[None, :]
                / (1000000 ** (torch.arange(4).float() / 4))[:, None]
            )
            mask = torch.full((1, 1, 4, 16), -10000.0)
            update = torch.zeros_like(mask)
            for row in range(4):
                mask[..., row, : min(start + row + 1, start + valid)] = 0
                if row < valid:
                    update[..., row, start + row] = 1
            actual.append(
                batched(
                    hidden,
                    phase.cos().reshape(1, 4, 1, 4),
                    phase.sin().reshape(1, 4, 1, 4),
                    mask,
                    update,
                )[..., :valid]
            )
        torch.testing.assert_close(
            torch.cat(actual, -1), torch.cat(expected, -1), atol=3e-6, rtol=3e-6
        )
        torch.testing.assert_close(batched.key_0, sequential.key_0, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(batched.value_0, sequential.value_0, atol=3e-6, rtol=3e-6)
