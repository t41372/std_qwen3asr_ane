"""Static, stateful decoder partitions with bounded FP16 intermediates.

The residual stream is represented as x / residual_scale. Attention values
and the MLP up branch use the same scale, so the transformation is algebraic,
without clipping activations or changing weights through calibration.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


class StableRMSNorm(nn.Module):
    """Channel RMS normalization without squaring large FP16 values."""

    def __init__(self, width: int, eps: float = 1e-6, input_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.input_scale = input_scale

    def forward(self, x):
        scale = torch.maximum(
            torch.amax(torch.abs(x), dim=1, keepdim=True),
            torch.tensor(self.eps**0.5, dtype=x.dtype, device=x.device),
        )
        bounded = x / scale
        variance = (bounded * bounded).mean(dim=1, keepdim=True)
        original_scale = scale * self.input_scale
        # ANE can flush subnormal FP16 values. Express epsilon through its
        # normal-range square root, before division, so small activations keep
        # the stabilizer instead of silently turning RMSNorm into x / rms(x).
        relative_epsilon = self.eps**0.5 / original_scale
        inverse = torch.rsqrt(variance + relative_epsilon * relative_epsilon)
        return bounded * inverse * self.weight.view(1, -1, 1, 1)


def projection(input_width: int, output_width: int) -> nn.Conv2d:
    return nn.Conv2d(input_width, output_width, 1, bias=False)


def stable_silu(x: torch.Tensor) -> torch.Tensor:
    """Exact SiLU algebra that avoids Core ML's inaccurate native ANE fusion.

    Both exponential arguments are nonpositive and the denominator is in [1, 2].
    Native ANE SiLU showed 1–2% error on actual Qwen activations; this expression
    reduced the isolated error by 26–39 times without moving it off ANE.
    """
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    numerator = x * torch.exp(torch.minimum(x, zero))
    return numerator / (1 + torch.exp(-torch.abs(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: dict, cache_length: int, residual_scale: float, token_batch_size=1):
        super().__init__()
        width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.cache_length = cache_length
        self.residual_scale = residual_scale
        self.token_batch_size = token_batch_size
        eps = config["rms_norm_eps"]
        self.input_layernorm = StableRMSNorm(width, eps, residual_scale)
        self.post_attention_layernorm = StableRMSNorm(width, eps, residual_scale)
        self.q_proj = projection(width, self.heads * self.head_dim)
        self.k_proj = projection(width, self.kv_heads * self.head_dim)
        self.v_proj = projection(width, self.kv_heads * self.head_dim)
        self.o_proj = projection(self.heads * self.head_dim, width)
        self.q_norm = StableRMSNorm(self.head_dim, eps)
        self.k_norm = StableRMSNorm(self.head_dim, eps)
        self.gate_proj = projection(width, config["intermediate_size"])
        self.up_proj = projection(width, config["intermediate_size"])
        self.down_proj = projection(config["intermediate_size"], width)

    def _rotary(self, x, cosine, sine):
        half = self.head_dim // 2
        left, right = x[:, :half], x[:, half:]
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), 1)

    def forward(self, x, cosine, sine, mask, update_mask, key_cache, value_cache):
        normalized = self.input_layernorm(x)
        q = self.q_norm(
            self.q_proj(normalized).reshape(self.heads, self.head_dim, 1, self.token_batch_size)
        )
        k = self.k_norm(
            self.k_proj(normalized).reshape(self.kv_heads, self.head_dim, 1, self.token_batch_size)
        )
        q, k = self._rotary(q, cosine, sine), self._rotary(k, cosine, sine)
        v = self.v_proj(normalized).reshape(self.kv_heads, self.head_dim, 1, self.token_batch_size)
        if self.token_batch_size == 1:
            key_cache.mul_(1 - update_mask)
            key_cache.add_(k * update_mask)
            value_cache.mul_(1 - update_mask)
            value_cache.add_(v * update_mask)
        else:
            occupied = update_mask.sum(dim=2, keepdim=True)
            key_cache.mul_(1 - occupied)
            key_cache.add_(torch.matmul(k.transpose(1, 2), update_mask).transpose(1, 2))
            value_cache.mul_(1 - occupied)
            value_cache.add_(torch.matmul(v.transpose(1, 2), update_mask).transpose(1, 2))
        # Each query head uses one KV head; avoiding a materialized GQA repeat
        # lets the compiler reuse the cache without a rank-five tensor.
        outputs = []
        group_size = self.heads // self.kv_heads
        for head in range(self.heads):
            kv = head // group_size
            scores = (
                torch.matmul(
                    q[head : head + 1].permute(0, 2, 3, 1),
                    key_cache[kv : kv + 1].transpose(1, 2),
                )
                * self.head_dim**-0.5
            )
            probabilities = torch.softmax(scores + mask, dim=-1)
            attended = torch.matmul(probabilities, value_cache[kv : kv + 1].permute(0, 2, 3, 1))
            outputs.append(attended.permute(0, 3, 1, 2))
        x = x + self.o_proj(torch.cat(outputs, dim=1))
        normalized = self.post_attention_layernorm(x)
        return x + self.down_proj(
            stable_silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        )


class DecoderPartition(nn.Module):
    """A few decoder layers and their persistent KV buffers."""

    def __init__(
        self, config: dict, layers: int, cache_length: int, residual_scale=1.0, token_batch_size=1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(config, cache_length, residual_scale, token_batch_size)
            for _ in range(layers)
        )
        self.token_batch_size = token_batch_size
        cache_shape = (config["num_key_value_heads"], config["head_dim"], 1, cache_length)
        for index in range(layers):
            self.register_buffer(f"key_{index}", torch.zeros(cache_shape))
            self.register_buffer(f"value_{index}", torch.zeros(cache_shape))

    def forward(self, hidden_states, cosine, sine, attention_mask, update_mask):
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cosine,
                sine,
                attention_mask,
                update_mask,
                getattr(self, f"key_{index}"),
                getattr(self, f"value_{index}"),
            )
        return hidden_states


class LanguageHead(nn.Module):
    """Vocabulary projections split below the ANE channel dimension limit."""

    def __init__(self, config: dict, residual_scale=1.0, vocabulary_chunk=8192):
        super().__init__()
        self.norm = StableRMSNorm(config["hidden_size"], config["rms_norm_eps"], residual_scale)
        self.heads = nn.ModuleList(
            projection(config["hidden_size"], min(vocabulary_chunk, config["vocab_size"] - start))
            for start in range(0, config["vocab_size"], vocabulary_chunk)
        )

    def forward(self, hidden_states):
        normalized = self.norm(hidden_states)
        return tuple(head(normalized).reshape(1, -1) for head in self.heads)


class SourceWeights:
    """Read only requested tensors from the sharded official checkpoint."""

    def __init__(self, source: Path):
        self.source = source
        self.index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        with safe_open(self.source / self.index[name], framework="pt", device="cpu") as archive:
            return archive.get_tensor(name).float()


def load_partition(module: DecoderPartition, weights: SourceWeights, first_layer: int):
    for local_index, layer in enumerate(module.layers):
        prefix = f"thinker.model.layers.{first_layer + local_index}."
        for name, parameter in layer.named_parameters():
            if name.startswith(("input_layernorm", "post_attention_layernorm")):
                source_name = prefix + name
            elif name.startswith(("gate_proj", "up_proj", "down_proj")):
                source_name = prefix + "mlp." + name
            else:
                source_name = prefix + "self_attn." + name
            value = weights.get(source_name)
            if name in ("v_proj.weight", "up_proj.weight"):
                value = value / layer.residual_scale
            parameter.data.copy_(value.reshape(parameter.shape))


def compute_precision(mode: str):
    """Keep heavy projections on FP16 while optionally preserving scalar math."""
    import coremltools as ct

    if mode == "fp16":
        return ct.precision.FLOAT16
    if mode == "mixed":
        return ct.transform.FP16ComputePrecision(
            op_selector=lambda operation: operation.op_type in {"conv", "matmul", "softmax"}
        )
    raise ValueError(f"Unknown decoder precision: {mode}")


def convert_partition(
    module: DecoderPartition, config: dict, output: Path, cache_length: int, precision="fp16"
):
    import coremltools as ct

    from .passes import verify_activation_operators

    count = module.token_batch_size
    examples = (
        torch.zeros(1, config["hidden_size"], 1, count),
        torch.ones(1, config["head_dim"] // 2, 1, count),
        torch.zeros(1, config["head_dim"] // 2, 1, count),
        torch.zeros(1, 1, count, cache_length),
        F.one_hot(torch.arange(count), cache_length).float().reshape(1, 1, count, cache_length),
    )
    module.eval()
    traced = torch.jit.trace(module, examples, check_trace=False)
    names = ("hidden_states", "cosine", "sine", "attention_mask", "update_mask")
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=tuple(value.shape), dtype=np.float16), name=name
        )
        for name, value in module.named_buffers()
    ]
    io_dtype = np.float32 if precision == "mixed" else np.float16
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=name, shape=tuple(value.shape), dtype=io_dtype)
            for name, value in zip(names, examples)
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=io_dtype)],
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=compute_precision(precision),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    verify_activation_operators(model)
    model.save(str(output))


def build_decoder(
    source: Path,
    output: Path,
    *,
    cache_length=1024,
    layers_per_partition=4,
    residual_scale=1.0,
    token_batch_size=1,
):
    import coremltools as ct

    config = json.loads((source / "config.json").read_text())["thinker_config"]["text_config"]
    weights = SourceWeights(source)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    partitions = []
    for start in range(0, config["num_hidden_layers"], layers_per_partition):
        name = f"decoder_{start:02d}.mlpackage"
        module = DecoderPartition(
            config,
            min(layers_per_partition, config["num_hidden_layers"] - start),
            cache_length,
            residual_scale=residual_scale,
            token_batch_size=token_batch_size,
        )
        load_partition(module, weights, start)
        convert_partition(module, config, output / name, cache_length)
        partitions.append(name)
        files["decoder" if start == 0 else f"decoder_{start:02d}"] = name
        del module
        gc.collect()
    embedding = weights.get("thinker.model.embed_tokens.weight")
    np.save(output / "embedding.npy", embedding.numpy().astype(np.float16))
    files["embedding"] = "embedding.npy"
    head = LanguageHead(config, residual_scale=residual_scale).eval()
    head.norm.weight.data.copy_(weights.get("thinker.model.norm.weight"))
    offset = 0
    for projection_layer in head.heads:
        count = projection_layer.out_channels
        projection_layer.weight.data.copy_(embedding[offset : offset + count, :, None, None])
        offset += count
    example = torch.zeros(1, config["hidden_size"], 1, 1)
    traced = torch.jit.trace(head, example)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=tuple(example.shape), dtype=np.float16)],
        outputs=[
            ct.TensorType(name=f"logits_{index}", dtype=np.float16)
            for index in range(len(head.heads))
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    model.save(str(output / "lm_head.mlpackage"))
    files["lm_head"] = "lm_head.mlpackage"
    return {
        "files": files,
        "decoder_partitions": partitions,
        "residual_scale": residual_scale,
        "head_dim": config["head_dim"],
        "rope_theta": config["rope_theta"],
        "max_sequence_length": cache_length,
        "token_batch_size": token_batch_size,
    }
