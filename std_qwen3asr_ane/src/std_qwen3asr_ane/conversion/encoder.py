"""Static, channels-first Qwen3-ASR audio encoder graphs.

The attention graph is one independent FA2 inference window, not whole-clip
attention. Upstream's eager/SDPA encoder currently omits the window mask when
calling its layers, so its whole-clip output is not a valid multi-window oracle.

A clip shorter than 100 mel frames needs intermediate convolution masks: simply
zero-padding its input to 100 changes the right-edge activations in later layers.
For a clip >=100 frames, upstream pads every chunk (including the tail) to 100,
so both masks must instead contain only ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..bundle import validate_offline_frontend_batch_size


def exact_gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU expression; conversion must disable native GELU fusion."""
    return 0.5 * x * (1 + torch.erf(x * (2**-0.5)))


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels without moving channels out of ANE's BCHW layout."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        variance = (centered * centered).mean(dim=1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def projection(in_features: int, out_features: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)


def convolution_masks(frame_count: int, chunk_frames: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Masks for the *whole clip* length, reused for every chunk of that clip."""
    if frame_count < 1:
        raise ValueError("An audio clip must contain at least one mel frame")
    width = min(frame_count, chunk_frames)
    padded_width = chunk_frames
    masks = []
    for _ in range(2):
        width = (width + 1) // 2
        padded_width = (padded_width + 1) // 2
        masks.append((np.arange(padded_width) < width).astype(np.float32)[None, None, None])
    return masks[0], masks[1]


def audio_token_count(frame_count: int) -> int:
    """Match upstream's chunk-wise output length, including a partial last chunk."""
    full_chunks, tail = divmod(frame_count, 100)
    return full_chunks * 13 + (tail + 7) // 8


class AudioFrontend(nn.Module):
    def __init__(self, config: dict[str, Any], *, batch_size: int = 1):
        super().__init__()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Frontend batch size must be a positive integer")
        self.batch_size = batch_size
        self.chunk_frames = config["n_window"] * 2
        channels = config["downsample_hidden_size"]
        self.conv2d1 = nn.Conv2d(1, channels, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        frequency_bins = (config["num_mel_bins"] + 7) // 8
        self.flattened_channels = channels * frequency_bins
        self.conv_out = projection(self.flattened_channels, config["d_model"], bias=False)
        token_count = (self.chunk_frames + 7) // 8
        inverse_timescales = torch.exp(
            -math.log(10000)
            / (config["d_model"] // 2 - 1)
            * torch.arange(config["d_model"] // 2).float()
        )
        phase = torch.arange(token_count)[:, None] * inverse_timescales[None]
        positions = torch.cat((phase.sin(), phase.cos()), dim=1).T[None, :, None]
        self.register_buffer("positions", positions, persistent=False)

    def forward(
        self, mel_features: torch.Tensor, conv1_mask: torch.Tensor, conv2_mask: torch.Tensor
    ) -> torch.Tensor:
        x = exact_gelu(self.conv2d1(mel_features)) * conv1_mask
        x = exact_gelu(self.conv2d2(x)) * conv2_mask
        x = exact_gelu(self.conv2d3(x))
        x = x.reshape(self.batch_size, self.flattened_channels, 1, -1)
        return self.conv_out(x) + self.positions


class AudioAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        channels = config["d_model"]
        self.heads = config["encoder_attention_heads"]
        self.head_dim = channels // self.heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = projection(channels, channels)
        self.k_proj = projection(channels, channels)
        self.v_proj = projection(channels, channels)
        self.out_proj = projection(channels, channels)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        queries = self.q_proj(x).split(self.head_dim, dim=1)
        keys = self.k_proj(x).split(self.head_dim, dim=1)
        values = self.v_proj(x).split(self.head_dim, dim=1)
        outputs = []
        for query, key, value in zip(queries, keys, values, strict=True):
            scores = torch.einsum("bchq,bchk->bkhq", query, key) * self.scaling
            probabilities = torch.softmax(scores + key_mask, dim=1)
            outputs.append(torch.einsum("bkhq,bchk->bchq", probabilities, value))
        return self.out_proj(torch.cat(outputs, dim=1))


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        channels = config["d_model"]
        self.self_attn_layer_norm = ChannelLayerNorm(channels)
        self.self_attn = AudioAttention(config)
        self.final_layer_norm = ChannelLayerNorm(channels)
        self.fc1 = projection(channels, config["encoder_ffn_dim"])
        self.fc2 = projection(config["encoder_ffn_dim"], channels)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), key_mask)
        x = x + self.fc2(exact_gelu(self.fc1(self.final_layer_norm(x))))
        return x


class AudioTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.layers = nn.ModuleList(
            [AudioEncoderLayer(config) for _ in range(config["encoder_layers"])]
        )
        self.ln_post = ChannelLayerNorm(config["d_model"])
        self.proj1 = projection(config["d_model"], config["d_model"])
        self.proj2 = projection(config["d_model"], config["output_dim"])

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, key_mask)
        return self.proj2(exact_gelu(self.proj1(self.ln_post(hidden_states))))


def load_encoder_weights(module: nn.Module, source_dir: Path) -> None:
    """Load only this graph's tensors, expanding linear weights into 1x1 kernels."""
    from safetensors import safe_open

    prefix = "thinker.audio_tower."
    expected = module.state_dict()
    loaded = {}
    for file in sorted(source_dir.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                name = key.removeprefix(prefix)
                if not key.startswith(prefix) or name not in expected:
                    continue
                tensor = tensors.get_tensor(key).float()
                if tensor.ndim == 2 and expected[name].ndim == 4:
                    tensor = tensor[:, :, None, None]
                loaded[name] = tensor
    module.load_state_dict(loaded, strict=True)


def build_encoder(
    source_dir: str | Path, output_dir: str | Path, *, frontend_batch_size: int = 1
) -> dict[str, Any]:
    """Export the frontend and one masked attention window; return bundle metadata."""
    import coremltools as ct

    from .passes import ane_pass_pipeline, verify_activation_operators

    source_dir, output_dir = Path(source_dir), Path(output_dir)
    validate_offline_frontend_batch_size(frontend_batch_size)
    config = json.loads((source_dir / "config.json").read_text())["thinker_config"]["audio_config"]
    if config["n_window"] != 50 or config["n_window_infer"] != 800:
        raise ValueError("This export targets Qwen3-ASR's 100-frame chunks and 800-frame windows")
    if config["activation_function"] != "gelu":
        raise ValueError("Only the reference GELU audio encoder is supported")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_frames = config["n_window"] * 2
    chunk_tokens = (chunk_frames + 7) // 8
    window_tokens = chunk_tokens * (config["n_window_infer"] // chunk_frames)
    graphs = [
        (
            "frontend",
            AudioFrontend,
            {
                "mel_features": (1, 1, config["num_mel_bins"], chunk_frames),
                "conv1_mask": (1, 1, 1, (chunk_frames + 1) // 2),
                "conv2_mask": (1, 1, 1, (chunk_frames + 3) // 4),
            },
            "chunk_embeddings",
        ),
        (
            "encoder",
            AudioTransformer,
            {
                "hidden_states": (1, config["d_model"], 1, window_tokens),
                "key_mask": (1, window_tokens, 1, 1),
            },
            "audio_embeddings",
        ),
    ]
    if frontend_batch_size > 1:
        graphs.append(
            (
                "frontend_batched",
                lambda config: AudioFrontend(config, batch_size=frontend_batch_size),
                {
                    "mel_features": (frontend_batch_size, 1, config["num_mel_bins"], chunk_frames),
                    "conv1_mask": (frontend_batch_size, 1, 1, (chunk_frames + 1) // 2),
                    "conv2_mask": (frontend_batch_size, 1, 1, (chunk_frames + 3) // 4),
                },
                "chunk_embeddings",
            )
        )
    files = []
    for role, module_type, inputs, output_name in graphs:
        module = module_type(config).eval()
        load_encoder_weights(module, source_dir)
        examples = tuple(torch.zeros(shape) for shape in inputs.values())
        with torch.inference_mode():
            traced = torch.jit.trace(module, examples)
        model = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            skip_model_load=True,
            pass_pipeline=ane_pass_pipeline(),
            inputs=[
                ct.TensorType(name=name, shape=shape, dtype=np.float32)
                for name, shape in inputs.items()
            ],
            outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        )
        verify_activation_operators(model)
        path = output_dir / f"{role}.mlpackage"
        model.save(str(path))
        files.append({"role": role, "path": path.name})
        del model, traced, module
    return {
        "files": files,
        "frontend": {
            "chunk_frames": chunk_frames,
            "chunk_tokens": chunk_tokens,
            "channels": config["d_model"],
            "layout": "BCHW",
            "short_clip_masks": True,
            **({"offline_batch_size": frontend_batch_size} if frontend_batch_size > 1 else {}),
        },
        "encoder": {
            "window_tokens": window_tokens,
            "output_channels": config["output_dim"],
            "layout": "BCHW",
            "attention_semantics": "independent_fa2_windows",
        },
    }
