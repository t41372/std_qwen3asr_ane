"""Scan every real Qwen3-ASR decoder layer for FP16 overflow and numerical drift.

Uses bounded memory by loading one official layer and two rewritten layers at a
time. Each path propagates its own residual stream across all layers. The
rewritten paths decode teacher-forced inputs through the actual stateful graph.
Optional audio embeddings must come from real audio; no synthetic fixture is
silently substituted. This CPU scan does not emulate ANE subnormal flushing.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
    Qwen3ASRTextConfig,
)
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRThinkerTextDecoderLayer,
)
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    StableRMSNorm,
    load_partition,
)
from torch.nn import functional as F

PROMPT = [151644, 8948, 198, 151645, 198, 151644, 872, 198]


def statistics(tensor: torch.Tensor) -> dict:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    return {
        "max_absolute_finite": float(values[finite].abs().max())
        if finite.any()
        else None,
        "nan_count": int(values.isnan().sum()),
        "infinity_count": int(values.isinf().sum()),
        "elements": values.numel(),
    }


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    if not torch.isfinite(actual).all():
        return {"max_absolute_error": None, "relative_l2_error": None}
    return {
        "max_absolute_error": float((actual - expected).abs().max()),
        "relative_l2_error": float((actual - expected).norm() / expected.norm()),
    }


class Activations:
    def __init__(self, layer, official: bool):
        self.values = {}
        mlp = layer.mlp if official else layer
        self.hooks = []
        for name in ("gate_proj", "up_proj"):
            self.hooks.append(
                getattr(mlp, name).register_forward_hook(self.output_hook(name))
            )
        self.hooks.append(mlp.down_proj.register_forward_pre_hook(self.product_hook))

    def record(self, name, value):
        row = statistics(value)
        previous = self.values.get(name)
        if previous:
            maxima = [
                n
                for n in [previous["max_absolute_finite"], row["max_absolute_finite"]]
                if n is not None
            ]
            row["max_absolute_finite"] = max(maxima) if maxima else None
            for key in ("nan_count", "infinity_count", "elements"):
                row[key] += previous[key]
        self.values[name] = row

    def output_hook(self, name):
        def hook(module, inputs, output):
            self.record(name, output)

        return hook

    def product_hook(self, module, inputs):
        self.record("silu_gate_times_up", inputs[0])

    def close(self):
        for hook in self.hooks:
            hook.remove()


def rewritten_forward(partition, hidden, config):
    sequence = hidden.shape[1]
    dtype = next(partition.parameters()).dtype
    width = config["head_dim"] // 2
    outputs = []
    for position in range(sequence):
        frequency = position / config["rope_theta"] ** (
            torch.arange(width).float() / width
        )
        cosine = frequency.cos().reshape(1, width, 1, 1).to(dtype)
        sine = frequency.sin().reshape(1, width, 1, 1).to(dtype)
        mask = torch.full((1, 1, 1, sequence), -10000.0, dtype=dtype)
        mask[..., : position + 1] = 0
        update = torch.zeros_like(mask)
        update[..., position] = 1
        x = hidden[:, position : position + 1].transpose(1, 2).unsqueeze(2).to(dtype)
        outputs.append(
            partition(x, cosine, sine, mask, update).squeeze(2).transpose(1, 2)
        )
    return torch.cat(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/probes/decoder-full-numerics.json"),
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--audio-embeddings",
        type=Path,
        help="Real audio embeddings in a [tokens,2048] .npy",
    )
    parser.add_argument("--max-audio-tokens", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    official_config = Qwen3ASRTextConfig(**config)
    official_config._attn_implementation = "eager"
    weights = SourceWeights(args.source)
    embedding = weights.get("thinker.model.embed_tokens.weight")
    initial = embedding[PROMPT].unsqueeze(0)
    if args.audio_embeddings:
        audio = torch.from_numpy(np.load(args.audio_embeddings)).float()
        if audio.ndim != 2 or audio.shape[1] != config["hidden_size"]:
            raise ValueError("--audio-embeddings must contain [tokens, hidden_size]")
        initial = torch.cat(
            (initial, audio[: args.max_audio_tokens].unsqueeze(0)), dim=1
        )
    del embedding
    sequence = initial.shape[1]
    reference, rewritten32, rewritten16 = (
        initial,
        initial / args.scale,
        (initial / args.scale).half(),
    )
    frequencies = torch.arange(sequence).float()[:, None] / config["rope_theta"] ** (
        torch.arange(config["head_dim"] // 2).float()[None] / (config["head_dim"] // 2)
    )
    full_cosine = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1)[None]
    full_sine = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1)[None]
    causal_mask = torch.full((sequence, sequence), -10000.0).triu(1)[None, None]
    report = {
        "residual_scale": args.scale,
        "prompt_ids": PROMPT,
        "audio_embeddings": str(args.audio_embeddings)
        if args.audio_embeddings
        else None,
        "sequence_length": sequence,
        "precision_scope": "CPU FP16 operations; ANE flush-to-zero behavior is not simulated",
        "layers": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for index in range(config["num_hidden_layers"]):
            official = Qwen3ASRThinkerTextDecoderLayer(official_config, index).eval()
            official.load_state_dict(
                {
                    name: weights.get(f"thinker.model.layers.{index}." + name)
                    for name in official.state_dict()
                }
            )
            partition32 = DecoderPartition(
                config, 1, sequence, residual_scale=args.scale
            ).eval()
            load_partition(partition32, weights, index)
            partition16 = DecoderPartition(
                config, 1, sequence, residual_scale=args.scale
            ).eval()
            load_partition(partition16, weights, index)
            partition16.half()
            trackers = [
                Activations(official, True),
                Activations(partition32.layers[0], False),
                Activations(partition16.layers[0], False),
            ]
            reference = official(
                reference, (full_cosine, full_sine), attention_mask=causal_mask
            )
            rewritten32 = rewritten_forward(partition32, rewritten32, config)
            rewritten16 = rewritten_forward(partition16, rewritten16, config)
            row = {"layer": index}
            for label, hidden, tracker in zip(
                ("official_fp32", "rewritten_fp32", "rewritten_fp16"),
                (reference, rewritten32, rewritten16),
                trackers,
                strict=True,
            ):
                tracker.record("residual", hidden)
                row[label] = tracker.values
                tracker.close()
            row["fp32_error"] = difference(rewritten32 * args.scale, reference)
            row["fp16_error"] = difference(rewritten16.float() * args.scale, reference)
            report["layers"].append(row)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "layer": index,
                        "fp32_error": row["fp32_error"],
                        "fp16_error": row["fp16_error"],
                        "fp16_residual": row["rewritten_fp16"]["residual"],
                        "fp16_product": row["rewritten_fp16"]["silu_gate_times_up"],
                    }
                ),
                flush=True,
            )
            del official, partition32, partition16, trackers
            gc.collect()
        norm_weight = weights.get("thinker.model.norm.weight")
        official_final = reference * torch.rsqrt(
            reference.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
        )
        official_final *= norm_weight
        normalized = {"official_fp32": official_final}
        for label, hidden in [
            ("rewritten_fp32", rewritten32),
            ("rewritten_fp16", rewritten16),
        ]:
            norm = StableRMSNorm(
                config["hidden_size"], config["rms_norm_eps"], args.scale
            ).to(hidden.dtype)
            norm.weight.copy_(norm_weight)
            normalized[label] = (
                norm(hidden.transpose(1, 2).unsqueeze(2)).squeeze(2).transpose(1, 2)
            )
        embedding = weights.get("thinker.model.embed_tokens.weight")
        logits = {label: [] for label in normalized}
        for chunk in embedding.split(4096):
            for label, hidden in normalized.items():
                logits[label].append(F.linear(hidden, chunk.to(hidden.dtype)).float())
        logits = {label: torch.cat(chunks, dim=-1) for label, chunks in logits.items()}
        reference_logits = logits["official_fp32"]
        report["logits"] = {}
        for label, actual in logits.items():
            report["logits"][label] = {
                **statistics(actual),
                **difference(actual, reference_logits),
                "argmax_ids": actual.argmax(-1).tolist(),
                "argmax_agreement": float(
                    (actual.argmax(-1) == reference_logits.argmax(-1)).float().mean()
                )
                if torch.isfinite(actual).all()
                else None,
            }
    report["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"logits": report["logits"], "elapsed_seconds": report["elapsed_seconds"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
