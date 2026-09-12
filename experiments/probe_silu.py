"""Separate ANE SiLU/product error from upstream decoder projection errors.

Four algebraically equivalent formulas use an identity 1x1 projection context
(2 MB per graph). No production decoder changes or full-model conversions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from probe_decoder_math import FirstToken
from probe_normalization import inspect_placement, metrics
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    load_partition,
)
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (1, 1024, 1, 104)
VARIANTS = ("silu", "sigmoid_product", "tanh_identity", "exp_division", "stable_exp")


class ActivationContext(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.projection = nn.Conv2d(1024, 1024, 1, bias=False)
        self.projection.weight.data.copy_(torch.eye(1024)[:, :, None, None])

    def forward(self, gate, up):
        x = self.projection(gate)
        if self.variant == "silu":
            activated = F.silu(x)
        elif self.variant == "sigmoid_product":
            activated = x * torch.sigmoid(x)
        elif self.variant == "tanh_identity":
            activated = x * (torch.tanh(x / 2) + 1) / 2
        elif self.variant == "exp_division":
            activated = x / (1 + torch.exp(-x))
        else:
            activated = (
                x
                * torch.exp(torch.minimum(x, torch.zeros_like(x)))
                / (1 + torch.exp(-torch.abs(x)))
            )
        return x, activated, activated * up


def pack(value):
    """Repeat actual channels solely to provide enough convolution work for ANE placement."""
    return np.resize(np.asarray(value, dtype=np.float16).reshape(-1), SHAPE).copy()


def point_examples(gate, actual, expected, count=10):
    gate, actual, expected = (
        np.asarray(value, dtype=np.float32).reshape(-1)
        for value in (gate, actual, expected)
    )
    finite_indices = np.flatnonzero(np.isfinite(actual) & np.isfinite(expected))
    indices = finite_indices[
        np.argsort(np.abs(actual[finite_indices] - expected[finite_indices]))[-count:][
            ::-1
        ]
    ]
    return [
        {
            "gate": float(gate[index]),
            "actual_silu": float(actual[index]),
            "exact_silu": float(expected[index]),
            "error": float(actual[index] - expected[index]),
        }
        for index in indices
    ]


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--first-token-model",
        type=Path,
        default=ROOT / "artifacts/probes/decoder-math/first-token.mlpackage",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/probes/silu"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:4])
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    config = json.loads((args.source_dir / "config.json").read_text())[
        "thinker_config"
    ]["text_config"]
    weights = SourceWeights(args.source_dir)
    embeddings = weights.get("thinker.model.embed_tokens.weight")[
        [151644, 198, 872, 8948]
    ].clone()
    report = {
        "schema_version": 1,
        "evidence_kind": "post_hoc_silu_microprobe",
        "input_shape": list(SHAPE),
        "first_token_product_consistency": [],
        "models": [],
        "cases": [],
    }
    fixtures = {
        "grid_minus20_plus20": (
            np.linspace(-20, 20, np.prod(SHAPE)).reshape(SHAPE).astype(np.float16),
            np.ones(SHAPE, dtype=np.float16),
        )
    }
    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
        model = ct.models.MLModel(str(args.first_token_model), compute_units=units)
        for token, embedding in zip((151644, 198, 872, 8948), embeddings, strict=True):
            actual = model.predict(
                {"hidden": embedding.reshape(1, 2048, 1, 1).numpy().astype(np.float16)}
            )
            gate, up, product = (
                np.asarray(actual[name], dtype=np.float32)
                for name in ("gate", "up", "product")
            )
            exact_activation = F.silu(torch.from_numpy(gate)).numpy()
            exact_product = exact_activation * up
            half_product = (
                (F.silu(torch.from_numpy(gate).half()) * torch.from_numpy(up).half())
                .float()
                .numpy()
            )
            row = {
                "compute_units": units.name,
                "token": token,
                "actual_product_vs_exact_from_actual_gate_up": metrics(
                    product, exact_product
                ),
                "actual_product_vs_torch_half_from_actual_gate_up": metrics(
                    product, half_product
                ),
                "activation_inferred_from_product": point_examples(
                    gate,
                    product / np.where(np.abs(up) > 1e-3, up, np.nan),
                    exact_activation,
                ),
            }
            # Restrict inferred activation diagnostics to non-tiny up values.
            row["activation_inferred_from_product"] = [
                item
                for item in row["activation_inferred_from_product"]
                if np.isfinite(item["actual_silu"])
            ]
            report["first_token_product_consistency"].append(row)
            np.savez(
                args.output_dir / f"first-token-{units.name}-{token}.npz",
                gate=gate,
                up=up,
                product=product,
                exact_activation=exact_activation,
                exact_product=exact_product,
            )
            fixtures[f"actual_gate_{units.name}_{token}"] = (pack(gate), pack(up))
            print(json.dumps({"first_token": row}), flush=True)
        del model
    with torch.inference_mode():
        for token, embedding in zip((151644, 198), embeddings[:2], strict=True):
            hidden = embedding.reshape(1, 2048, 1, 1)
            for index in range(3):
                partition = DecoderPartition(config, 1, 1, residual_scale=1.0).eval()
                load_partition(partition, weights, index)
                values = FirstToken(partition.layers[0]).eval()(hidden)
                gate, up, hidden = values[5], values[6], values[9]
                fixtures[f"torch_layer_{index}_{token}"] = (
                    pack(gate.numpy()),
                    pack(up.numpy()),
                )
                del partition, values
    for variant in args.variants:
        module = ActivationContext(variant).eval()
        traced = torch.jit.trace(module, (torch.zeros(SHAPE), torch.ones(SHAPE)))
        converted = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name=name, shape=SHAPE, dtype=np.float16)
                for name in ("gate", "up")
            ],
            outputs=[
                ct.TensorType(name=name, dtype=np.float16)
                for name in ("context_gate", "activated", "product")
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
        )
        operations = [
            op.op_type
            for op in converted._mil_program.functions["main"].operations
            if op.op_type != "const"
        ]
        path = args.output_dir / f"{variant}.mlpackage"
        converted.save(str(path))
        for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
            model = ct.models.MLModel(str(path), compute_units=units)
            metadata = {
                "variant": variant,
                "compute_units": units.name,
                "mil_nonconstant_operations": operations,
            }
            if units == ct.ComputeUnit.CPU_AND_NE:
                metadata["placement"] = inspect_placement(model)
            report["models"].append(metadata)
            for name, (gate, up) in fixtures.items():
                actual = model.predict({"gate": gate, "up": up})
                oracle = F.silu(torch.from_numpy(gate.astype(np.float32))).numpy()
                actual_gate = actual["context_gate"].astype(np.float32)
                contextual_oracle = F.silu(torch.from_numpy(actual_gate)).numpy()
                row = {
                    "case": name,
                    "variant": variant,
                    "compute_units": units.name,
                    "identity_projection": metrics(actual_gate, gate),
                    "activation_vs_fp32": metrics(actual["activated"], oracle),
                    "activation_vs_actual_context_gate_fp32": metrics(
                        actual["activated"], contextual_oracle
                    ),
                    "product_vs_fp32": metrics(
                        actual["product"], oracle * up.astype(np.float32)
                    ),
                    "product_vs_reported_activation_times_up_fp32": metrics(
                        actual["product"],
                        actual["activated"].astype(np.float32) * up.astype(np.float32),
                    ),
                }
                if name == "grid_minus20_plus20":
                    row["largest_activation_errors"] = point_examples(
                        gate, actual["activated"], oracle
                    )
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
            del model
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        del converted, traced, module
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
