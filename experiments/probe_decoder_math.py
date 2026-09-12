"""Localize first-token ANE error without KV state or attention ambiguity."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    load_partition,
)
from torch import nn
from torch.nn import functional as F


class FirstToken(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        normalized = self.layer.input_layernorm(hidden)
        value = self.layer.v_proj(normalized)
        expanded = value.reshape(1, self.layer.kv_heads, self.layer.head_dim, 1)
        expanded = expanded.repeat_interleave(
            self.layer.heads // self.layer.kv_heads, dim=1
        )
        attended = self.layer.o_proj(
            expanded.reshape(1, self.layer.heads * self.layer.head_dim, 1, 1)
        )
        residual = hidden + attended
        postnorm = self.layer.post_attention_layernorm(residual)
        gate = self.layer.gate_proj(postnorm)
        up = self.layer.up_proj(postnorm)
        product = F.silu(gate) * up
        down = self.layer.down_proj(product)
        output = residual + down
        return (
            normalized,
            value,
            attended,
            residual,
            postnorm,
            gate,
            up,
            product,
            down,
            output,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/source/Qwen3-ASR-1.7B"
    folder = root / "artifacts/probes/decoder-math"
    folder.mkdir(exist_ok=True)
    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    weights = SourceWeights(source)
    torch.set_num_threads(4)
    partition = DecoderPartition(config, 1, 128).eval()
    load_partition(partition, weights, 0)
    module = FirstToken(partition.layers[0]).eval()
    names = [
        "normalized",
        "value",
        "attended",
        "residual",
        "postnorm",
        "gate",
        "up",
        "product",
        "down",
        "output",
    ]
    path = folder / "first-token.mlpackage"
    sample = torch.zeros(1, 2048, 1, 1)
    if not path.exists():
        converted = ct.convert(
            torch.jit.trace(module, sample),
            inputs=[ct.TensorType(name="hidden", shape=sample.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name=name, dtype=np.float16) for name in names],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            skip_model_load=True,
        )
        converted.save(str(path))
    if args.export_only:
        return
    embeddings = weights.get("thinker.model.embed_tokens.weight")
    rows = []
    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
        try:
            model = ct.models.MLModel(str(path), compute_units=units)
            for token in [151644, 198, 872, 8948]:
                hidden = embeddings[token].reshape(1, 2048, 1, 1)
                with torch.inference_mode():
                    expected = module(hidden)
                actual = model.predict({"hidden": hidden.numpy().astype(np.float16)})
                for name, value in zip(names, expected, strict=True):
                    reference = value.numpy()
                    got = actual[name].astype(np.float32)
                    rows.append(
                        {
                            "compute_units": units.name,
                            "token": token,
                            "stage": name,
                            "reference_max": float(np.max(np.abs(reference))),
                            "relative_l2": float(
                                np.linalg.norm(reference - got)
                                / max(np.linalg.norm(reference), 1e-20)
                            ),
                            "max_error": float(np.max(np.abs(reference - got))),
                            "reference_subnormal_fraction": float(
                                np.mean((np.abs(reference) < 2**-14) & (reference != 0))
                            ),
                            "output_zero_fraction": float(np.mean(got == 0)),
                        }
                    )
        except Exception as error:  # noqa: BLE001 - persist native backend failures per device
            rows.append({"compute_units": units.name, "error": str(error)})
    (folder / "numerics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
