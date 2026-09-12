"""Compare real-weight stateful Core ML decoding with official PyTorch layers."""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
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
    load_partition,
)
from transformers.cache_utils import DynamicCache

parser = argparse.ArgumentParser()
parser.add_argument(
    "--compute-units", choices=["cpu_only", "cpu_and_ne"], default="cpu_and_ne"
)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--precision", choices=("fp16", "mixed"), default="fp16")
parser.add_argument("--tag", default="")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / "artifacts/source/Qwen3-ASR-1.7B"
config = json.loads((source / "config.json").read_text())["thinker_config"][
    "text_config"
]
weights = SourceWeights(source)
torch.set_num_threads(4)
torch.manual_seed(42)
official_config = Qwen3ASRTextConfig(**config)
official_config._attn_implementation = "eager"
official = Qwen3ASRThinkerTextDecoderLayer(official_config, 0).eval()
official.load_state_dict(
    {
        name: weights.get("thinker.model.layers.0." + name)
        for name in official.state_dict()
    }
)
partition = DecoderPartition(config, 1, 128, residual_scale=args.scale).eval()
load_partition(partition, weights, 0)
units = (
    ct.ComputeUnit.CPU_ONLY
    if args.compute_units == "cpu_only"
    else ct.ComputeUnit.CPU_AND_NE
)
suffix = "" if args.precision == "fp16" else "-mixed"
suffix += f"-{args.tag}" if args.tag else ""
start = time.perf_counter()
model = ct.models.MLModel(
    str(
        root / f"artifacts/probes/decoder-layer-0-scale{args.scale:g}{suffix}.mlpackage"
    ),
    compute_units=units,
)
load_seconds = time.perf_counter() - start
state = model.make_state()
cache = DynamicCache()
rows = []
embedding = weights.get("thinker.model.embed_tokens.weight")
with torch.inference_mode():
    for position, token in enumerate(
        [151644, 8948, 198, 151645, 198, 151644, 872, 198]
    ):
        hidden = embedding[token].reshape(1, 1, -1)
        frequencies = position / config["rope_theta"] ** (torch.arange(64).float() / 64)
        cosine = frequencies.cos().reshape(1, 64, 1, 1)
        sine = frequencies.sin().reshape(1, 64, 1, 1)
        full_cosine = torch.cat((frequencies.cos(), frequencies.cos())).reshape(
            1, 1, 128
        )
        full_sine = torch.cat((frequencies.sin(), frequencies.sin())).reshape(1, 1, 128)
        expected = official(hidden, (full_cosine, full_sine), past_key_values=cache)
        mask = torch.full((1, 1, 1, 128), -10000.0)
        mask[..., : position + 1] = 0
        update = torch.zeros_like(mask)
        update[..., position] = 1
        scaled = hidden.transpose(1, 2).unsqueeze(2) / args.scale
        tensors = (scaled, cosine, sine, mask, update)
        actual_torch = partition(*tensors).squeeze(2).transpose(1, 2) * args.scale
        inputs = dict(
            zip(
                ("hidden_states", "cosine", "sine", "attention_mask", "update_mask"),
                (
                    value.numpy().astype(
                        np.float32 if args.precision == "mixed" else np.float16
                    )
                    for value in tensors
                ),
            )
        )
        start = time.perf_counter()
        actual = (
            model.predict(inputs, state=state)["output_hidden_states"].reshape(1, 1, -1)
            * args.scale
        )
        seconds = time.perf_counter() - start
        reference = expected.numpy()
        rows.append(
            {
                "position": position,
                "seconds": seconds,
                "finite": bool(np.isfinite(actual).all()),
                "pytorch_max_absolute_error": float(
                    np.max(np.abs(actual_torch.numpy() - reference))
                ),
                "coreml_max_absolute_error": float(np.max(np.abs(actual - reference))),
                "coreml_relative_l2_error": float(
                    np.linalg.norm(actual - reference) / np.linalg.norm(reference)
                ),
            }
        )
report = {
    "compute_units": args.compute_units,
    "load_seconds": load_seconds,
    "tokens": rows,
}
destination = (
    root
    / f"artifacts/probes/decoder-layer-0-scale{args.scale:g}{suffix}-numerics-{args.compute_units}.json"
)
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(report, indent=2, allow_nan=False))
