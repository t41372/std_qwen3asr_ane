"""Build a real-weight decoder layer before spending time on the entire model."""

import argparse
import json
from pathlib import Path

import torch
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    convert_partition,
    load_partition,
)

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--precision", choices=("fp16", "mixed"), default="fp16")
parser.add_argument("--tag", default="")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / "artifacts/source/Qwen3-ASR-1.7B"
suffix = "" if args.precision == "fp16" else "-mixed"
suffix += f"-{args.tag}" if args.tag else ""
output = (
    root / f"artifacts/probes/decoder-layer-0-scale{args.scale:g}{suffix}.mlpackage"
)
output.parent.mkdir(parents=True, exist_ok=True)
config = json.loads((source / "config.json").read_text())["thinker_config"][
    "text_config"
]
torch.set_num_threads(4)
module = DecoderPartition(config, 1, 128, residual_scale=args.scale)
load_partition(module, SourceWeights(source), 0)
convert_partition(module, config, output, 128, precision=args.precision)
print(output)
