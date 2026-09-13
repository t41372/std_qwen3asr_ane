"""Convert one real decoder partition with fixed or enumerated token widths."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    convert_partition,
    load_partition,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--enumerated", action="store_true")
    parser.add_argument("--grouped-attention", action="store_true")
    parser.add_argument("--fused-attention", action="store_true")
    parser.add_argument("--fused-projections", action="store_true")
    args = parser.parse_args()
    source = Path("artifacts/source/Qwen3-ASR-1.7B")
    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    started = perf_counter()
    module = DecoderPartition(
        config, args.layers, args.cache_length, token_batch_size=args.width
    )
    load_partition(module, SourceWeights(source), 0)
    if args.grouped_attention:
        for layer in module.layers:
            layer.enable_grouped_attention()
    if args.fused_attention:
        for layer in module.layers:
            layer.fused_attention = True
    if args.fused_projections:
        for layer in module.layers:
            layer.fuse_projections()
    convert_partition(
        module,
        config,
        args.output,
        args.cache_length,
        token_sizes=(1, args.width) if args.enumerated else None,
    )
    print(json.dumps({"model": str(args.output), "seconds": perf_counter() - started}))


if __name__ == "__main__":
    main()
