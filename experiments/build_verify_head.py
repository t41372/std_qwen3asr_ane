"""Compress an existing T16 compact vocabulary head so it matches a palettized bundle."""

import argparse
from pathlib import Path

from std_qwen3asr_ane.conversion.compress import compress_model, weight_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/probes/lm-head-compact-t16.mlpackage"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    counts = compress_model(
        args.source, args.output, "palette", args.bits, args.group_size
    )
    print({"counts": counts, "weight_bytes": weight_bytes(args.output)})


if __name__ == "__main__":
    main()
