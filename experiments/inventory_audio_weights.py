"""Count and hash actual source audio tensors before making memory claims."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for file in sorted(args.source.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():  # noqa: SIM118 (safe_open is not iterable)
                if not key.startswith("thinker.audio_tower."):
                    continue
                tensor = tensors.get_tensor(key)
                role = (
                    "frontend"
                    if key.split("audio_tower.")[1].startswith(("conv2d", "conv_out"))
                    else "encoder"
                )
                rows.append(
                    {
                        "name": key,
                        "source_file": file.name,
                        "role": role,
                        "shape": list(tensor.shape),
                        "source_dtype": str(tensor.dtype),
                        "elements": tensor.numel(),
                        "source_bytes": tensor.numel() * tensor.element_size(),
                        "fp16_bytes": tensor.numel() * 2,
                        "large_matrix": tensor.ndim == 2 and tensor.numel() >= 4096,
                        "sha256": hashlib.sha256(
                            tensor.contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    }
                )
    report = {
        "source_revision": json.loads((args.source / "source.json").read_text())["revision"],
        "config_sha256": digest(args.source / "config.json"),
        "tensors": rows,
        "totals": {
            role: {
                field: sum(row[field] for row in rows if row["role"] == role)
                for field in ("elements", "source_bytes", "fp16_bytes")
            }
            for role in ("frontend", "encoder")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"]))


if __name__ == "__main__":
    main()
