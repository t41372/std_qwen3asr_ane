"""Quantize only host embedding rows, retaining immutable reconstruction evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    source = args.bundle / manifest["files"]["embedding"]
    original = np.load(source, mmap_mode="r", allow_pickle=False)
    if original.ndim != 2 or original.dtype != np.float16:
        raise ValueError("Expected the baseline FP16 embedding table")
    table_path = args.output / "embedding.int8.npy"
    scale_path = args.output / "embedding.scale.fp32.npy"
    table = np.lib.format.open_memmap(table_path, mode="w+", dtype=np.int8, shape=original.shape)
    scales = np.lib.format.open_memmap(
        scale_path, mode="w+", dtype=np.float32, shape=(original.shape[0],)
    )
    maximum, total = 0.0, 0.0
    for start in range(0, len(original), 1024):
        block = np.asarray(original[start : start + 1024], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Non-finite source embedding")
        scale = np.max(np.abs(block), axis=1) / 127
        scale[scale == 0] = 1
        quantized = np.clip(np.rint(block / scale[:, None]), -127, 127).astype(np.int8)
        table[start : start + len(block)] = quantized
        scales[start : start + len(block)] = scale
        error = np.abs(quantized.astype(np.float32) * scale[:, None] - block)
        maximum = max(maximum, float(error.max()))
        total += float(error.sum(dtype=np.float64))
    table.flush()
    scales.flush()
    del table, scales
    report = {
        "scheme": "symmetric_int8_per_row",
        "axis": 0,
        "scale_dtype": "float32",
        "shape": list(original.shape),
        "quantizer_version": "max_abs_rint_int8_v1",
        "source_sha256": digest(source),
        "parent_manifest_sha256": digest(args.bundle / "manifest.json"),
        "max_abs_error": maximum,
        "mean_abs_error": total / original.size,
        "files": {"embedding": table_path.name, "scales": scale_path.name},
        "payload_sha256": {
            table_path.name: digest(table_path),
            scale_path.name: digest(scale_path),
        },
        "bytes_saved": source.stat().st_size
        - table_path.stat().st_size
        - scale_path.stat().st_size,
        "builder_sha256": digest(Path(__file__)),
        "validation_status": "unvalidated",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
