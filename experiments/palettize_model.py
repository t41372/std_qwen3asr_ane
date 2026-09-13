"""Palettize projection weights with deterministic histogram-weighted Lloyd steps.

FP16 has at most 65536 bit patterns. Histogramming those patterns avoids running
clustering over millions of repeated BF16-origin weights. The objective is still
the full element-weighted squared error, rather than a random subsample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np


def histogram_palette(
    weight: np.ndarray, *, bits: int
) -> tuple[np.ndarray, np.ndarray]:
    if weight.dtype != np.float16 or bits not in (4, 6, 8):
        raise ValueError("This experiment requires FP16 weights and 4, 6 or 8 bits")
    patterns = np.ascontiguousarray(weight).view(np.uint16).reshape(-1)
    histogram = np.bincount(patterns, minlength=65536)
    present = np.flatnonzero(histogram).astype(np.uint16)
    values = present.view(np.float16).astype(np.float64)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError("Weights must be finite and nonempty")
    order = np.argsort(values, kind="stable")
    values, present = values[order], present[order]
    counts = histogram[present].astype(np.float64)
    size = 1 << bits
    if values.size <= size:
        centers = np.pad(values, (0, size - values.size), mode="edge")
    else:
        # Mix data quantiles with the full range so rare tails have centroids
        # from the first step. Lloyd assignments use every histogram count.
        cumulative = np.cumsum(counts)
        quantiles = np.interp(np.linspace(0, cumulative[-1], size), cumulative, values)
        centers = (quantiles + np.linspace(values[0], values[-1], size)) / 2
        for _ in range(100):
            indices = np.searchsorted((centers[:-1] + centers[1:]) / 2, values)
            mass = np.bincount(indices, weights=counts, minlength=size)
            totals = np.bincount(indices, weights=counts * values, minlength=size)
            updated = centers.copy()
            np.divide(totals, mass, out=updated, where=mass != 0)
            updated.sort()
            if np.array_equal(updated, centers):
                break
            centers = updated
    # Quantize centroids to the actual serialized LUT precision before final
    # nearest-neighbor assignment. Tie breaking chooses the lower centroid.
    lut = centers.astype(np.float16)
    boundaries = (lut[:-1].astype(np.float64) + lut[1:].astype(np.float64)) / 2
    mapping = np.zeros(65536, dtype=np.uint8)
    mapping[present] = np.searchsorted(boundaries, values).astype(np.uint8)
    return lut, mapping[patterns]


def main():
    import coremltools as ct
    import coremltools.optimize.coreml as optimization
    from std_qwen3asr_ane.conversion.passes import verify_activation_operators

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), required=True)
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = ct.models.MLModel(
        str(args.model), skip_model_load=True, compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    config = optimization.OptimizationConfig(
        op_type_configs={
            "conv": optimization.OpPalettizerConfig(
                mode="custom",
                lut_function=partial(histogram_palette, bits=args.bits),
                granularity="per_grouped_channel" if args.group_size else "per_tensor",
                group_size=args.group_size or 32,
                weight_threshold=4096,
            )
        }
    )
    started = perf_counter()
    compressed = optimization.palettize_weights(model, config=config)
    verify_activation_operators(compressed)
    compressed.save(str(args.output))
    report = {
        "source": str(args.model),
        "output": str(args.output),
        "bits": args.bits,
        "group_size": args.group_size,
        "seconds": perf_counter() - started,
        "palette_algorithm": "full_fp16_histogram_weighted_lloyd_v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
