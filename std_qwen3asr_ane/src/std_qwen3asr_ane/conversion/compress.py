"""Weight-only compression of a built bundle into a new, immutable bundle.

Only the decoder partitions and the vocabulary head are compressed: they hold
all but a few percent of the bytes read per generated token. The audio graphs
stay FP16. Activations, KV caches, normalization and the numerically verified
activation expressions are unchanged, so the runtime contract is identical.

Two schemes are offered. Palettization (a per-group lookup table) is the
compression the ANE decompresses natively on macOS 15+. Linear per-block
quantization matches the affine scheme used by common GPU runtimes. Neither
scheme is a quality claim; compressed bundles start as ``unvalidated``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

COMPRESSIBLE_ROLES = ("decoder", "lm_head")
PALETTE_BITS = (4, 6, 8)
LINEAR_BITS = (4, 8)
ALGORITHM = "full_fp16_histogram_weighted_lloyd_v1"


def histogram_palette(weight: np.ndarray, *, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic histogram-weighted Lloyd palettization of one FP16 group.

    FP16 has at most 65536 bit patterns, so the histogram makes the objective the
    exact element-weighted squared error over every weight, with no subsampling.
    Centroids are rounded to the serialized FP16 LUT precision before the final
    nearest-centroid assignment; ties choose the lower centroid.
    """
    if weight.dtype != np.float16 or bits not in PALETTE_BITS:
        raise ValueError("Palettization requires FP16 weights and 4, 6 or 8 bits")
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
        # Seed with a blend of data quantiles and the full range so rare tails
        # keep a centroid from the first Lloyd step.
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
    lut = centers.astype(np.float16)
    boundaries = (lut[:-1].astype(np.float64) + lut[1:].astype(np.float64)) / 2
    mapping = np.zeros(65536, dtype=np.uint8)
    mapping[present] = np.searchsorted(boundaries, values).astype(np.uint8)
    return lut, mapping[patterns]


def compression_config(scheme: str, bits: int, group_size: int):
    """Build the coremltools optimization config for 1x1 convolution weights."""
    import coremltools.optimize.coreml as optimization

    if group_size < 1:
        raise ValueError("group_size must be positive")
    if scheme == "palette":
        if bits not in PALETTE_BITS:
            raise ValueError(f"Palettization supports {PALETTE_BITS} bits")
        config = optimization.OpPalettizerConfig(
            mode="custom",
            lut_function=partial(histogram_palette, bits=bits),
            granularity="per_grouped_channel",
            group_size=group_size,
            weight_threshold=4096,
        )
    elif scheme == "linear":
        if bits not in LINEAR_BITS:
            raise ValueError(f"Linear quantization supports {LINEAR_BITS} bits")
        config = optimization.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=f"int{bits}",
            granularity="per_block",
            block_size=group_size,
            weight_threshold=4096,
        )
    else:
        raise ValueError("scheme must be 'palette' or 'linear'")
    return optimization.OptimizationConfig(op_type_configs={"conv": config})


def compress_model(source: Path, destination: Path, scheme: str, bits: int, group_size: int):
    import coremltools as ct
    import coremltools.optimize.coreml as optimization

    from .passes import verify_activation_operators

    model = ct.models.MLModel(str(source), skip_model_load=True)
    config = compression_config(scheme, bits, group_size)
    if scheme == "palette":
        compressed = optimization.palettize_weights(model, config=config)
    else:
        compressed = optimization.linear_quantize_weights(model, config=config)
    verify_activation_operators(compressed)
    compressed.save(str(destination))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def weight_bytes(package: Path) -> int:
    return sum(path.stat().st_size for path in package.rglob("weight.bin"))


def compress_bundle(
    source: Path,
    output: Path,
    *,
    scheme: str = "palette",
    bits: int = 8,
    group_size: int = 32,
    roles: tuple[str, ...] = COMPRESSIBLE_ROLES,
) -> dict:
    """Write a compressed copy of an uncompiled bundle; the source is never modified."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new directory to preserve previous artifacts")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or "weight_compression" in manifest:
        raise ValueError("Compression requires an uncompressed schema-1 bundle")
    if not set(roles) <= set(COMPRESSIBLE_ROLES) or not roles:
        raise ValueError(f"roles must be a nonempty subset of {COMPRESSIBLE_ROLES}")
    compression_config(scheme, bits, group_size)
    partitions = manifest["decoder_partitions"]
    targets = set(partitions) if "decoder" in roles else set()
    if "lm_head" in roles:
        targets.add(manifest["files"]["lm_head"])
    everything = set(manifest["files"].values()) | set(partitions)
    for relative in everything:
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or path == source or not path.exists():
            raise ValueError(f"Invalid source artifact: {relative}")
        if relative in targets and path.suffix != ".mlpackage":
            raise ValueError("Compress the uncompiled .mlpackage bundle, then compile it")
    output.mkdir(parents=True)
    records = []
    for relative in sorted(everything):
        path, destination = source / relative, output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative not in targets:
            subprocess.run(["/bin/cp", "-cR", str(path), str(destination)], check=True)
            continue
        started = perf_counter()
        compress_model(path, destination, scheme, bits, group_size)
        record = {
            "file": relative,
            "seconds": perf_counter() - started,
            "source_weight_bytes": weight_bytes(path),
            "compressed_weight_bytes": weight_bytes(destination),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest["weight_compression"] = {
        "scheme": scheme,
        "bits": bits,
        "group_size": group_size,
        "roles": list(roles),
        "algorithm": ALGORITHM if scheme == "palette" else "coremltools_linear_symmetric_per_block",
        "compressed_files": records,
        "parent_manifest_sha256": digest(source / "manifest.json"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest["validation_status"] = "unvalidated"
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest
