"""Deterministic per-row INT8 embedding export for an unpublished bundle."""

from pathlib import Path

import numpy as np

from ..bundle import digest


def open_fp16_embedding(source: Path) -> np.ndarray:
    """Memory-map the FP16 table an INT8 export starts from; reject other layouts."""
    original = np.load(source, mmap_mode="r", allow_pickle=False)
    if original.ndim != 2 or original.dtype != np.float16 or min(original.shape) < 1:
        raise ValueError("Expected a nonempty FP16 embedding matrix")
    return original


def write_int8_embedding(source: Path, destination: Path) -> dict:
    """Write new mmap-compatible arrays; never overwrite the source or old output."""
    destination.mkdir(parents=True, exist_ok=True)
    values_path = destination / "embedding.int8.npy"
    scales_path = destination / "embedding.scale.fp32.npy"
    if values_path.exists() or scales_path.exists():
        raise FileExistsError("Use a new destination for quantized embedding arrays")
    original = open_fp16_embedding(source)
    values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.int8, shape=original.shape)
    scales = np.lib.format.open_memmap(
        scales_path, mode="w+", dtype=np.float32, shape=(original.shape[0],)
    )
    max_error, total_error = 0.0, 0.0
    for start in range(0, len(original), 1024):
        block = np.asarray(original[start : start + 1024], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Non-finite source embedding")
        scale = np.max(np.abs(block), axis=1) / 127
        scale[scale == 0] = 1
        quantized = np.clip(np.rint(block / scale[:, None]), -127, 127).astype(np.int8)
        values[start : start + len(block)] = quantized
        scales[start : start + len(block)] = scale
        error = np.abs(quantized.astype(np.float32) * scale[:, None] - block)
        max_error = max(max_error, float(error.max()))
        total_error += float(error.sum(dtype=np.float64))
    values.flush()
    scales.flush()
    return {
        "scheme": "symmetric_int8_per_row",
        "axis": 0,
        "scale_dtype": "float32",
        "shape": list(original.shape),
        "quantizer_version": "max_abs_rint_int8_v1",
        "source_sha256": digest(source),
        "max_abs_error": max_error,
        "mean_abs_error": total_error / original.size,
        "files": {"embedding": values_path.name, "embedding_scales": scales_path.name},
        "payload_sha256": {
            values_path.name: digest(values_path),
            scales_path.name: digest(scales_path),
        },
        "bytes_saved": source.stat().st_size
        - values_path.stat().st_size
        - scales_path.stat().st_size,
    }
