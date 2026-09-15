"""Memory-mapped embedding storage with row-only INT8 reconstruction."""

from pathlib import Path

import numpy as np


class Int8EmbeddingTable:
    """Array-like row access without expanding the full table at load time."""

    ndim = 2

    def __init__(self, values_path: Path, scales_path: Path, *, shape: tuple[int, int]):
        self._values = np.load(values_path, mmap_mode="r", allow_pickle=False)
        self._scales = np.load(scales_path, mmap_mode="r", allow_pickle=False)
        if (
            self._values.ndim != 2
            or self._values.dtype != np.int8
            or self._values.shape != shape
            or min(shape) < 1
            or self._scales.dtype != np.float32
            or self._scales.shape != (shape[0],)
            or not np.isfinite(self._scales).all()
            or not (self._scales > 0).all()
        ):
            raise ValueError("Invalid per-row INT8 embedding values or scales")
        self.shape = self._values.shape

    def __getitem__(self, rows):
        return self._values[rows].astype(np.float32) * np.expand_dims(self._scales[rows], -1)


def embedding_quantization(manifest: dict) -> dict | None:
    """Validate the optional storage contract without opening any array files."""
    descriptor = manifest.get("embedding_quantization")
    if descriptor is None:
        return None
    if manifest.get("schema_version") != 3:
        raise ValueError("INT8 embeddings require bundle schema 3")
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("scheme") != "symmetric_int8_per_row"
        or type(descriptor.get("axis")) is not int
        or descriptor["axis"] != 0
        or descriptor.get("scale_dtype") != "float32"
    ):
        raise ValueError("Unsupported embedding quantization descriptor")
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError("Embedding shape must contain vocabulary and hidden width")
    return descriptor
