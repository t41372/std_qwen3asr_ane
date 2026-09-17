"""INT8 storage must reconstruct only selected rows, with owned FP32 outputs."""

import json

import numpy as np
import pytest

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.embedding import write_int8_embedding
from std_qwen3asr_ane.embedding import Int8EmbeddingTable, embedding_quantization


def test_int8_scalar_and_prompt_gather_return_owned_values(tmp_path):
    values = np.array([[1, -2, 0], [-127, 127, 7], [0, 0, 0]], dtype=np.int8)
    scales = np.array([0.1, 0.02, 1], dtype=np.float32)
    np.save(tmp_path / "values.npy", values)
    np.save(tmp_path / "scales.npy", scales)
    table = Int8EmbeddingTable(tmp_path / "values.npy", tmp_path / "scales.npy", shape=values.shape)
    expected = values.astype(np.float32) * scales[:, None]
    np.testing.assert_array_equal(table[1], expected[1])
    gathered = table[[2, 0, 1, 0]]
    np.testing.assert_array_equal(gathered, expected[[2, 0, 1, 0]])
    assert gathered.dtype == np.float32 and gathered.flags.owndata
    assert isinstance(table._values, np.memmap) and table._values.dtype == np.int8
    gathered.fill(99)
    np.testing.assert_array_equal(table[0], expected[0])


@pytest.mark.parametrize(
    "scales",
    [
        np.array([0, 1], np.float32),
        np.array([np.nan, 1], np.float32),
        np.array([1, 1], np.float16),
        np.array([1], np.float32),
    ],
)
def test_invalid_scales_fail_before_inference(tmp_path, scales):
    np.save(tmp_path / "values.npy", np.ones((2, 3), np.int8))
    np.save(tmp_path / "scales.npy", scales)
    with pytest.raises(ValueError, match="Invalid per-row"):
        Int8EmbeddingTable(tmp_path / "values.npy", tmp_path / "scales.npy", shape=(2, 3))


def test_quantization_contract_rejects_legacy_or_malformed_metadata():
    descriptor = {
        "scheme": "symmetric_int8_per_row",
        "axis": 0,
        "scale_dtype": "float32",
        "shape": [10, 8],
    }
    assert embedding_quantization({"schema_version": 1}) is None
    assert (
        embedding_quantization({"schema_version": 3, "embedding_quantization": descriptor})
        == descriptor
    )
    with pytest.raises(ValueError, match="schema 3"):
        embedding_quantization({"schema_version": 1, "embedding_quantization": descriptor})
    for field, value in (("axis", True), ("shape", [True, 8]), ("scale_dtype", "float16")):
        invalid = json.loads(json.dumps(descriptor))
        invalid[field] = value
        with pytest.raises(ValueError):
            embedding_quantization({"schema_version": 3, "embedding_quantization": invalid})


def test_export_preserves_source_and_handles_zero_rows(tmp_path):
    original = np.array([[0, 0, 0], [-0.2, 0.01, 0.3]], dtype=np.float16)
    source, destination = tmp_path / "source.npy", tmp_path / "quantized"
    np.save(source, original)
    before = digest(source)
    metadata = write_int8_embedding(source, destination)
    table = Int8EmbeddingTable(
        destination / metadata["files"]["embedding"],
        destination / metadata["files"]["embedding_scales"],
        shape=original.shape,
    )
    np.testing.assert_array_equal(table[0], original[0])
    error = np.abs(table[:] - original.astype(np.float32))
    assert float(error.max()) == metadata["max_abs_error"]
    assert float(error.sum(dtype=np.float64) / error.size) == metadata["mean_abs_error"]
    assert digest(source) == before == metadata["source_sha256"]
    with pytest.raises(FileExistsError):
        write_int8_embedding(source, destination)
