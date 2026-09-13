"""Weight compression must be deterministic, bounded and leave other assets untouched."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from std_qwen3asr_ane.conversion.compress import (
    compress_bundle,
    compression_config,
    histogram_palette,
    weight_bytes,
)


def test_histogram_palette_reconstructs_small_alphabets_exactly():
    values = np.array([-1.5, 0.25, 0.25, 3.0, -1.5, 0.0], dtype=np.float16)
    lut, indices = histogram_palette(values, bits=4)
    assert lut.dtype == np.float16 and lut.shape == (16,)
    assert indices.dtype == np.uint8 and indices.shape == values.shape
    np.testing.assert_array_equal(lut[indices], values)


def test_histogram_palette_is_deterministic_and_beats_uniform_grid():
    generator = np.random.default_rng(20260913)
    weight = generator.standard_normal((32, 2048)).astype(np.float16)
    lut, indices = histogram_palette(weight, bits=4)
    lut_again, indices_again = histogram_palette(weight, bits=4)
    np.testing.assert_array_equal(lut, lut_again)
    np.testing.assert_array_equal(indices, indices_again)
    assert np.all(np.diff(lut.astype(np.float64)) >= 0)
    assert indices.shape == (weight.size,), "coremltools expects flattened indices"
    restored = lut[indices].reshape(weight.shape).astype(np.float64)
    error = np.mean((restored - weight.astype(np.float64)) ** 2)
    grid = np.linspace(weight.min(), weight.max(), 16)
    uniform = grid[np.abs(weight[..., None].astype(np.float64) - grid).argmin(-1)]
    uniform_error = np.mean((uniform - weight.astype(np.float64)) ** 2)
    assert error < uniform_error


def test_histogram_palette_rejects_unsupported_inputs():
    with pytest.raises(ValueError):
        histogram_palette(np.ones(8, dtype=np.float32), bits=8)
    with pytest.raises(ValueError):
        histogram_palette(np.ones(8, dtype=np.float16), bits=3)
    with pytest.raises(ValueError):
        histogram_palette(np.array([np.inf], dtype=np.float16), bits=8)


def test_compression_config_validates_scheme_and_bits():
    pytest.importorskip("coremltools")
    compression_config("palette", 8, 32)
    compression_config("linear", 4, 32)
    with pytest.raises(ValueError):
        compression_config("linear", 6, 32)
    with pytest.raises(ValueError):
        compression_config("palette", 8, 0)
    with pytest.raises(ValueError):
        compression_config("other", 8, 32)


def _toy_package(path: Path, seed: int) -> None:
    ct = pytest.importorskip("coremltools")
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    module = torch.nn.Conv2d(64, 128, 1, bias=False).eval()
    example = torch.zeros(1, 64, 1, 1)
    model = ct.convert(
        torch.jit.trace(module, example),
        inputs=[ct.TensorType(name="x", shape=tuple(example.shape), dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    model.save(str(path))


def _digest_tree(path: Path) -> dict[str, str]:
    return {
        str(child.relative_to(path)): hashlib.sha256(child.read_bytes()).hexdigest()
        for child in sorted(path.rglob("*"))
        if child.is_file()
    }


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for name, seed in (
        ("frontend.mlpackage", 1),
        ("encoder.mlpackage", 2),
        ("decoder_00.mlpackage", 3),
        ("decoder_04.mlpackage", 4),
        ("lm_head.mlpackage", 5),
    ):
        _toy_package(source / name, seed)
    np.save(source / "embedding.npy", np.zeros((4, 8), dtype=np.float16))
    np.save(source / "mel_filters.npy", np.ones((201, 128), dtype=np.float32))
    (source / "tokenizer.json").write_text("{}")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "Qwen/Qwen3-ASR-1.7B",
                "files": {
                    "frontend": "frontend.mlpackage",
                    "encoder": "encoder.mlpackage",
                    "decoder": "decoder_00.mlpackage",
                    "decoder_04": "decoder_04.mlpackage",
                    "lm_head": "lm_head.mlpackage",
                    "embedding": "embedding.npy",
                    "mel_filters": "mel_filters.npy",
                    "tokenizer": "tokenizer.json",
                },
                "decoder_partitions": ["decoder_00.mlpackage", "decoder_04.mlpackage"],
                "validation_status": "validated_elsewhere",
            }
        )
    )
    return source


def test_compress_bundle_compresses_only_decoder_and_head(bundle: Path, tmp_path: Path):
    ct = pytest.importorskip("coremltools")
    output = tmp_path / "compressed"
    before = _digest_tree(bundle)
    manifest = compress_bundle(bundle, output, scheme="palette", bits=8, group_size=32)
    assert _digest_tree(bundle) == before, "the source bundle must stay immutable"
    for name in ("frontend.mlpackage", "encoder.mlpackage"):
        assert _digest_tree(output / name) == _digest_tree(bundle / name)
    for name in ("embedding.npy", "mel_filters.npy", "tokenizer.json"):
        assert (output / name).read_bytes() == (bundle / name).read_bytes()
    for name in ("decoder_00.mlpackage", "decoder_04.mlpackage", "lm_head.mlpackage"):
        assert weight_bytes(output / name) < weight_bytes(bundle / name)
    compression = manifest["weight_compression"]
    assert compression["bits"] == 8 and compression["scheme"] == "palette"
    assert {record["file"] for record in compression["compressed_files"]} == {
        "decoder_00.mlpackage",
        "decoder_04.mlpackage",
        "lm_head.mlpackage",
    }
    assert manifest["validation_status"] == "unvalidated"
    written = json.loads((output / "manifest.json").read_text())
    assert (
        written["weight_compression"]["parent_manifest_sha256"]
        == hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    )
    original = ct.models.MLModel(
        str(bundle / "lm_head.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    compressed = ct.models.MLModel(
        str(output / "lm_head.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    x = np.random.default_rng(0).standard_normal((1, 64, 1, 1)).astype(np.float32)
    reference = original.predict({"x": x})["y"]
    candidate = compressed.predict({"x": x})["y"]
    assert np.abs(reference - candidate).max() < 0.05 * np.abs(reference).max()


def test_compress_bundle_refuses_unsafe_or_repeated_inputs(bundle: Path, tmp_path: Path):
    pytest.importorskip("coremltools")
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        compress_bundle(bundle, output, roles=("encoder",))
    with pytest.raises(ValueError):
        compress_bundle(bundle, output, scheme="linear", bits=6)
    assert not output.exists()
    compress_bundle(bundle, output, roles=("lm_head",))
    with pytest.raises(FileExistsError):
        compress_bundle(bundle, output)
    with pytest.raises(ValueError, match="uncompressed"):
        compress_bundle(output, tmp_path / "twice")
    (bundle / "decoder_00.mlmodelc").mkdir()
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["decoder_partitions"][0] = "decoder_00.mlmodelc"
    manifest["files"]["decoder"] = "decoder_00.mlmodelc"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="uncompiled"):
        compress_bundle(bundle, tmp_path / "compiled-source")
