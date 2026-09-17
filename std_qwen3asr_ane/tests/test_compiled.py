"""Host compilation shares identical weights when it can and never loses them."""

import errno
import json
import os
import shutil
from pathlib import Path

import pytest

from std_qwen3asr_ane import compiled
from std_qwen3asr_ane.compiled import compile_bundle

WEIGHTS = b"immutable weights" * 64


def _source(root: Path) -> Path:
    source = root / "source"
    package = source / "frontend.mlpackage" / "Data" / "com.apple.CoreML"
    (package / "weights").mkdir(parents=True)
    (package / "weights" / "weight.bin").write_bytes(WEIGHTS)
    (package / "model.mlmodel").write_bytes(b"spec")
    (source / "tokenizer.json").write_text("{}")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {"frontend": "frontend.mlpackage", "tokenizer": "tokenizer.json"},
                "decoder_partitions": [],
            }
        )
    )
    return source


@pytest.fixture
def fake_compiler(monkeypatch: pytest.MonkeyPatch):
    ct = pytest.importorskip("coremltools")

    def compile_model(path, destination_path):
        destination = Path(destination_path)
        (destination / "weights").mkdir(parents=True)
        shutil.copyfile(
            Path(path) / "Data" / "com.apple.CoreML" / "weights" / "weight.bin",
            destination / "weights" / "weight.bin",
        )
        (destination / "model.mil").write_text("mil")

    monkeypatch.setattr(ct.models.utils, "compile_model", compile_model)


def test_identical_weights_are_shared_by_hard_link(tmp_path: Path, fake_compiler):
    source, output = _source(tmp_path), tmp_path / "compiled"
    compile_bundle(source, output)
    binary = output / "frontend.mlmodelc" / "weights" / "weight.bin"
    original = (
        source / "frontend.mlpackage" / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    )
    assert os.path.samefile(binary, original)
    record = json.loads((output / "manifest.json").read_text())["compiled_from"]["models"][0]
    assert record["shared_immutable_weights"] == ["weights/weight.bin"]


def test_compiled_weights_survive_when_linking_fails(
    tmp_path: Path, fake_compiler, monkeypatch: pytest.MonkeyPatch
):
    source, output = _source(tmp_path), tmp_path / "compiled"

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(compiled.os, "link", cross_device)
    compile_bundle(source, output)
    binary = output / "frontend.mlmodelc" / "weights" / "weight.bin"
    assert binary.read_bytes() == WEIGHTS
    assert not binary.with_name("weight.bin.shared").exists()
    record = json.loads((output / "manifest.json").read_text())["compiled_from"]["models"][0]
    assert record["shared_immutable_weights"] == []
