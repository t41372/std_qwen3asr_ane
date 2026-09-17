"""Bundle assembly binds every reused graph to the checkpoint the manifest names."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"


def _source(root: Path, revision: str) -> Path:
    source = root / "source"
    source.mkdir(exist_ok=True)
    (source / "config.json").write_text("{}")
    (source / "chat_template.json").write_text(json.dumps({"chat_template": "x"}))
    (source / "source.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": revision}))
    return source


@pytest.fixture
def fake_conversion(monkeypatch: pytest.MonkeyPatch):
    """Stub the graph converters and tokenizer tooling; only bundle assembly runs."""
    transformers = pytest.importorskip("transformers")
    from std_qwen3asr_ane.conversion import decoder, encoder

    calls = {"encoder": 0, "decoder_fails": False}

    def build_encoder(source, output, *, frontend_batch_size=1):
        calls["encoder"] += 1
        return {
            "files": [{"role": "frontend", "path": "frontend.mlpackage"}],
            "frontend": {"chunk_frames": 100},
            "encoder": {"window_tokens": 104},
        }

    def build_decoder(source, output, **settings):
        if calls["decoder_fails"]:
            raise RuntimeError("decoder conversion interrupted")
        return {
            "files": {"lm_head": "lm_head.mlpackage"},
            "decoder_partitions": ["decoder_00.mlpackage"],
        }

    class Tokenizer:
        backend_tokenizer = SimpleNamespace(save=lambda path: Path(path).write_text("{}"))

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Tokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return "<prompt>"

    class Extractor:
        mel_filters = np.zeros((2, 2), np.float32)

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Extractor()

    monkeypatch.setattr(encoder, "build_encoder", build_encoder)
    monkeypatch.setattr(decoder, "build_decoder", build_decoder)
    monkeypatch.setattr(transformers, "AutoTokenizer", Tokenizer)
    monkeypatch.setattr(transformers, "WhisperFeatureExtractor", Extractor)
    return calls


def test_reused_encoder_must_come_from_the_same_checkpoint(tmp_path: Path, fake_conversion):
    from std_qwen3asr_ane.conversion.build import build_bundle

    source, output = _source(tmp_path, "a" * 40), tmp_path / "bundle"
    fake_conversion["decoder_fails"] = True
    with pytest.raises(RuntimeError, match="interrupted"):
        build_bundle(source, output)
    assert json.loads((output / "encoder-manifest.json").read_text())["source"] == {
        "model_id": MODEL_ID,
        "revision": "a" * 40,
    }
    assert not (output / "manifest.json").exists()
    fake_conversion["decoder_fails"] = False
    # An architecture-compatible revision must not inherit the other checkpoint's encoder.
    _source(tmp_path, "b" * 40)
    with pytest.raises(ValueError, match="not built from this checkpoint"):
        build_bundle(source, output, reuse_encoder=True)
    assert not (output / "manifest.json").exists()
    _source(tmp_path, "a" * 40)
    with pytest.raises(ValueError, match="frontend batch"):
        build_bundle(source, output, reuse_encoder=True, frontend_batch_size=4)
    manifest = build_bundle(source, output, reuse_encoder=True)
    assert manifest["source_revision"] == "a" * 40
    assert fake_conversion["encoder"] == 1


def test_reused_encoder_without_recorded_source_is_refused(tmp_path: Path, fake_conversion):
    from std_qwen3asr_ane.conversion.build import build_bundle

    source, output = _source(tmp_path, "a" * 40), tmp_path / "bundle"
    fake_conversion["decoder_fails"] = True
    with pytest.raises(RuntimeError):
        build_bundle(source, output)
    fake_conversion["decoder_fails"] = False
    metadata = output / "encoder-manifest.json"
    legacy = json.loads(metadata.read_text())
    del legacy["source"]
    metadata.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="not built from this checkpoint"):
        build_bundle(source, output, reuse_encoder=True)
