"""Host-only benchmark lifecycle tests; scoring and sample semantics stay unchanged."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import soundfile as sf
from pydantic import SecretStr
from standard_asr.engine import BaseConfig, secret_field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import evaluate


@pytest.fixture
def args(tmp_path: Path):
    sf.write(tmp_path / "audio.wav", np.zeros(1600, dtype=np.float32), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "one",
                "audio_path": "audio.wav",
                "reference": "hello",
                "language": "en",
            }
        )
        + "\n"
    )
    return argparse.Namespace(
        manifest=manifest,
        model_dir=tmp_path,
        output=tmp_path / "run.jsonl",
        backend="coreml",
        compute_units="cpu_and_ne",
        warmups=0,
        repeats=1,
        max_new_tokens=256,
        language_mode="auto",
        seed=42,
        torch_threads=0,
    )


def test_factory_exposes_coreml_cleanup_without_loading_real_models(args, monkeypatch):
    import std_qwen3asr_ane.runtime as module

    closed = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            return SimpleNamespace(text="hello", language="en", timings=None)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "CoreMLRuntime", Runtime)
    backend = evaluate.make_backend(args)
    assert backend.transcribe(np.zeros(1600), None) == ("hello", "en", None)
    backend.close()
    assert closed == [True]


@pytest.mark.parametrize("close_fails", [False, True])
def test_close_is_recorded_without_relabeling_quality_or_latency(args, monkeypatch, close_fails):
    calls = []

    def transcribe(audio, language):
        calls.append("transcribe")
        return "hello", "en", None

    def close():
        calls.append("close")
        if close_fails:
            raise RuntimeError("native still holds input buffers")

    monkeypatch.setattr(evaluate, "make_backend", lambda args: evaluate.Backend(transcribe, close))
    code = evaluate.run(args)
    assert code == int(close_fails)
    assert calls == ["transcribe", "close"]
    record = json.loads(args.output.read_text())
    summary = json.loads(Path(str(args.output) + ".summary.json").read_text())
    cleanup = json.loads(Path(str(args.output) + ".cleanup.json").read_text())
    assert record["error"] is None
    assert record["scores"]["wer"]["rate"] == 0
    assert summary["quality_complete"] is True
    assert summary["latency"]["total_seconds"] == record["seconds"]
    assert cleanup["status"] == ("failed" if close_fails else "succeeded")
    assert cleanup["lifecycle_revision"] == "explicit_close_v1"
    assert summary["model_close_error"] == cleanup["error"]
    assert summary["model_close_seconds"] == cleanup["seconds"]


@pytest.mark.parametrize("failure", [RuntimeError("decode failed"), KeyboardInterrupt()])
def test_finally_closes_on_sample_error_or_interruption(args, monkeypatch, failure):
    closed = []

    def transcribe(audio, language):
        raise failure

    monkeypatch.setattr(
        evaluate,
        "make_backend",
        lambda args: evaluate.Backend(transcribe, lambda: closed.append(True)),
    )
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            evaluate.run(args)
    else:
        assert evaluate.run(args) == 1
    assert closed == [True]
    cleanup = json.loads(Path(str(args.output) + ".cleanup.json").read_text())
    assert cleanup["status"] == "succeeded"


def test_setup_failure_is_not_reported_as_closed(args, monkeypatch):
    def fail(args):
        raise RuntimeError("load failed")

    monkeypatch.setattr(evaluate, "make_backend", fail)
    assert evaluate.run(args) == 1
    cleanup = json.loads(Path(str(args.output) + ".cleanup.json").read_text())
    assert cleanup["status"] == "not_loaded"
    assert cleanup["seconds"] is None


@pytest.mark.parametrize("failure", [None, "setup", "transcribe", "close"])
def test_generic_standard_backend_redacts_config_and_errors(args, monkeypatch, failure, capsys):
    import standard_asr

    secret = "do-not-record-this-credential"
    calls = []

    class Config(BaseConfig[Literal["fake"]]):
        engine: Literal["fake"] = "fake"
        api_key: SecretStr = secret_field(...)
        model_dir: Path | None = None

    class Engine:
        properties = SimpleNamespace(model_id="fake/base")

        def __init__(self, **config):
            self.config = Config(**config)

        def transcribe(self, audio, params):
            samples, rate = audio
            calls.append(("transcribe", rate, params.language, len(samples)))
            if failure == "transcribe":
                raise RuntimeError(f"Provider leaked token: {secret}")
            return SimpleNamespace(text="hello", detected_language="en")

        def close(self):
            calls.append("close")
            if failure == "close":
                raise RuntimeError(f"Shutdown leaked token: {secret}")

    def create(key, **config):
        assert key == "fake/base"
        assert config["api_key"] == secret
        if failure == "setup":
            raise ValueError(f"Invalid API key: {secret}")
        return Engine(**config)

    def discover(*, strict):
        assert strict is True
        return SimpleNamespace(create=create)

    monkeypatch.setattr(standard_asr, "discover_models", discover)
    args.backend = "standard"
    args.model_dir = None
    args.model_key = "fake/base"
    args.engine_config = {"api_key": secret, "model_dir": "local-model"}
    args.language_mode = "manifest"
    assert evaluate.run(args) == int(failure is not None)
    record = json.loads(args.output.read_text())
    summary_path = Path(str(args.output) + ".summary.json")
    cleanup_path = Path(str(args.output) + ".cleanup.json")
    summary = json.loads(summary_path.read_text())
    text = args.output.read_text() + summary_path.read_text() + cleanup_path.read_text()
    captured = capsys.readouterr()
    assert secret not in text + captured.out + captured.err
    assert record["model_key"] == "fake/base"
    assert record["model_dir"] is None
    assert record["compute_units"] == "plugin_managed"
    assert record["max_new_tokens"] is None
    assert summary["model_load_seconds"] is None
    assert summary["engine_create_seconds"] >= 0
    if failure != "setup":
        assert calls[0] == ("transcribe", 16000, "en", 1600)
        assert calls[-1] == "close"
        assert (
            summary["model_metadata"]["engine_config"] == Config(**args.engine_config).public_dump()
        )
    if failure in (None, "close"):
        assert record["scores"]["wer"]["rate"] == 0


def test_standard_factory_allows_optional_close(args, monkeypatch):
    import standard_asr

    engine = SimpleNamespace(
        properties=SimpleNamespace(model_id="fake/base"),
        config=SimpleNamespace(public_dump=lambda: {"engine": "fake"}),
        transcribe=lambda audio, params: SimpleNamespace(text="hello", detected_language=None),
    )
    monkeypatch.setattr(
        standard_asr,
        "discover_models",
        lambda **kwargs: SimpleNamespace(create=lambda *args, **kwargs: engine),
    )
    args.backend, args.model_key, args.engine_config = "standard", "fake/base", {}
    backend = evaluate.make_backend(args)
    assert backend.close is None
    assert backend.transcribe(np.zeros(1600), None) == ("hello", None, None)


def test_standard_cli_requires_key_but_not_artifact_root(monkeypatch, tmp_path):
    argv = [
        "evaluate.py",
        "--backend",
        "standard",
        "--manifest",
        "corpus.jsonl",
        "--output",
        str(tmp_path / "run.jsonl"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    observed = []
    monkeypatch.setattr(evaluate, "run", lambda args: observed.append(args) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--model-key", "fake/base", "--engine-config", '{"device":"accelerator"}'],
    )
    assert evaluate.main() == 0
    assert observed[0].model_dir is None
    assert observed[0].engine_config == {"device": "accelerator"}


@pytest.mark.parametrize("value", ["[]", "null", "not-json-with-secret"])
def test_engine_config_cli_rejects_nonobjects_without_echoing_input(value):
    with pytest.raises(argparse.ArgumentTypeError) as caught:
        evaluate.engine_config_json(value)
    assert str(caught.value) == "Engine configuration must be a JSON object."
