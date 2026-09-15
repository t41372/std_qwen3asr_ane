"""Contract regressions that interface-only compliance does not exercise."""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np
import pytest
import test_plugin as plugin_fixtures
from standard_asr import discover_models
from standard_asr.contract.exceptions import ArtifactAcquisitionError
from standard_asr.engine import ArtifactContext, RuntimeParams

from std_qwen3asr_ane.acquisition import acquisition_lock
from std_qwen3asr_ane.plugin import Qwen3ASRParams, ShortDictationEngine, create_engine

bundle = plugin_fixtures.bundle
fake_runtime = plugin_fixtures.fake_runtime
make_bundle = plugin_fixtures.make_bundle


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")


def test_cache_location_is_independent_of_working_directory(monkeypatch, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    a = create_engine()
    monkeypatch.chdir(second)
    b = create_engine()
    assert a.config.model_dir == b.config.model_dir
    assert a.config.source_dir == b.config.source_dir
    assert a.config.model_dir == tmp_path / "cache/std-qwen3asr-ane/qwen3-asr-1.7b"
    assert not (tmp_path / "cache").exists()


def test_download_root_and_explicit_paths_follow_standard_precedence(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    engine = create_engine(download_root=explicit)
    assert engine.config.model_dir == explicit / "std-qwen3asr-ane/qwen3-asr-1.7b"
    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR", str(tmp_path / "env-model"))
    assert create_engine(download_root=explicit).config.model_dir == tmp_path / "env-model"
    assert create_engine(model_dir=tmp_path / "local").config.model_dir == tmp_path / "local"
    monkeypatch.delenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR")
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert create_engine().config.model_dir == (
        tmp_path / "xdg/standard-asr/std-qwen3asr-ane/qwen3-asr-1.7b"
    )


def test_constructor_does_not_probe_or_create_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("constructor touched the filesystem")

    with monkeypatch.context() as scoped:
        for name in ("stat", "exists", "is_file", "mkdir", "open", "resolve"):
            scoped.setattr(Path, name, forbidden)
        create_engine(use_draft=True)
        ShortDictationEngine()


def test_short_preset_is_discoverable_and_has_its_own_limits():
    registry = discover_models()
    engine = registry.create("std-qwen3asr-ane/1.7b-short-dictation")
    assert engine.properties.model_id == "std-qwen3asr-ane/1.7b-short-dictation"
    assert engine.properties.max_audio_duration == 12
    assert engine.config.max_new_tokens == 128
    assert engine.config.profile == "short-dictation"
    assert engine.config.model_dir.name == "qwen3-asr-1.7b-short-dictation"
    assert not engine.supports("batch.word_timestamps")
    assert engine.supports("streaming_input") and engine.supports("streaming_output")


def test_runtime_budget_does_not_mutate_engine_defaults(bundle, fake_runtime):
    engine = create_engine(model_dir=bundle, max_new_tokens=123)
    audio = (np.zeros(1600, dtype=np.float32), 16000)
    engine.transcribe(audio, RuntimeParams(provider_params=Qwen3ASRParams(max_new_tokens=42)))
    engine.transcribe(audio)
    assert [call[2] for call in fake_runtime.instances[0].calls] == [42, 123]
    assert engine.config.max_new_tokens == 123


def test_draft_is_required_only_for_batch(bundle):
    engine = create_engine(model_dir=bundle, use_draft=True)
    assert engine.config.draft_dir == bundle.with_name(bundle.name + "-draft")
    assert engine.artifact_status().readiness == "unavailable"
    streaming = engine.artifact_status(ArtifactContext(mode="streaming"))
    assert streaming.readiness == "ready"
    assert len(streaming.requirements) == 1


def test_plain_install_uses_managed_conversion_and_final_status(monkeypatch, tmp_path):
    from std_qwen3asr_ane import acquisition

    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: False)
    calls = []

    def worker(config, targets, progress):
        calls.append(targets)
        make_bundle(config.model_dir)

    monkeypatch.setattr(acquisition, "acquire_in_worker", worker)
    engine = create_engine(model_dir=tmp_path / "model")
    assert engine.acquire_artifacts().readiness == "ready"
    assert engine.acquire_artifacts(refresh=True).readiness == "ready"
    assert len(calls) == 1


def test_offline_draft_cannot_bypass_policy_with_cached_target_source(
    monkeypatch, bundle, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.json").write_text("{}")
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = create_engine(model_dir=bundle, source_dir=source, use_draft=True)
    report = engine.artifact_status()
    assert report.requirements[1].acquisition_blocker == "downloads_disabled"
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "downloads_disabled"
    assert engine.artifact_status(ArtifactContext(mode="streaming")).readiness == "ready"


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"])
def test_download_policy_is_checked_before_network(monkeypatch, tmp_path, model_id):
    from std_qwen3asr_ane.conversion.build import download_source

    def forbidden(*args, **kwargs):
        raise AssertionError("download disabled but network client was constructed")

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    monkeypatch.setattr("huggingface_hub.HfApi", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        download_source(tmp_path / "source", model_id=model_id)
    assert caught.value.reason == "downloads_disabled"
    assert not (tmp_path / "source").exists()


def test_concurrent_pull_reports_busy_instead_of_writing_shared_source(tmp_path):
    a = create_engine(model_dir=tmp_path / "a", source_dir=tmp_path / "shared-source")
    b = create_engine(model_dir=tmp_path / "b", source_dir=tmp_path / "shared-source")
    with acquisition_lock(a.config):
        with pytest.raises(ArtifactAcquisitionError) as caught:
            b.acquire_artifacts()
        assert caught.value.reason == "busy"
        assert caught.value.retriable_after == 1
        assert not b.config.model_dir.exists()
    # OS locks release on close; lock files remaining on disk are not stale locks.
    with acquisition_lock(b.config):
        pass


def test_remedy_preserves_paths_with_spaces_and_draft_configuration(tmp_path):
    engine = create_engine(model_dir=tmp_path / "custom model", use_draft=True)
    command = shlex.split(engine._pull_command())
    assert command[:3] == ["standard-asr", "pull", "std-qwen3asr-ane/1.7b"]
    assert f"model_dir={tmp_path / 'custom model'}" in command
    assert f"draft_dir={tmp_path / 'custom model-draft'}" in command
    assert not any("uv run" in item or "--project" in item for item in command)


def test_conversion_failure_never_publishes_partial_target(monkeypatch, tmp_path):
    import json

    from std_qwen3asr_ane import compiled
    from std_qwen3asr_ane.conversion import build, compress
    from std_qwen3asr_ane.plugin import MODEL_ID, SOURCE_REVISION

    engine = create_engine(model_dir=tmp_path / "model", source_dir=tmp_path / "source")
    engine.config.source_dir.mkdir()
    (engine.config.source_dir / "source.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": SOURCE_REVISION})
    )
    stale = tmp_path / "model.work"
    stale.mkdir()
    (stale / "keep.txt").write_text("not owned by this acquisition")
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: True)
    monkeypatch.setattr(build, "build_bundle", lambda source, output, **kw: output.mkdir())
    monkeypatch.setattr(compress, "compress_bundle", lambda source, output, **kw: output.mkdir())

    def fail_compile(source, output):
        output.mkdir()
        (output / "partial").write_text("not a complete model")
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(compiled, "compile_bundle", fail_compile)
    with pytest.raises(ArtifactAcquisitionError):
        engine.acquire_artifacts()
    assert not engine.config.model_dir.exists()
    assert (stale / "keep.txt").read_text() == "not owned by this acquisition"
    assert not list(tmp_path.glob(".model-*/"))
    monkeypatch.setattr(compiled, "compile_bundle", lambda source, output: make_bundle(output))
    assert engine.acquire_artifacts().readiness == "ready"


def test_progress_observer_failure_does_not_cancel_acquisition(monkeypatch, tmp_path):
    from standard_asr.contract.exceptions import ArtifactProgressCallbackError
    from standard_asr.engine import ArtifactProgress

    from std_qwen3asr_ane import acquisition

    engine = create_engine(model_dir=tmp_path / "model")
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: False)

    def worker(config, targets, progress):
        progress(ArtifactProgress(phase="converting"))
        make_bundle(config.model_dir)

    def broken_observer(event):
        raise RuntimeError("observer failed")

    monkeypatch.setattr(acquisition, "acquire_in_worker", worker)
    with pytest.raises(ArtifactProgressCallbackError) as caught:
        engine.acquire_artifacts(progress=broken_observer)
    assert caught.value.report.readiness == "ready"
    assert engine.artifact_status().readiness == "ready"


def test_standard_server_reads_plugin_schemas_and_transcribes(bundle, fake_runtime, monkeypatch):
    import base64
    import io
    import wave

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from standard_asr.toolchain.server import create_app

    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR", str(bundle))
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 1600)
    key = "std-qwen3asr-ane/1.7b"
    with TestClient(create_app(registry=discover_models())) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert {item["key"] for item in models.json()} >= {
            key, "std-qwen3asr-ane/1.7b-short-dictation",
        }
        for surface in ("capabilities", "metadata", "config-schema", "params-schema"):
            response = client.get(f"/v1/{surface}/{key}")
            assert response.status_code == 200, response.text
        response = client.post("/v1/transcribe:json", json={
            "model": key, "audio": base64.b64encode(audio.getvalue()).decode("ascii"),
        })
        assert response.status_code == 200, response.text
        assert response.json()["result"]["text"] == "hello"
