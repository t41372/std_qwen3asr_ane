"""Protocol integration tests; fake runtimes test the adapter, never model quality."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from standard_asr import discover_models
from standard_asr.compliance import (
    check_entrypoints,
    check_provider_params_swap_safety,
    check_recommended_wire_format,
    check_transcription_result,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.engine import (
    DIARIZE,
    ArtifactContext,
    AudioFormat,
    RuntimeParams,
    resolve_cache_dir,
)

from std_qwen3asr_ane.errors import ModelLimitError
from std_qwen3asr_ane.languages import LANGUAGE_NAMES, normalize_model_language
from std_qwen3asr_ane.plugin import (
    MODEL_ID,
    PROMPT_MAX_TOKENS,
    Qwen3ASREngine,
    create_engine,
    detected_language,
)


@pytest.fixture(autouse=True)
def mock_acquisition_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocked transfers set their policy explicitly, independently of the host."""
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")


def make_bundle(tmp_path: Path) -> Path:
    """A complete file layout, intentionally not a usable model."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {
        "frontend": "frontend.mlpackage",
        "encoder": "encoder.mlpackage",
        "decoder": "decoder.mlpackage",
        "lm_head": "lm_head.mlpackage",
        "embedding": "embedding.npy",
        "tokenizer": "tokenizer.json",
        "mel_filters": "mel_filters.npy",
    }
    for filename in files.values():
        payload = tmp_path / filename
        if payload.suffix == ".mlpackage":
            data = payload / "Data/com.apple.CoreML"
            (data / "weights").mkdir(parents=True)
            (data / "model.mlmodel").write_bytes(b"model specification fixture")
            (data / "weights/custom-name.bin").write_bytes(b"weight fixture")
            (payload / "Manifest.json").write_text(
                json.dumps(
                    {
                        "fileFormatVersion": "1.0.0",
                        "itemInfoEntries": {
                            "model": {"path": "com.apple.CoreML/model.mlmodel"},
                            "weights": {"path": "com.apple.CoreML/weights"},
                        },
                        "rootModelIdentifier": "model",
                    }
                )
            )
        else:
            payload.write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": MODEL_ID,
                "source_revision": "a" * 40,
                "files": files,
            }
        )
    )
    return tmp_path


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return make_bundle(tmp_path)


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    class Runtime:
        instances: ClassVar[list] = []

        def __init__(self, model_dir):
            self.model_dir = model_dir
            self.token_batch_size = 16
            self.calls = []
            self.active = 0
            self.max_active = 0
            self.closed = False
            self.instances.append(self)

        def close(self, *, timeout=5):
            assert self.active == 0
            self.closed = True

        def transcribe(self, samples, *, language, max_new_tokens, context=""):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.01)
                self.calls.append((samples.copy(), language, max_new_tokens))
                return SimpleNamespace(text="hello", language="English")
            finally:
                self.active -= 1

    module = ModuleType("std_qwen3asr_ane.runtime")
    module.CoreMLRuntime = Runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return Runtime


def test_discovery_and_official_compliance() -> None:
    registry = discover_models(strict=True)
    engine = registry.create("std-qwen3asr-ane/1.7b")
    assert type(engine) is Qwen3ASREngine
    reports = [
        check_entrypoints(),
        check_provider_params_swap_safety(engine),
        check_recommended_wire_format(engine),
    ]
    for report in reports:
        assert report.passed, report.issues
        assert not report.issues


def test_construction_is_pure_and_runtime_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Engine construction accessed the filesystem")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    engine = create_engine(model_dir="does-not-exist")
    assert engine._runtime is None
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from std_qwen3asr_ane.plugin import create_engine; create_engine(); "
                "assert not {'coremltools', 'torch', 'std_qwen3asr_ane.runtime'} & sys.modules.keys()"
            ),
        ],
        check=True,
    )


def test_config_environment_and_explicit_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR", "local/env-bundle")
    assert create_engine().config.model_dir == Path("local/env-bundle")
    assert create_engine(model_dir="local/explicit").config.model_dir == Path("local/explicit")


def test_short_profile_defaults_preserve_explicit_and_environment_values(monkeypatch):
    engine = create_engine(profile="short-dictation")
    assert (
        engine.config.model_dir
        == resolve_cache_dir() / "std-qwen3asr-ane/qwen3-asr-1.7b-short-dictation"
    )
    assert engine.config.max_new_tokens == 128
    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MAX_NEW_TOKENS", "200")
    assert create_engine(profile="short-dictation").config.max_new_tokens == 200
    assert create_engine(profile="short-dictation", max_new_tokens=64).config.max_new_tokens == 64
    assert create_engine(profile="short-dictation", model_dir="custom").config.model_dir == Path(
        "custom"
    )


def test_short_profile_rejects_a_general_bundle_before_native_loading(bundle, fake_runtime):
    from standard_asr.contract.exceptions import ConfigError

    with pytest.raises(ConfigError, match="cache-512"):
        create_engine(profile="short-dictation", model_dir=bundle).prepare()
    assert not fake_runtime.instances


def test_missing_artifact_reports_actions_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "missing"
    engine = create_engine(model_dir=root)
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: False)
    report = engine.artifact_status()
    (requirement,) = report.requirements
    assert report.readiness == "unavailable"
    assert requirement.state == "missing"
    assert requirement.location == root
    assert requirement.can_acquire_now
    assert requirement.acquisition_blocker is None
    assert requirement.required_actions == ()
    assert not requirement.may_acquire_during_inference
    assert engine.artifact_status() == report
    with pytest.raises(ArtifactUnavailableError) as unavailable:
        engine.prepare()
    assert "standard-asr pull" in (unavailable.value.hint or "")
    with pytest.raises(ArtifactUnavailableError):
        engine.transcribe((np.zeros(1600, dtype=np.float32), 16000))
    assert not root.exists()
    # With the toolchain installed the framework may run the conversion itself.
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: True)
    (requirement,) = engine.artifact_status().requirements
    assert requirement.can_acquire_now and requirement.acquisition_blocker is None
    assert requirement.required_actions == ()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = create_engine(model_dir=root, source_dir=tmp_path / "no-source")
    (requirement,) = engine.artifact_status().requirements
    assert requirement.acquisition_blocker == "downloads_disabled"
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "downloads_disabled"


def test_complete_bundle_acquisition_is_noop(bundle: Path) -> None:
    engine = create_engine(model_dir=bundle)
    report = engine.artifact_status()
    assert report.readiness == "ready"
    assert engine.acquire_artifacts() == report
    assert engine._runtime is None


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"schema_version": True}, "corrupt"),
        ({"schema_version": 2}, "corrupt"),
        ({"model_id": "Qwen/Qwen3-ASR-0.6B"}, "corrupt"),
        ({"source_revision": None}, "corrupt"),
        ({"files": {}}, "incomplete"),
    ],
)
def test_invalid_manifest(bundle: Path, change: dict, expected: str) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(change)
    path.write_text(json.dumps(manifest))
    report = create_engine(model_dir=bundle).artifact_status()
    assert report.requirements[0].state == expected


def test_corrupt_json_and_missing_payload(bundle: Path) -> None:
    engine = create_engine(model_dir=bundle)
    (bundle / "embedding.npy").unlink()
    assert engine.artifact_status().requirements[0].state == "incomplete"
    (bundle / "manifest.json").write_text("{")
    assert engine.artifact_status().requirements[0].state == "corrupt"


def test_manifest_cannot_escape_bundle(bundle: Path) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["tokenizer"] = "../outside.json"
    path.write_text(json.dumps(manifest))
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "corrupt"


@pytest.mark.parametrize(
    "relative",
    [
        "Data/com.apple.CoreML/weights/custom-name.bin",
        "Data/com.apple.CoreML/weights",
        "Data/com.apple.CoreML/model.mlmodel",
        "Data",
    ],
)
def test_package_missing_payload_is_incomplete(bundle: Path, relative: str):
    path = bundle / "frontend.mlpackage" / relative
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "incomplete"


@pytest.mark.parametrize(
    "change",
    [
        {"itemInfoEntries": []},
        {"rootModelIdentifier": "absent"},
        {"itemInfoEntries": {"model": {"path": "../../../outside.bin"}}},
        {"itemInfoEntries": {"model": {"path": "/outside.bin"}}},
        {"itemInfoEntries": {"model": {"path": "invalid\u0000path"}}},
    ],
)
def test_package_malformed_manifest_is_corrupt(bundle: Path, change: dict):
    path = bundle / "frontend.mlpackage/Manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(change)
    path.write_text(json.dumps(manifest))
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "corrupt"


def test_package_invalid_json_is_corrupt(bundle: Path):
    (bundle / "frontend.mlpackage/Manifest.json").write_text("{")
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "corrupt"


@pytest.mark.parametrize("relative", ["Data/com.apple.CoreML/weights", "Data", "Manifest.json"])
def test_package_symlink_escape_is_corrupt(bundle: Path, relative: str):
    package = bundle / "frontend.mlpackage"
    outside = bundle / "outside-package"
    source = package / relative
    if source.is_dir():
        shutil.copytree(source, outside)
        shutil.rmtree(source)
    else:
        outside.write_bytes(source.read_bytes())
        source.unlink()
    source.symlink_to(outside, target_is_directory=outside.is_dir())
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "corrupt"


def test_package_nested_weight_symlink_escape_is_corrupt(bundle: Path):
    outside = bundle / "unrelated.bin"
    outside.write_bytes(b"not package weights")
    (bundle / "frontend.mlpackage/Data/com.apple.CoreML/weights/extra.bin").symlink_to(outside)
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "corrupt"


def test_weightless_inline_package_is_ready(bundle: Path):
    package = bundle / "frontend.mlpackage"
    path = package / "Manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["itemInfoEntries"]["weights"]
    path.write_text(json.dumps(manifest))
    shutil.rmtree(package / "Data/com.apple.CoreML/weights")
    assert create_engine(model_dir=bundle).artifact_status().readiness == "ready"


def test_empty_weight_payload_is_incomplete(bundle: Path):
    (bundle / "frontend.mlpackage/Data/com.apple.CoreML/weights/custom-name.bin").write_bytes(b"")
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "incomplete"


def test_compiled_bundle_presence_and_weight_validation(bundle: Path):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    compiled = bundle / "frontend.mlmodelc"
    (compiled / "weights").mkdir(parents=True)
    for relative in ("coremldata.bin", "model.mil", "weights/weight.bin"):
        (compiled / relative).write_bytes(b"compiled fixture")
    manifest["files"]["frontend"] = compiled.name
    manifest_path.write_text(json.dumps(manifest))
    assert create_engine(model_dir=bundle).artifact_status().readiness == "ready"
    (compiled / "weights/weight.bin").unlink()
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "incomplete"
    (compiled / "weights/weight.bin").symlink_to(bundle / "embedding.npy")
    assert create_engine(model_dir=bundle).artifact_status().requirements[0].state == "corrupt"


def test_language_mapping() -> None:
    assert len(LANGUAGE_NAMES) == 30
    assert normalize_model_language("Cantonese") == "yue"
    assert normalize_model_language("Filipino") == "fil"
    assert normalize_model_language(" auto ") is None
    assert normalize_model_language("Chinese,English") is None


def test_batch_result_language_and_audio_contract(bundle: Path, fake_runtime) -> None:
    engine = create_engine(model_dir=bundle, max_new_tokens=123)
    samples = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe((samples, 16000))
    assert result.text == "hello"
    assert result.detected_language == "en"
    assert result.duration == 1.0
    assert result.words is None
    assert result.segments is None
    report = check_transcription_result(result, capabilities=engine.declared_capabilities)
    assert report.passed, report.issues
    (instance,) = fake_runtime.instances
    received, language, max_tokens = instance.calls[0]
    np.testing.assert_array_equal(received, samples)
    assert language is None
    assert max_tokens == 123
    forced = engine.transcribe((samples, 16000), RuntimeParams(language="yue"))
    assert instance.calls[-1][1] == "yue"
    assert forced.detected_language is None


def test_gating_rejects_unsupported_requests_before_loading(tmp_path: Path) -> None:
    engine = create_engine(model_dir=tmp_path)
    audio = (np.zeros(1600, dtype=np.float32), 16000)
    with pytest.raises(UnsupportedFeatureError):
        engine.transcribe(audio, RuntimeParams(language="sw"))
    with pytest.raises(UnsupportedFeatureError):
        engine.transcribe(audio, RuntimeParams(diarization=DIARIZE))
    with pytest.raises(UnsupportedFeatureError):
        engine.start_transcription(
            audio_format=AudioFormat(sample_rate=16000, encoding="pcm_s24le")
        )
    assert engine.artifact_status(ArtifactContext(mode="streaming")).mode == "streaming"
    assert engine._runtime is None


def test_prepare_is_idempotent_and_inference_is_serialized(bundle: Path, fake_runtime) -> None:
    engine = create_engine(model_dir=bundle)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(engine.prepare) for _ in range(4)]
        for future in futures:
            future.result()
        futures = [
            pool.submit(engine.transcribe, (np.zeros(1600, dtype=np.float32), 16000))
            for _ in range(8)
        ]
        assert [future.result().text for future in futures] == ["hello"] * 8
    (instance,) = fake_runtime.instances
    assert instance.max_active == 1
    assert len(instance.calls) == 8


def test_native_inference_errors_have_portable_type(
    bundle: Path, fake_runtime, monkeypatch
) -> None:
    native = RuntimeError("decode failed")

    def fail(*args, **kwargs):
        raise native

    monkeypatch.setattr(fake_runtime, "transcribe", fail)
    with pytest.raises(TranscriptionError) as caught:
        create_engine(model_dir=bundle).transcribe((np.zeros(1600, dtype=np.float32), 16000))
    assert caught.value.__cause__ is native
    assert "capacity" not in str(caught.value)


def test_bundle_capacity_errors_name_their_limit(bundle: Path, fake_runtime, monkeypatch) -> None:
    limit = ModelLimitError("Audio exceeds this bundle's 30-second limit")

    def fail(*args, **kwargs):
        raise limit

    monkeypatch.setattr(fake_runtime, "transcribe", fail)
    with pytest.raises(TranscriptionError, match="30-second limit") as caught:
        create_engine(model_dir=bundle).transcribe((np.zeros(1600, dtype=np.float32), 16000))
    assert caught.value.__cause__ is limit


def test_prompt_budget_is_declared_and_gated(bundle: Path, fake_runtime) -> None:
    engine = create_engine(model_dir=bundle)
    for scope in (engine.declared_capabilities.batch, engine.declared_capabilities.streaming):
        assert scope.guidance.prompt.constraints.max_tokens == PROMPT_MAX_TOKENS
    words = " ".join(f"w{index}" for index in range(PROMPT_MAX_TOKENS + 50))
    audio = (np.zeros(1600, dtype=np.float32), 16000)
    with pytest.raises(UnsupportedFeatureError):
        create_engine(model_dir=bundle, strict=True).transcribe(audio, RuntimeParams(prompt=words))
    lenient = create_engine(model_dir=bundle, strict=False)
    result = lenient.transcribe(audio, RuntimeParams(prompt=words))
    assert any(diagnostic.code == "prompt_truncated" for diagnostic in result.diagnostics)
    (instance,) = fake_runtime.instances
    assert instance.calls, "best-effort mode still transcribes with a truncated prompt"


def test_unmapped_detected_language_is_disclosed(bundle: Path, fake_runtime, monkeypatch) -> None:
    assert detected_language("English", None) == ("en", [])
    assert detected_language("English", "en") == (None, [])
    assert detected_language(None, None) == (None, [])
    detected, diagnostics = detected_language("Klingon", None)
    assert detected is None and [d.code for d in diagnostics] == ["detected_language_unmapped"]
    assert diagnostics[0].provided == "Klingon"

    def transcribe(self, samples, *, language, max_new_tokens, context=""):
        return SimpleNamespace(text="hi", language="Klingon")

    monkeypatch.setattr(fake_runtime, "transcribe", transcribe)
    result = create_engine(model_dir=bundle).transcribe((np.zeros(1600, dtype=np.float32), 16000))
    assert result.detected_language is None
    assert [d.code for d in result.diagnostics] == ["detected_language_unmapped"]
    report = check_transcription_result(result, capabilities=Qwen3ASREngine.declared_capabilities)
    assert report.passed, report.issues


def test_close_waits_for_active_inference_and_can_reopen(bundle: Path, fake_runtime):
    engine = create_engine(model_dir=bundle)
    engine.prepare()
    runtime = engine._runtime
    entered, release, close_started = Event(), Event(), Event()
    original = runtime.transcribe

    def blocking_transcribe(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    runtime.transcribe = blocking_transcribe

    def close():
        close_started.set()
        engine.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        transcription = pool.submit(engine.transcribe, (np.zeros(1600, dtype=np.float32), 16000))
        assert entered.wait(1)
        closing = pool.submit(close)
        assert close_started.wait(1)
        try:
            assert not closing.done()
            assert not runtime.closed
            assert engine._runtime is runtime
        finally:
            release.set()
        assert transcription.result().text == "hello"
        closing.result()
    assert runtime.closed and engine._runtime is None
    engine.close()  # Idempotent while unloaded.
    engine.prepare()
    assert engine._runtime is not runtime
    assert len(fake_runtime.instances) == 2
    engine.close()


def test_close_failure_keeps_runtime_owner_for_retry(bundle: Path, fake_runtime):
    engine = create_engine(model_dir=bundle)
    engine.prepare()
    runtime = engine._runtime
    original = runtime.close

    def fail(*, timeout):
        raise RuntimeError("native still borrows buffers")

    runtime.close = fail
    with pytest.raises(RuntimeError, match="still borrows"):
        engine.close(timeout=0.01)
    assert engine._runtime is runtime
    assert not runtime.closed
    runtime.close = original
    engine.close()
    assert engine._runtime is None


def test_context_manager_closes_when_body_raises(bundle: Path, fake_runtime):
    engine = create_engine(model_dir=bundle)
    with pytest.raises(ValueError, match="body failed"), engine as opened:
        assert opened is engine
        assert engine._runtime is not None
        raise ValueError("body failed")
    assert engine._runtime is None
    assert fake_runtime.instances[0].closed


def draft_bundle(root: Path) -> Path:
    """A complete draft-bundle layout, intentionally not usable models."""
    root.mkdir()
    head = root / "verify_head.mlmodelc"
    (head / "weights").mkdir(parents=True)
    for name in ("coremldata.bin", "model.mil", "weights/weight.bin"):
        (head / name).write_bytes(b"x")
    checkpoint = root / "Qwen3-ASR-0.6B"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"x")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "qwen3-asr-ane-draft",
                "draft": {
                    "model_id": "Qwen/Qwen3-ASR-0.6B",
                    "revision": "b" * 40,
                    "path": "Qwen3-ASR-0.6B",
                },
                "verify_head": {"path": "verify_head.mlmodelc", "token_batch_size": 16},
                "target": {},
            }
        )
    )
    return root


def test_draft_requirement_is_reported_and_actionable(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_engine(model_dir=bundle, draft_dir=tmp_path / "draft")
    report = engine.artifact_status()
    assert report.readiness != "ready"
    draft = [r for r in report.requirements if r.artifact_id == "qwen3-asr-0.6b-gpu-draft"]
    assert len(draft) == 1 and draft[0].state == "missing"
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: False)
    draft = [
        r
        for r in engine.artifact_status().requirements
        if r.artifact_id == "qwen3-asr-0.6b-gpu-draft"
    ]
    assert draft[0].can_acquire_now and draft[0].required_actions == ()
    assert engine.artifact_status(ArtifactContext(mode="streaming")).readiness == "ready"
    with pytest.raises(ArtifactUnavailableError) as info:
        engine.transcribe((np.zeros(16000, dtype=np.float32), 16000), RuntimeParams())
    assert (
        str(tmp_path / "draft") in (info.value.hint or "")
        and "standard-asr pull" in info.value.hint
    )
    draft_bundle(tmp_path / "draft")
    assert engine.artifact_status().readiness == "ready"
    (tmp_path / "draft/Qwen3-ASR-0.6B/model.safetensors").write_bytes(b"")
    assert [r.state for r in engine.artifact_status().requirements][1] == "incomplete"
    engine_without = create_engine(model_dir=bundle)
    assert len(engine_without.artifact_status().requirements) == 1


def test_draft_path_verifies_on_the_runtime(
    bundle: Path, tmp_path: Path, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft_bundle(tmp_path / "draft")
    loaded = []

    class Draft:
        def __init__(self, root, runtime, *, quantize_bits):
            loaded.append((root, runtime, quantize_bits))
            self.closed = False

        def close(self, *, timeout=5):
            self.closed = True

    module = ModuleType("std_qwen3asr_ane.draft")
    module.DraftRuntime = Draft
    module.DraftDependencyError = type("DraftDependencyError", (ImportError,), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def transcribe_speculative(
        self, samples, draft, *, language, max_new_tokens, context, lookahead
    ):
        self.calls.append(("speculative", draft, lookahead, max_new_tokens))
        return SimpleNamespace(text="fast", language="English")

    fake_runtime.transcribe_speculative = transcribe_speculative
    engine = create_engine(model_dir=bundle, draft_dir=tmp_path / "draft", draft_lookahead=7)
    engine.prepare()
    assert not loaded and engine._draft is None
    result = engine.transcribe((np.zeros(16000, dtype=np.float32), 16000), RuntimeParams())
    assert result.text == "fast"
    runtime = fake_runtime.instances[-1]
    assert runtime.calls[-1][0] == "speculative" and runtime.calls[-1][2] == 7
    assert loaded[0][0] == (tmp_path / "draft").resolve() and loaded[0][2] == 4
    engine.close()
    assert runtime.closed and loaded and engine._draft is None


def test_missing_mlx_is_a_configuration_error(
    bundle: Path, tmp_path: Path, fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from standard_asr.contract.exceptions import ConfigError

    draft_bundle(tmp_path / "draft")
    module = ModuleType("std_qwen3asr_ane.draft")
    module.DraftDependencyError = type("DraftDependencyError", (ImportError,), {})

    def fail(root, runtime, *, quantize_bits):
        raise module.DraftDependencyError("no mlx")

    module.DraftRuntime = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = create_engine(model_dir=bundle, draft_dir=tmp_path / "draft")
    with pytest.raises(ConfigError) as info:
        engine.transcribe((np.zeros(16000, dtype=np.float32), 16000), RuntimeParams())
    assert "gpu-draft" in (info.value.hint or "")
    assert not fake_runtime.instances[-1].closed
    assert engine._runtime is fake_runtime.instances[-1]
    assert engine._draft is None
    engine.close()


def test_prepare_target_does_not_require_a_configured_draft(bundle, tmp_path, fake_runtime):
    engine = create_engine(model_dir=bundle, draft_dir=tmp_path / "missing-draft")
    assert engine.artifact_status().readiness != "ready"
    engine.prepare()
    assert engine._runtime is fake_runtime.instances[-1]
    assert engine._draft is None
    with pytest.raises(ArtifactUnavailableError):
        engine.transcribe((np.zeros(16000, dtype=np.float32), 16000), RuntimeParams())
    assert not engine._runtime.closed
    engine.close()


def test_pull_runs_the_measured_recipe_in_a_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import std_qwen3asr_ane.conversion.draft as draft_build
    from std_qwen3asr_ane import compiled
    from std_qwen3asr_ane.conversion import build, compress

    calls = []
    source = tmp_path / "source"
    target = tmp_path / "bundle"
    draft = tmp_path / "draft"

    def fake_download(destination, *, revision, model_id):
        calls.append(("download", destination, revision, model_id))
        destination.mkdir(parents=True)
        (destination / "source.json").write_text(
            json.dumps({"model_id": model_id, "revision": revision})
        )

    def fake_build(src, output, **recipe):
        calls.append(("build", src, output, recipe))
        output.mkdir(parents=True)

    def fake_compress(src, output, **recipe):
        calls.append(("compress", src, output, recipe))
        output.mkdir(parents=True)

    def fake_compile(src, output):
        calls.append(("compile", src, output))
        make_bundle(output)

    def fake_draft(target_dir, src, output, *, draft_source):
        calls.append(("draft", target_dir, src, output))
        draft_bundle(output)

    monkeypatch.setattr(build, "download_source", fake_download)
    monkeypatch.setattr(build, "build_bundle", fake_build)
    monkeypatch.setattr(compress, "compress_bundle", fake_compress)
    monkeypatch.setattr(compiled, "compile_bundle", fake_compile)
    monkeypatch.setattr(draft_build, "build_draft_bundle", fake_draft)
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: True)
    engine = create_engine(
        model_dir=target,
        source_dir=source,
        draft_dir=draft,
        draft_source_dir=tmp_path / "draft-source",
    )
    assert engine.artifact_status().readiness == "unavailable"
    events = []
    report = engine.acquire_artifacts(progress=events.append)
    assert report.readiness == "ready"
    assert [c[0] for c in calls] == [
        "download",
        "build",
        "compress",
        "compile",
        "download",
        "draft",
    ]
    assert calls[0][2] == "7278e1e70fe206f11671096ffdd38061171dd6e5"
    assert calls[1][3] == {"cache_length": 1024, "token_batch_size": 16, "layers_per_partition": 14}
    assert calls[2][3] == {"scheme": "palette", "bits": 8, "group_size": 32}
    work = calls[1][2].parent
    assert calls[2][2] == work / "lut8" and calls[3][2] == work / "compiled"
    assert calls[5][1] == target and calls[5][3].name == "bundle"
    assert not work.exists() and target.is_dir() and draft.is_dir()
    phases = [event.phase for event in events]
    assert phases[0] == "resolving" and phases[-1] == "finalizing"
    assert phases[1:-1] == ["transferring", "converting", "verifying", "transferring", "converting"]
    # A second pull has nothing to do; an existing target is never overwritten.
    calls.clear()
    assert engine.acquire_artifacts().readiness == "ready" and calls == []
    shutil.rmtree(draft)
    (draft / "stale").mkdir(parents=True)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "action_required"
    assert any("move it away" in action.message for action in caught.value.required_actions)


def test_pull_failure_is_structured_and_keeps_no_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from std_qwen3asr_ane.conversion import build

    source = tmp_path / "source"
    source.mkdir()
    (source / "source.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5"})
    )

    def failing_build(src, output, **recipe):
        raise RuntimeError("conversion exploded")

    monkeypatch.setattr(build, "build_bundle", failing_build)
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: True)
    engine = create_engine(model_dir=tmp_path / "bundle", source_dir=source)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "failed" and "conversion exploded" in str(caught.value.__cause__)
    assert "standard-asr pull" in (caught.value.hint or "")
    assert not (tmp_path / "bundle").exists()


def test_draft_close_failure_retains_target_until_retry_succeeds(bundle, fake_runtime):
    engine = create_engine(model_dir=bundle)
    engine.prepare()
    runtime = engine._runtime
    calls = []

    class Draft:
        def close(self, *, timeout):
            calls.append("draft")
            if len(calls) == 1:
                raise RuntimeError("draft still owns target buffers")

    draft = Draft()
    engine._draft = draft
    with pytest.raises(RuntimeError, match="owns target buffers"):
        engine.close()
    assert engine._draft is draft
    assert engine._runtime is runtime
    assert not runtime.closed
    engine.close()
    assert engine._draft is None and engine._runtime is None
    assert runtime.closed
