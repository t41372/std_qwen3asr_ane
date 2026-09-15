"""Conversion process boundaries must preserve the Standard ASR contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from standard_asr.contract.exceptions import ArtifactAcquisitionError
from standard_asr.engine import ArtifactAction

from std_qwen3asr_ane import acquisition, conversion_worker
from std_qwen3asr_ane.conversion import toolchain
from std_qwen3asr_ane.plugin import BUNDLE_ARTIFACT_ID, create_engine


def test_worker_metadata_and_inline_conversion_require_the_same_versions():
    script = Path(conversion_worker.__file__).read_text()
    metadata = script.split("# /// script\n", 1)[1].split("# ///", 1)[0]
    requirements = tomllib.loads(
        "\n".join(line.removeprefix("# ") for line in metadata.splitlines())
    )
    pinned = {
        name: version
        for spec in requirements["dependencies"]
        if "==" in spec
        for name, version in [spec.split("==")]
    }
    assert pinned == toolchain.CONVERSION_VERSIONS


@pytest.mark.parametrize("changed", list(toolchain.CONVERSION_VERSIONS))
def test_any_conversion_version_drift_uses_the_managed_worker(monkeypatch, changed):
    installed = dict(toolchain.CONVERSION_VERSIONS)
    monkeypatch.setattr(toolchain, "version", installed.__getitem__)
    assert toolchain.conversion_toolchain_available()
    installed[changed] = "999.0"
    assert not toolchain.conversion_toolchain_available()


def test_missing_conversion_distribution_is_not_imported(monkeypatch):
    def absent(name):
        raise toolchain.PackageNotFoundError(name)

    monkeypatch.setattr(toolchain, "version", absent)
    assert not toolchain.conversion_toolchain_available()


def test_failure_frame_preserves_operator_actions_and_retry_delay():
    error = ArtifactAcquisitionError(
        "A prerequisite needs attention.",
        reason="action_required",
        hint="Select a valid source.",
        required_actions=(
            ArtifactAction(kind="provide_artifacts", message="Choose the pinned source."),
        ),
        retriable_after=2.0,
    )
    frame = acquisition.AcquisitionFailure.from_exception(error).model_dump_json()
    restored = acquisition.AcquisitionFailure.model_validate_json(frame).to_exception()
    assert restored.reason == error.reason
    assert restored.hint == error.hint
    assert restored.required_actions == error.required_actions
    assert restored.retriable_after == error.retriable_after


def test_real_worker_reports_invalid_source_as_action_required(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.json").write_text("{}")
    engine = create_engine(download_root=tmp_path, source_dir=source)
    request = {"config": engine.config.model_dump(mode="json"), "targets": [BUNDLE_ARTIFACT_ID]}
    result = subprocess.run(
        [sys.executable, conversion_worker.__file__],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    failure = json.loads(result.stdout)["error"]
    assert failure["reason"] == "action_required"
    assert "pinned" in failure["message"]
    assert "Traceback" not in result.stderr
    assert not engine.config.model_dir.exists()


def test_parent_retains_worker_error_reason_and_preflight_report(monkeypatch, tmp_path):
    import uv

    worker = tmp_path / "fake-uv"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "json.loads(sys.stdin.readline())\n"
        'print(json.dumps({"error": {"reason": "downloads_disabled", "message": "Downloads disabled."}}), flush=True)\n'
        "sys.exit(1)\n"
    )
    worker.chmod(0o755)
    monkeypatch.setattr(uv, "find_uv_bin", lambda: str(worker))
    monkeypatch.setattr("std_qwen3asr_ane.plugin._conversion_toolchain_available", lambda: False)
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    engine = create_engine(download_root=tmp_path)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "downloads_disabled"
    assert caught.value.report.requirements[0].artifact_id == BUNDLE_ARTIFACT_ID
    assert not engine.config.model_dir.exists()


@pytest.mark.parametrize("draft", [False, True])
def test_existing_directory_without_manifest_requires_action(tmp_path, draft):
    engine = create_engine(download_root=tmp_path, use_draft=draft)
    target = engine.config.draft_dir if draft else engine.config.model_dir
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("existing data")
    requirement = engine.artifact_status().requirements[-1]
    assert requirement.state == "incomplete"
    assert not requirement.can_acquire_now
    assert requirement.acquisition_blocker == "action_required"
    assert requirement.required_actions
    assert sentinel.read_text() == "existing data"
