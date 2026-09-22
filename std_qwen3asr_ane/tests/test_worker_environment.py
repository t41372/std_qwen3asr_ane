from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from std_qwen3asr_ane import worker_environment


def _spec() -> worker_environment.WorkerEnvironmentSpec:
    return worker_environment.WorkerEnvironmentSpec(
        name="test-worker",
        version=2,
        dependencies=("example==1.2.3",),
    )


def _write_ready_environment(root: Path, spec: worker_environment.WorkerEnvironmentSpec) -> None:
    (root / "bin").mkdir(parents=True)
    python = root / "bin/python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    (root / ".std-qwen3asr-worker.json").write_text(
        json.dumps(worker_environment._expected_receipt(spec))
    )


def test_inspection_is_read_only_when_environment_is_missing(tmp_path):
    root = tmp_path / "runtime-v2"

    result = worker_environment.inspect_worker_environment(root, _spec())

    assert not result.ready
    assert result.reason == "missing"
    assert not root.exists()


def test_environment_receipt_must_match_exact_dependencies(tmp_path):
    root = tmp_path / "runtime-v2"
    (root / "bin").mkdir(parents=True)
    python = root / "bin/python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    wrong = worker_environment._expected_receipt(_spec())
    wrong["dependencies"] = ["example==9.9.9"]
    (root / ".std-qwen3asr-worker.json").write_text(json.dumps(wrong))

    result = worker_environment.inspect_worker_environment(root, _spec())

    assert not result.ready
    assert result.reason == "receipt_mismatch"


def test_explicit_environment_creation_publishes_atomically_and_supports_offline(
    tmp_path, monkeypatch
):
    root = tmp_path / "runtime-v2"
    calls: list[list[str]] = []

    def run(command, *, check, stdout, stderr):
        assert stdout is sys.stderr
        assert stderr is sys.stderr
        calls.append(command)
        if command[1] == "venv":
            environment = Path(command[2])
            (environment / "bin").mkdir(parents=True)
            python = environment / "bin/python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_environment.subprocess, "run", run)
    import uv

    monkeypatch.setattr(uv, "find_uv_bin", lambda: "/test/uv")

    python = worker_environment.ensure_worker_environment(root, _spec(), allow_downloads=False)

    assert python == root / "bin/python"
    assert worker_environment.inspect_worker_environment(root, _spec()).ready
    assert "--offline" in calls[1]
    assert calls[1][-1] == "example==1.2.3"
    assert not any(path.name.startswith(".runtime-v2.tmp-") for path in tmp_path.iterdir())
    assert os.access(python, os.X_OK)


def test_cache_name_changes_with_python_abi_and_dependencies():
    base = _spec()
    another_abi = worker_environment.WorkerEnvironmentSpec(
        name=base.name,
        version=base.version,
        dependencies=base.dependencies,
        python_abi="cpython-999",
    )
    another_dependency = worker_environment.WorkerEnvironmentSpec(
        name=base.name,
        version=base.version,
        dependencies=("example==4.5.6",),
    )

    name = worker_environment.worker_environment_name("runtime", base)

    assert name == worker_environment.worker_environment_name("runtime", base)
    assert name != worker_environment.worker_environment_name("runtime", another_abi)
    assert name != worker_environment.worker_environment_name("runtime", another_dependency)
    assert name.startswith("runtime-v2-")


def test_matching_legacy_is_reused_but_corrupt_legacy_does_not_block_new_generation(
    tmp_path, monkeypatch
):
    spec = _spec()
    legacy = tmp_path / "runtime-v2"
    _write_ready_environment(legacy, spec)

    selected = worker_environment.select_worker_environment_root(
        tmp_path, "runtime", spec, legacy_names=(legacy.name,)
    )

    assert selected == legacy
    (legacy / ".std-qwen3asr-worker.json").write_text("not json")
    selected = worker_environment.select_worker_environment_root(
        tmp_path, "runtime", spec, legacy_names=(legacy.name,)
    )
    assert selected == tmp_path / worker_environment.worker_environment_name("runtime", spec)
    assert legacy.exists()

    def run(command, *, check, stdout, stderr):
        if command[1] == "venv":
            environment = Path(command[2])
            (environment / "bin").mkdir(parents=True)
            python = environment / "bin/python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_environment.subprocess, "run", run)
    import uv

    monkeypatch.setattr(uv, "find_uv_bin", lambda: "/test/uv")
    python = worker_environment.ensure_worker_environment(selected, spec, allow_downloads=False)
    assert python == selected / "bin/python"
    assert worker_environment.inspect_worker_environment(selected, spec).ready
    assert (legacy / ".std-qwen3asr-worker.json").read_text() == "not json"


def test_environment_creation_refuses_a_different_python_abi(tmp_path):
    spec = worker_environment.WorkerEnvironmentSpec(
        name="test-worker",
        version=1,
        dependencies=("example==1.2.3",),
        python_abi="cpython-999",
    )
    with pytest.raises(RuntimeError, match="requires Python ABI"):
        worker_environment.ensure_worker_environment(
            tmp_path / "runtime-v1", spec, allow_downloads=False
        )
    assert not (tmp_path / "runtime-v1").exists()
