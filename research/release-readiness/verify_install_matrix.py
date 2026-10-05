"""Verify clean source-tree and wheel installs across supported Python profiles.

The verifier creates one isolated virtual environment per matrix cell. It never
syncs the checkout's ``.venv``. Logs and the detailed result are written below
an ignored artifact directory; ``--portable-output`` writes a path-independent
summary suitable for committing as release evidence.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_provenance import evidence_date

PROJECT_NAME = "std-qwen3asr-ane"
PROJECT_VERSION = "0.2.0a1"
MODEL_IDS = ("std-qwen3asr-ane/1.7b", "std-qwen3asr-ane/1.7b-short-dictation")
EXTRAS = ("server", "diarization", "gpu-draft")
PROFILES = (
    (),
    ("server",),
    ("diarization",),
    ("gpu-draft",),
    ("server", "diarization"),
    ("server", "gpu-draft"),
    ("diarization", "gpu-draft"),
    EXTRAS,
)
CONVERTER_ONLY_DISTRIBUTIONS = {
    "gradio",
    "jiwer",
    "qwen-asr",
    "safetensors",
    "torch",
    "transformers",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(entries: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, contents in sorted(entries.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def profile_name(extras: tuple[str, ...]) -> str:
    return "+".join(extras) if extras else "base"


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    seconds: float
    stdout_file: str
    stderr_file: str


class CellRunner:
    def __init__(self, cell_dir: Path, env: dict[str, str]) -> None:
        self.cell_dir = cell_dir
        self.env = env
        self.commands: list[CommandResult] = []
        self.index = 0

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        check: bool = True,
        timeout: float = 900,
    ) -> subprocess.CompletedProcess[str]:
        self.index += 1
        label = f"{self.index:02d}"
        stdout_path = self.cell_dir / f"{label}.stdout.log"
        stderr_path = self.cell_dir / f"{label}.stderr.log"
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout_path.write_text(error.stdout or "")
            stderr_path.write_text(error.stderr or "")
            raise RuntimeError(f"Command timed out after {timeout}s: {argv!r}") from error
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        self.commands.append(
            CommandResult(
                argv=argv,
                returncode=completed.returncode,
                seconds=round(time.monotonic() - started, 3),
                stdout_file=stdout_path.name,
                stderr_file=stderr_path.name,
            )
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit {completed.returncode}: {argv!r}; "
                f"see {stdout_path} and {stderr_path}"
            )
        return completed


def requirement_for(install_kind: str, root: Path, wheel: Path, extras: tuple[str, ...]) -> str:
    suffix = f"[{','.join(extras)}]" if extras else ""
    if install_kind == "source":
        return f"{PROJECT_NAME}{suffix} @ {root.as_uri()}"
    return f"{PROJECT_NAME}{suffix} @ {wheel.as_uri()}"


def probe_script(extras: tuple[str, ...], standard_asr_commit: str) -> str:
    return f"""
import importlib.metadata as metadata
import json
import sys

expected_extras = {list(extras)!r}
expected_models = {list(MODEL_IDS)!r}
converter_only_distributions = {sorted(CONVERTER_ONLY_DISTRIBUTIONS)!r}
installed = {{
    dist.metadata['Name'].lower().replace('_', '-').replace('.', '-'): dist.version
    for dist in metadata.distributions()
    if dist.metadata.get('Name')
}}
distribution = metadata.distribution('{PROJECT_NAME}')
assert distribution.version == '{PROJECT_VERSION}'
standard_asr_distribution = metadata.distribution('standard-asr')
standard_asr_direct_url = json.loads(standard_asr_distribution.read_text('direct_url.json'))
resolved_standard_asr_commit = standard_asr_direct_url['vcs_info']['commit_id']
assert resolved_standard_asr_commit == '{standard_asr_commit}'
entrypoints = {{ep.name: ep.value for ep in distribution.entry_points}}
assert entrypoints['standard-asr'] == 'standard_asr.toolchain.cli:main'
assert entrypoints['qwen3-asr-ane'] == 'std_qwen3asr_ane.cli:main'
models = {{ep.name: ep.value for ep in distribution.entry_points if ep.group == 'standard_asr.models'}}
assert sorted(models) == sorted(expected_models)

import coremltools
import numpy
import scipy
import soundfile
import standard_asr
import std_qwen3asr_ane
import tokenizers

if expected_extras in ([], ['server']):
    unexpected = sorted(set(converter_only_distributions) & set(installed))
    assert not unexpected, f'converter distributions leaked into {{expected_extras or "base"}}: {{unexpected}}'
if 'server' in expected_extras:
    import fastapi
    import multipart
    import uvicorn
    import websockets
if 'diarization' in expected_extras:
    import sherpa_onnx
    assert installed.get('sherpa-onnx') == '1.13.8'
    assert installed.get('sherpa-onnx-core') == '1.13.8'
else:
    assert 'sherpa-onnx' not in installed
    assert 'sherpa-onnx-core' not in installed
if 'gpu-draft' in expected_extras:
    import mlx.core
    from mlx_audio.stt import load as mlx_audio_load
    assert callable(mlx_audio_load)
else:
    assert 'mlx-audio' not in installed

print(json.dumps({{
    'python': platform_python if (platform_python := sys.version.split()[0]) else None,
    'project_version': distribution.version,
    'entrypoints': entrypoints,
    'installed': dict(sorted(installed.items())),
    'extras': expected_extras,
    'standard_asr_vcs_commit': resolved_standard_asr_commit,
}}, sort_keys=True))
"""


def verify_cell(
    *,
    root: Path,
    run_dir: Path,
    cache_dir: Path,
    wheel: Path,
    install_kind: str,
    python: str,
    extras: tuple[str, ...],
    standard_asr_commit: str,
) -> dict[str, Any]:
    name = f"{install_kind}-py{python}-{profile_name(extras)}"
    cell_dir = run_dir / name
    cell_dir.mkdir(parents=True)
    environment_dir = cell_dir / "environment"
    model_dir = cell_dir / "must-not-be-created"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "STANDARD_ASR_ALLOW_DOWNLOAD": "0",
            "STANDARD_ASR_MODEL_DIR": str(model_dir),
            "UV_CACHE_DIR": str(cache_dir),
        }
    )
    runner = CellRunner(cell_dir, env)
    cell: dict[str, Any] = {
        "name": name,
        "install_kind": install_kind,
        "python_requested": python,
        "profile": profile_name(extras),
        "extras": list(extras),
        "passed": False,
    }
    try:
        runner.run(["uv", "venv", "--python", python, str(environment_dir)], cwd=root)
        interpreter = environment_dir / "bin" / "python"
        runner.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                requirement_for(install_kind, root, wheel, extras),
            ],
            cwd=root,
        )
        probe = runner.run(
            [str(interpreter), "-c", probe_script(extras, standard_asr_commit)], cwd=cell_dir
        )
        probe_data = json.loads(probe.stdout)
        cli = environment_dir / "bin" / "standard-asr"
        listed = runner.run([str(cli), "list"], cwd=cell_dir)
        for model_id in MODEL_IDS:
            if model_id not in listed.stdout:
                raise AssertionError(f"{model_id!r} missing from standard-asr list")
        compliance = runner.run([str(cli), "compliance", "run"], cwd=cell_dir)
        if "Compliance run passed" not in compliance.stdout:
            raise AssertionError("standard-asr compliance did not report a passing run")
        readiness: dict[str, Any] = {}
        for model_id in MODEL_IDS:
            status = runner.run([str(cli), "status", model_id, "--json"], cwd=cell_dir)
            report = json.loads(status.stdout)
            if report.get("readiness") != "unavailable":
                raise AssertionError(f"Unexpected readiness for {model_id}: {report!r}")
            blockers = {
                requirement.get("acquisition_blocker")
                for requirement in report.get("requirements", [])
                if requirement.get("required_for_inference")
            }
            if blockers != {"downloads_disabled"}:
                raise AssertionError(f"Unexpected offline blocker for {model_id}: {blockers!r}")
            readiness[model_id] = {
                "readiness": report["readiness"],
                "required_blockers": sorted(blockers),
            }
        if model_dir.exists():
            raise AssertionError(f"Read-only commands created the model root: {model_dir}")
        cell.update(
            {
                "passed": True,
                "python_installed": probe_data["python"],
                "project_version": probe_data["project_version"],
                "package_count": len(probe_data["installed"]),
                "packages": probe_data["installed"],
                "entrypoints": probe_data["entrypoints"],
                "standard_asr_vcs_commit": probe_data["standard_asr_vcs_commit"],
                "readiness": readiness,
            }
        )
    except Exception as error:  # noqa: BLE001 - preserve failed-cell evidence
        cell["error"] = f"{type(error).__name__}: {error}"
    finally:
        cell["commands"] = [vars(command) for command in runner.commands]
        (cell_dir / "result.json").write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    return cell


def inspect_archive(path: Path, root: Path) -> dict[str, Any]:
    module_root = root / "std_qwen3asr_ane" / "src"
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    source_entries = {
        source.relative_to(module_root).as_posix(): source.read_bytes()
        for source in module_root.rglob("*.py")
    }
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = sorted(archive.namelist())
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = archive.read(metadata_name).decode()
            archived_sources = {
                name: archive.read(name)
                for name in names
                if name.startswith("std_qwen3asr_ane/") and name.endswith(".py")
            }
        required_suffixes = (
            "std_qwen3asr_ane/conversion_worker.py",
            ".dist-info/METADATA",
            ".dist-info/entry_points.txt",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/std_qwen3asr_ane/NOTICE",
        )
    else:
        with tarfile.open(path, "r:gz") as archive:
            names = sorted(member.name for member in archive.getmembers() if member.isfile())
            metadata_name = next(name for name in names if name.endswith("/PKG-INFO"))
            member = archive.getmember(metadata_name)
            stream = archive.extractfile(member)
            assert stream is not None
            metadata = stream.read().decode()
            prefix = metadata_name.removesuffix("PKG-INFO")
            archived_sources = {}
            for source_name in source_entries:
                name = f"{prefix}std_qwen3asr_ane/src/{source_name}"
                source_member = archive.getmember(name)
                source_stream = archive.extractfile(source_member)
                assert source_stream is not None
                archived_sources[source_name] = source_stream.read()
        required_suffixes = (
            "/pyproject.toml",
            "/README.md",
            "/LICENSE",
            "/std_qwen3asr_ane/NOTICE",
            "/std_qwen3asr_ane/src/std_qwen3asr_ane/conversion_worker.py",
            "/PKG-INFO",
        )
    if archived_sources != source_entries:
        changed = sorted(
            name
            for name in set(archived_sources) | set(source_entries)
            if archived_sources.get(name) != source_entries.get(name)
        )
        raise AssertionError(f"Archive source does not match checkout: {changed[:10]}")
    missing_content = [
        suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)
    ]
    if missing_content:
        raise AssertionError(f"Archive is missing required content: {missing_content}")
    forbidden_parts = ("/.cache/", "/artifacts/", "/references/", "/.git/")
    leaked = [name for name in names if any(part in f"/{name}" for part in forbidden_parts)]
    if leaked:
        raise AssertionError(f"Archive contains forbidden workspace content: {leaked[:10]}")
    message = email.parser.Parser().parsestr(metadata)
    requires_python = (message["Requires-Python"] or "").replace(" ", "")
    expected_python = project["requires-python"].replace(" ", "")
    if requires_python != expected_python:
        raise AssertionError(f"Unexpected Requires-Python: {message['Requires-Python']!r}")
    extras = set(message.get_all("Provides-Extra", []))
    optional_dependencies = project["optional-dependencies"]
    if extras != set(optional_dependencies) or extras != set(EXTRAS):
        raise AssertionError(f"Unexpected extras: {sorted(extras)!r}")
    expected_diarization = {"sherpa-onnx==1.13.8", "sherpa-onnx-core==1.13.8"}
    if set(optional_dependencies["diarization"]) != expected_diarization:
        raise AssertionError(
            "The diarization extra must retain both sherpa-onnx and its separately "
            "packaged sherpa-onnx-core runtime at 1.13.8"
        )
    normalized_requirements = {
        requirement.replace(" ", "").replace("'", '"')
        for requirement in message.get_all("Requires-Dist", [])
    }
    required_requirements = [requirement.replace(" ", "") for requirement in project["dependencies"]]
    required_requirements.extend(
        f'{requirement.replace(" ", "")};extra=="{extra}"'
        for extra, requirements in optional_dependencies.items()
        for requirement in requirements
    )
    missing = [requirement for requirement in required_requirements if requirement not in normalized_requirements]
    if missing:
        raise AssertionError(f"Archive metadata is missing: {missing}")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "file_count": len(names),
        "metadata_path": metadata_name,
        "forbidden_workspace_entries": leaked,
        "required_content_present": list(required_suffixes),
        "source_file_count": len(source_entries),
        "source_tree_sha256": source_digest(source_entries),
        "requires_python": project["requires-python"],
        "base_dependencies": project["dependencies"],
        "optional_dependencies": optional_dependencies,
    }


def portable_summary(detailed: dict[str, Any]) -> dict[str, Any]:
    selected_distribution_names = {
        "coremltools",
        "fastapi",
        "mlx",
        "mlx-audio",
        "numpy",
        "sherpa-onnx",
        "sherpa-onnx-core",
        "standard-asr",
        "std-qwen3asr-ane",
        "tokenizers",
        "transformers",
    }
    cells = []
    for cell in detailed["cells"]:
        portable_cell = {
                key: cell[key]
                for key in (
                    "name",
                    "install_kind",
                    "python_requested",
                    "python_installed",
                    "profile",
                    "extras",
                    "passed",
                    "project_version",
                    "package_count",
                    "readiness",
                    "standard_asr_vcs_commit",
                    "error",
                )
                if key in cell
            }
        if "packages" in cell:
            portable_cell["selected_versions"] = {
                name: version
                for name, version in cell["packages"].items()
                if name in selected_distribution_names
            }
        cells.append(portable_cell)
    return {
        "schema_version": 1,
        "verifier": {
            "path": "research/release-readiness/verify_install_matrix.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "generated_date": detailed["generated_date"],
        "project_commit": detailed["project_commit"],
        "worktree_dirty": detailed["worktree_dirty"],
        "project_version": PROJECT_VERSION,
        "platform": detailed["platform"],
        "uv_version": detailed["uv_version"],
        "archives": detailed["archives"],
        "matrix": {
            "passed": all(cell["passed"] for cell in cells),
            "passed_cells": sum(cell["passed"] for cell in cells),
            "total_cells": len(cells),
            "cells": cells,
        },
        "contracts": {
            "source_tree_base_installs": ["3.12", "3.13"],
            "wheel_profiles": [profile_name(profile) for profile in PROFILES],
            "offline_readiness": "unavailable with downloads_disabled; model root remains absent",
            "base_and_server_converter_only_distributions_absent": sorted(
                CONVERTER_ONLY_DISTRIBUTIONS
            ),
            "base_runtime_transitive": {
                "distribution": "huggingface-hub",
                "required_by": "tokenizers",
                "reason": "tokenizers is a direct runtime dependency",
            },
            "diarization_runtime": "sherpa-onnx and sherpa-onnx-core pinned together at 1.13.8",
            "gpu_and_diarization_coexistence": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--portable-output", type=Path)
    parser.add_argument("--python", action="append", choices=("3.12", "3.13"))
    parser.add_argument("--wheel-profile", action="append", choices=tuple(profile_name(p) for p in PROFILES))
    parser.add_argument("--skip-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    wheel = args.wheel.resolve()
    sdist = args.sdist.resolve()
    cache_dir = (args.cache_dir or root / ".cache" / "uv").resolve()
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse an existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    pythons = args.python or ["3.12", "3.13"]
    profile_lookup = {profile_name(profile): profile for profile in PROFILES}
    wheel_profiles = [profile_lookup[name] for name in args.wheel_profile] if args.wheel_profile else list(PROFILES)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    worktree_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    standard_asr_requirement = next(
        requirement
        for requirement in project["dependencies"]
        if requirement.startswith("standard-asr @ git+")
    )
    standard_asr_commit = standard_asr_requirement.rsplit("@", 1)[1]
    uv_version = subprocess.run(
        ["uv", "--version"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    detailed: dict[str, Any] = {
        "schema_version": 1,
        "generated_date": evidence_date(),
        "project_commit": commit,
        "worktree_dirty": worktree_dirty,
        "project_version": PROJECT_VERSION,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "uv_version": uv_version,
        "archives": [inspect_archive(wheel, root), inspect_archive(sdist, root)],
        "cells": [],
    }
    if not args.skip_source:
        for python in pythons:
            detailed["cells"].append(
                verify_cell(
                    root=root,
                    run_dir=run_dir,
                    cache_dir=cache_dir,
                    wheel=wheel,
                    install_kind="source",
                    python=python,
                    extras=(),
                    standard_asr_commit=standard_asr_commit,
                )
            )
    for python in pythons:
        for extras in wheel_profiles:
            detailed["cells"].append(
                verify_cell(
                    root=root,
                    run_dir=run_dir,
                    cache_dir=cache_dir,
                    wheel=wheel,
                    install_kind="wheel",
                    python=python,
                    extras=extras,
                    standard_asr_commit=standard_asr_commit,
                )
            )
    detailed["passed"] = all(cell["passed"] for cell in detailed["cells"])
    detailed_path = run_dir / "results.json"
    detailed_path.write_text(json.dumps(detailed, indent=2, sort_keys=True) + "\n")
    if args.portable_output:
        portable_output = args.portable_output.resolve()
        portable_output.parent.mkdir(parents=True, exist_ok=True)
        portable_output.write_text(
            json.dumps(portable_summary(detailed), indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "passed": detailed["passed"],
                "passed_cells": sum(cell["passed"] for cell in detailed["cells"]),
                "total_cells": len(detailed["cells"]),
                "results": str(detailed_path),
            },
            sort_keys=True,
        )
    )
    return 0 if detailed["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
