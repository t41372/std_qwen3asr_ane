"""Versioned Python environments for optional, dependency-isolated workers.

Inspection is deliberately read-only.  Environment creation belongs to an
explicit acquisition operation and publishes a complete environment atomically,
so inference never invokes an installer or consults the network.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RECEIPT_NAME = ".std-qwen3asr-worker.json"
_RECEIPT_SCHEMA = 1
_ENVIRONMENT_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class WorkerEnvironmentSpec:
    """Immutable identity and requirements for one managed worker runtime."""

    name: str
    version: int
    dependencies: tuple[str, ...]
    python_abi: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Worker environment name must be non-empty and unpadded")
        if self.version < 1:
            raise ValueError("Worker environment version must be positive")
        if not self.dependencies or any(
            not item or item != item.strip() for item in self.dependencies
        ):
            raise ValueError("Worker environment dependencies must be non-empty and unpadded")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("Worker environment dependencies must not contain duplicates")
        if self.python_abi is not None and (
            not self.python_abi or self.python_abi != self.python_abi.strip()
        ):
            raise ValueError("Worker environment Python ABI must be non-empty and unpadded")


@dataclass(frozen=True)
class WorkerEnvironmentInspection:
    """Read-only readiness result for a managed worker environment."""

    ready: bool
    reason: str
    python_executable: Path | None
    receipt: dict[str, Any] | None = None


def _current_abi() -> str:
    return (
        sys.implementation.cache_tag
        or f"{sys.implementation.name}-{sys.version_info.major}{sys.version_info.minor}"
    )


def _expected_receipt(spec: WorkerEnvironmentSpec) -> dict[str, Any]:
    return {
        "schema_version": _RECEIPT_SCHEMA,
        "name": spec.name,
        "version": spec.version,
        "dependencies": list(spec.dependencies),
        "python": {
            "implementation": platform.python_implementation(),
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "abi": spec.python_abi or _current_abi(),
        },
        "python_executable": "bin/python",
    }


def worker_environment_cache_key(spec: WorkerEnvironmentSpec) -> str:
    """Return a stable key for the exact dependency and parent-Python receipt."""
    encoded = json.dumps(_expected_receipt(spec), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def worker_environment_name(prefix: str, spec: WorkerEnvironmentSpec) -> str:
    """Name one immutable cache generation without consulting the filesystem."""
    if not _ENVIRONMENT_PREFIX.fullmatch(prefix):
        raise ValueError("Worker environment prefix must be a safe path component")
    return f"{prefix}-v{spec.version}-{worker_environment_cache_key(spec)}"


def select_worker_environment_root(
    parent: str | Path,
    prefix: str,
    spec: WorkerEnvironmentSpec,
    *,
    legacy_names: tuple[str, ...] = (),
) -> Path:
    """Reuse a matching legacy environment or select the immutable generation.

    Legacy inspection is read-only. A stale or corrupt legacy directory is
    intentionally left in place and cannot block the fingerprinted sibling.
    """
    parent = Path(parent).expanduser()
    for name in legacy_names:
        if not _ENVIRONMENT_PREFIX.fullmatch(name):
            raise ValueError("Legacy worker environment name must be a safe path component")
        legacy = parent / name
        if inspect_worker_environment(legacy, spec).ready:
            return legacy
    return parent / worker_environment_name(prefix, spec)


def inspect_worker_environment(
    root: str | Path, spec: WorkerEnvironmentSpec
) -> WorkerEnvironmentInspection:
    """Inspect an exact environment without creating files or starting Python."""
    root = Path(root)
    receipt_path = root / _RECEIPT_NAME
    python_executable = root / "bin" / "python"
    if not root.exists():
        return WorkerEnvironmentInspection(False, "missing", None)
    if not root.is_dir():
        return WorkerEnvironmentInspection(False, "not_a_directory", None)
    if not receipt_path.is_file():
        return WorkerEnvironmentInspection(False, "receipt_missing", None)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return WorkerEnvironmentInspection(False, "receipt_invalid", None)
    if receipt != _expected_receipt(spec):
        return WorkerEnvironmentInspection(False, "receipt_mismatch", None, receipt)
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        return WorkerEnvironmentInspection(False, "python_missing", None, receipt)
    return WorkerEnvironmentInspection(True, "ready", python_executable, receipt)


def ensure_worker_environment(
    root: str | Path,
    spec: WorkerEnvironmentSpec,
    *,
    allow_downloads: bool,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Create and atomically publish an exact worker environment with ``uv``.

    This function is intended only for an explicit artifact acquisition path.
    ``allow_downloads=False`` adds uv's offline gate and succeeds only when all
    distributions are already present in its cache.
    """
    root = Path(root).expanduser().resolve()
    if spec.python_abi is not None and spec.python_abi != _current_abi():
        raise RuntimeError(
            f"Worker requires Python ABI {spec.python_abi}; current ABI is {_current_abi()}"
        )
    ready = inspect_worker_environment(root, spec)
    if ready.ready:
        assert ready.python_executable is not None
        return ready.python_executable

    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        ready = inspect_worker_environment(root, spec)
        if ready.ready:
            assert ready.python_executable is not None
            return ready.python_executable
        if root.exists():
            raise RuntimeError(
                f"Managed worker environment exists but is not usable ({ready.reason}): {root}"
            )

        temporary = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
        try:
            from uv import find_uv_bin

            if progress is not None:
                progress("Creating isolated worker environment")
            subprocess.run(
                [find_uv_bin(), "venv", str(temporary), "--python", sys.executable],
                check=True,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            python_executable = temporary / "bin" / "python"
            command = [
                find_uv_bin(),
                "pip",
                "install",
                "--python",
                str(python_executable),
                "--strict",
            ]
            if not allow_downloads:
                command.append("--offline")
            command.extend(spec.dependencies)
            if progress is not None:
                progress("Installing pinned worker dependencies")
            subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
            receipt_path = temporary / _RECEIPT_NAME
            receipt_path.write_text(
                json.dumps(_expected_receipt(spec), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    inspection = inspect_worker_environment(root, spec)
    if not inspection.ready or inspection.python_executable is None:
        raise RuntimeError(f"Published worker environment is not usable: {inspection.reason}")
    return inspection.python_executable
