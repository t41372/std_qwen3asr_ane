"""Identify the code actually imported by opt-in release evidence tools."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import tomllib
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo


def evidence_date() -> str:
    """Use the project's release-validation timezone regardless of host settings."""
    return datetime.now(ZoneInfo("America/Phoenix")).date().isoformat()


def module_sha256(name: str) -> str:
    module = importlib.import_module(name)
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def package_identity(
    distribution_name: str, package_name: str, *, allow_source_override: bool = False
) -> dict:
    """Record distribution provenance and the complete imported Python source tree.

    Wheel installations must resolve to their installed package and retain their
    RECORD hashes. Editable installations are identified as mutable source, never
    presented as immutable Git installations. An intentional baseline override is
    allowed only when explicitly requested and remains visible in the report.
    """
    distribution = importlib.metadata.distribution(distribution_name)
    module = importlib.import_module(package_name)
    imported_init = Path(module.__file__).resolve()
    source_root = imported_init.parent
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    editable = bool((direct_url or {}).get("dir_info", {}).get("editable"))
    records = {
        str(file): file
        for file in distribution.files or ()
        if str(file).startswith(f"{package_name}/") and str(file).endswith(".py")
    }
    expected_init = records.get(f"{package_name}/__init__.py")
    if editable:
        location = urlparse(direct_url["url"])
        editable_root = Path(unquote(location.path)).resolve()
        candidates = [editable_root / package_name, editable_root / "src" / package_name]
        project_file = editable_root / "pyproject.toml"
        if project_file.is_file():
            project = tomllib.loads(project_file.read_text(encoding="utf-8"))
            module_root = (
                project.get("tool", {}).get("uv", {}).get("build-backend", {}).get("module-root")
            )
            if module_root is not None:
                candidates.append(editable_root / module_root / package_name)
        matches_installation = location.scheme == "file" and source_root in {
            path.resolve() for path in candidates
        }
    else:
        matches_installation = (
            expected_init is not None
            and imported_init == Path(distribution.locate_file(expected_init)).resolve()
        )
    if not matches_installation and not allow_source_override:
        raise RuntimeError(
            f"{package_name} imports from {imported_init}, outside its installed distribution; "
            "remove the source/PYTHONPATH override before recording release evidence"
        )

    inventory = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }
    if matches_installation and not editable:
        expected_names = {name.removeprefix(f"{package_name}/") for name in records}
        if inventory.keys() != expected_names:
            raise RuntimeError(f"{package_name} source files differ from its installed RECORD")
        for name, record in records.items():
            actual = inventory[name.removeprefix(f"{package_name}/")]
            expected = record.hash
            if expected is None or expected.mode != "sha256":
                raise RuntimeError(f"No SHA-256 installation record for {name}")
            encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
            if encoded != expected.value:
                raise RuntimeError(f"{name} differs from its installed RECORD hash")
    source_kind = "source_override"
    if matches_installation:
        source_kind = "editable" if editable else "installed"
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "direct_url": direct_url,
        "imported_init": str(imported_init),
        "source_kind": source_kind,
        "python_source_sha256": inventory,
    }


def runtime_provenance(*, allow_plugin_override: bool = False) -> dict:
    """Standard ASR must always come from the distribution being reported."""
    return {
        "standard_asr": package_identity("standard-asr", "standard_asr"),
        "plugin": package_identity(
            "std-qwen3asr-ane", "std_qwen3asr_ane", allow_source_override=allow_plugin_override
        ),
    }
