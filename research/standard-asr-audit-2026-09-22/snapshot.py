"""Record declarations and source identity without loading or acquiring models.

Run from the repository root with the existing development interpreter:
    .venv/bin/python research/standard-asr-audit-2026-09-22/snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import standard_asr

from std_qwen3asr_ane.plugin import Qwen3ASREngine, ShortDictationEngine

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "references/standard-asr-audit-2026-09-22"


def revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def source_inventory(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.py"))
    }


def capability_nodes(node: dict, prefix: str = "") -> dict[str, dict]:
    nodes = {}
    for name, child in node.items():
        if not isinstance(child, dict):
            continue
        path = f"{prefix}.{name}" if prefix else name
        if "supported" in child or "mode" in child:
            nodes[path] = child
        nodes.update(capability_nodes(child, path))
    return nodes


def main() -> None:
    upstream_files = source_inventory(UPSTREAM / "src/standard_asr")
    installed_files = source_inventory(Path(standard_asr.__file__).parent)
    models = {}
    for engine_type in (Qwen3ASREngine, ShortDictationEngine):
        capabilities = engine_type.declared_capabilities.canonical_json()
        models[engine_type.properties.model_id] = {
            "properties": engine_type.properties.model_dump(mode="json"),
            "capabilities": capabilities,
            "capability_nodes": capability_nodes(capabilities),
            "metadata": engine_type.declared_metadata.canonical_json(),
            "config_schema": engine_type.config_type.model_json_schema(),
            "provider_params_schema": engine_type.provider_params_type.model_json_schema(),
        }
    report = {
        "plugin_revision": revision(ROOT),
        "upstream_revision": revision(UPSTREAM),
        "upstream_python_sha256": upstream_files,
        "installed_code_differences": [
            name for name, digest in upstream_files.items()
            if installed_files.get(name) != digest
        ],
        "plugin_python_sha256": source_inventory(ROOT / "std_qwen3asr_ane/src"),
        "models": models,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
