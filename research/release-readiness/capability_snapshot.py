"""Record the full public capability surface without acquiring or loading models.

Run with the installed plugin and the checked-out Standard ASR revision that the
package pins. Synthetic head descriptors exercise configuration narrowing only;
they are not model-readiness or inference evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

import standard_asr

from std_qwen3asr_ane.plugin import Qwen3ASREngine, ShortDictationEngine

ROOT = Path(__file__).resolve().parents[2]


def git_output(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def source_inventory(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.py"))
    }


def query_surface(capabilities) -> dict[str, bool]:
    return {
        path: capabilities.supports(path)
        for path in sorted(capabilities.iter_queryable_paths())
    }


def model_snapshot(engine_type, root: Path) -> dict:
    declared = engine_type.declared_capabilities
    configurations = {}
    for head in ("logits", "chunk_max"):
        model_dir = root / head
        model_dir.mkdir(exist_ok=True)
        descriptor = {"kind": head, "token_batch_size": 1}
        if head == "chunk_max":
            descriptor["vocabulary_chunk"] = 4096
        (model_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 2, "head_output": descriptor})
        )
        for auxiliary in ("disabled", "alignment", "diarization"):
            engine = engine_type(
                model_dir=model_dir,
                use_alignment=auxiliary == "alignment",
                use_diarization=auxiliary == "diarization",
                use_draft=False,
                draft_dir=None,
                use_batching=False,
                batch_head_dir=None,
            )
            try:
                effective = engine.effective_capabilities
                if not declared.covers(effective):
                    raise AssertionError("Effective capabilities exceed the declaration")
                if engine._runtime is not None or engine._auxiliary is not None:
                    raise AssertionError("Capability inspection initialized a model owner")
                configurations[f"{head}/{auxiliary}"] = query_surface(effective)
            finally:
                engine.close()
    return {
        "properties": engine_type.properties.model_dump(mode="json"),
        "declared_capabilities": declared.canonical_json(),
        "declared_query_surface": query_surface(declared),
        "effective_query_surfaces": configurations,
        "metadata": engine_type.declared_metadata.canonical_json(),
        "config_schema": engine_type.config_type.model_json_schema(),
        "provider_params_schema": engine_type.provider_params_type.model_json_schema(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream_sources = source_inventory(args.upstream / "src/standard_asr")
    installed_sources = source_inventory(Path(standard_asr.__file__).parent)
    differences = sorted(
        name
        for name in upstream_sources.keys() | installed_sources.keys()
        if upstream_sources.get(name) != installed_sources.get(name)
    )
    if differences:
        raise RuntimeError(f"Installed Standard ASR differs from the checkout: {differences}")
    with tempfile.TemporaryDirectory(prefix="qwen-capabilities-") as directory:
        models = {
            engine_type.properties.model_id: model_snapshot(engine_type, Path(directory))
            for engine_type in (Qwen3ASREngine, ShortDictationEngine)
        }
    report = {
        "scope": "Static declarations and synthetic configuration narrowing; no inference",
        "plugin_revision": git_output(ROOT, "rev-parse", "HEAD"),
        "plugin_version": importlib.metadata.version("std-qwen3asr-ane"),
        "plugin_python_sha256": source_inventory(ROOT / "std_qwen3asr_ane/src"),
        "upstream_revision": git_output(args.upstream, "rev-parse", "HEAD"),
        "upstream_python_sha256": upstream_sources,
        "installed_code_differences": differences,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
