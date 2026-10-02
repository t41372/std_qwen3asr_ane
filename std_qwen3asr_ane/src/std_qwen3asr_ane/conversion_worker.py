# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "standard-asr @ git+https://github.com/standard-voice/standard_asr.git@cad09d412479937f7fdbede0e0c54437bef25ef7",
#   "coremltools==9.0", "torch==2.14.0", "numpy==2.5.3",
#   "transformers==4.57.6", "huggingface-hub==0.36.2",
#   "safetensors==0.8.0", "scipy==1.18.1", "tokenizers==0.22.2",
# ]
# ///
"""Private pull worker, shipped in the wheel; not a user-facing install step."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    # Execute exactly the package that launched us, including unpublished local
    # builds. No repository checkout or reinstall of the plugin is needed.
    package = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "std_qwen3asr_ane", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from standard_asr.contract.exceptions import ArtifactAcquisitionError
    from standard_asr.runtime.redaction import log_exception_safely

    from std_qwen3asr_ane.acquisition import AcquisitionFailure
    from std_qwen3asr_ane.artifact_lifecycle import (
        BATCH_HEAD_ARTIFACT_ID,
        BUNDLE_ARTIFACT_ID,
        DRAFT_ARTIFACT_ID,
        ArtifactManager,
    )

    request = json.loads(sys.stdin.readline())
    output = sys.stdout

    def emit(phase: str, artifact_id: str, **counts: object) -> None:
        frame = {"phase": phase, "artifact_id": artifact_id, **counts}
        print(json.dumps(frame), file=output, flush=True)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            raw_config = request["config"]
            config = SimpleNamespace(
                profile=raw_config["profile"],
                model_dir=Path(raw_config["model_dir"]),
                source_dir=Path(raw_config["source_dir"]),
                draft_dir=(
                    Path(raw_config["draft_dir"])
                    if raw_config.get("draft_dir") is not None
                    else None
                ),
                draft_source_dir=Path(raw_config["draft_source_dir"]),
                use_batching=bool(raw_config.get("use_batching", False)),
                batch_head_dir=(
                    Path(raw_config["batch_head_dir"])
                    if raw_config.get("batch_head_dir") is not None
                    else None
                ),
            )
            model_name = "1.7b-short-dictation" if config.profile == "short-dictation" else "1.7b"
            manager = ArtifactManager(config, f"std-qwen3asr-ane/{model_name}")
            if BUNDLE_ARTIFACT_ID in request["targets"]:
                manager.acquire_bundle(emit)
            if BATCH_HEAD_ARTIFACT_ID in request["targets"]:
                manager.acquire_batch_head(emit)
            if DRAFT_ARTIFACT_ID in request["targets"]:
                manager.acquire_draft(emit)
    except ArtifactAcquisitionError as error:
        failure = AcquisitionFailure.from_exception(error)
    except Exception:  # noqa: BLE001 - SDK failures must cross the process boundary as data.
        log_exception_safely(logging.getLogger(__name__), "Conversion failed")
        failure = AcquisitionFailure(
            reason="failed", message="The conversion worker failed; see its stderr output."
        )
    else:
        return 0
    print(json.dumps({"error": failure.model_dump(mode="json")}), file=output, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
