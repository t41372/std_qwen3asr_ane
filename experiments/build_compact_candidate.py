"""Build an immutable serial T1 compact-head candidate bound to target weights."""

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
from std_qwen3asr_ane.bundle import clone, digest, validate_bundle_paths
from std_qwen3asr_ane.conversion.compress import compress_model
from std_qwen3asr_ane.conversion.draft import build_compact_head
from std_qwen3asr_ane.draft import weight_digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target, source, output = (
        args.target.resolve(),
        args.source.resolve(),
        args.output.resolve(),
    )
    if output.exists():
        parser.error("Use a fresh output")
    manifest = json.loads((target / "manifest.json").read_text())
    provenance = json.loads((source / "source.json").read_text())
    if (
        provenance.get("model_id") != manifest["model_id"]
        or provenance.get("revision") != manifest["source_revision"]
    ):
        raise ValueError("Checkpoint must match the target model and revision")
    validate_bundle_paths(
        target, set(manifest["files"].values()) | set(manifest["decoder_partitions"])
    )
    clone(target, output)
    (output / "manifest.json").unlink()
    fp16 = output / "compact_head_fp16.mlpackage"
    metadata = build_compact_head(
        source, fp16, token_batch_size=1, residual_scale=manifest["residual_scale"]
    )
    compression = manifest.get("weight_compression")
    package = fp16
    if compression and "lm_head" in compression["roles"]:
        package = output / "compact_head.mlpackage"
        compress_model(
            fp16,
            package,
            compression["scheme"],
            compression["bits"],
            compression["group_size"],
        )
    compiled = output / "compact_head.mlmodelc"
    ct.models.utils.compile_model(str(package), destination_path=str(compiled))
    original_head = target / manifest["files"]["lm_head"]
    if weight_digests(compiled) != weight_digests(original_head):
        raise RuntimeError(
            "Compact head weights differ from the target; refusing the candidate"
        )
    for intermediate in {fp16, package}:
        shutil.rmtree(intermediate)
    manifest["files"]["lm_head"] = compiled.name
    manifest["schema_version"] = 2
    manifest["head_output"] = {"kind": "chunk_max", "token_batch_size": 1, **metadata}
    manifest["validation_status"] = "unvalidated"
    manifest["compact_head_parent"] = {
        "manifest_sha256": digest(target / "manifest.json"),
        "head_weight_sha256": weight_digests(original_head),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "head": manifest["head_output"]}))


if __name__ == "__main__":
    main()
