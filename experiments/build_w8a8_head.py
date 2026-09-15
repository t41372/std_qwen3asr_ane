"""Calibrate head inputs, preserving FP16 normalization and output logits.

Coremltools 9's intermediate-range collector overwrites previous extrema. This
experiment avoids that collector: the only quantized activation is the shared
normalized input to the vocabulary projections. Its range is measured across
all captured real hidden rows with the source's FP16 StableRMSNorm. Weight
quantization uses the supported per-channel API; MIL adds one shared Q/DQ pair.
This is a calibrated candidate, never an exact-output claim.
"""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as optimization
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from std_qwen3asr_ane.bundle import clone, digest
from std_qwen3asr_ane.conversion.compress import compress_model
from std_qwen3asr_ane.conversion.decoder import SourceWeights, StableRMSNorm
from std_qwen3asr_ane.conversion.passes import verify_activation_operators
from std_qwen3asr_ane.draft import weight_digests


def quantize_projection_inputs(program, scale):
    """Quantize only conv inputs; other consumers of each FP16 value are intact."""
    scale = np.float16(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Activation scale must be positive and finite in FP16")
    function = program.functions["main"]
    projections = [
        operation for operation in function.operations if operation.op_type == "conv"
    ]
    if not projections:
        raise ValueError("The head has no vocabulary projections")
    shared = {}
    with function:
        for operation in projections:
            value = operation.x
            if value.dtype != types.fp16:
                raise ValueError(
                    "Head activation calibration expects FP16 projection inputs"
                )
            if value.name not in shared:
                quantized = mb.quantize(
                    input=value, scale=scale, output_dtype="int8", before_op=operation
                )
                shared[value.name] = mb.dequantize(
                    input=quantized, scale=scale, before_op=operation
                )
            operation.set_inputs(x=shared[value.name])
    return len(shared)


def calibration_scale(source, manifest, archive):
    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    with np.load(archive, allow_pickle=False) as data:
        hidden = np.array(data["hidden_states"], copy=True)
    if (
        hidden.ndim != 2
        or hidden.shape[1] != config["hidden_size"]
        or not np.isfinite(hidden).all()
    ):
        raise ValueError("Invalid captured head inputs")
    norm = StableRMSNorm(
        config["hidden_size"], config["rms_norm_eps"], manifest["residual_scale"]
    )
    norm.weight.data.copy_(SourceWeights(source).get("thinker.model.norm.weight"))
    norm = norm.half().eval()
    with torch.inference_mode():
        normalized = (
            norm(torch.from_numpy(hidden).half()[:, :, None, None]).float().numpy()
        )
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced non-finite calibration values")
    minimum, maximum = float(normalized.min()), float(normalized.max())
    scale = np.float16(max(abs(minimum), abs(maximum)) / 127)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration has no usable activation range")
    return scale, {"minimum": minimum, "maximum": maximum, "rows": len(hidden)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--fp16-head", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    torch.set_num_threads(4)
    manifest = json.loads((args.target / "manifest.json").read_text())
    provenance = json.loads((args.source / "source.json").read_text())
    if provenance != {
        "model_id": manifest["model_id"],
        "revision": manifest["source_revision"],
    }:
        raise ValueError("Source revision must match target")
    capture = json.loads(args.calibration.with_suffix(".json").read_text())
    if not capture.get("complete") or capture.get("input_archive_sha256") != digest(
        args.calibration
    ):
        raise ValueError("Calibration capture is incomplete or changed")
    if capture["model_manifest_sha256"] != digest(args.target / "manifest.json"):
        raise ValueError("Calibration capture belongs to a different target")
    head_name = Path(manifest["files"]["lm_head"]).stem
    source_record = next(
        row
        for row in manifest["weight_compression"]["compressed_files"]
        if Path(row["file"]).stem == head_name
    )
    actual_source = {
        str(path.relative_to(args.fp16_head)): digest(path)
        for path in sorted(args.fp16_head.rglob("*"))
        if path.is_file()
    }
    if "source_sha256" in source_record:
        if actual_source != source_record["source_sha256"]:
            raise ValueError(
                "FP16 head differs from the target's recorded compression source"
            )
    else:
        # Historical bundles lack per-source payload hashes. Reproduce their
        # compression and compare actual head weights before using this source.
        compression = manifest["weight_compression"]
        if (
            digest(args.fp16_head.parent / "manifest.json")
            != compression["parent_manifest_sha256"]
        ):
            raise ValueError("FP16 head parent differs from the recorded source")
        with tempfile.TemporaryDirectory(prefix="head-source-check-") as temporary:
            reference = Path(temporary) / "reference.mlpackage"
            compress_model(
                args.fp16_head,
                reference,
                compression["scheme"],
                compression["bits"],
                compression["group_size"],
            )
            if weight_digests(reference) != weight_digests(
                args.target / manifest["files"]["lm_head"]
            ):
                raise ValueError(
                    "Recompressed FP16 source does not reproduce the target head"
                )
    scale, ranges = calibration_scale(args.source, manifest, args.calibration)
    original = ct.models.MLModel(
        str(args.fp16_head),
        skip_model_load=True,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    quantized = optimization.linear_quantize_weights(
        original,
        config=optimization.OptimizationConfig(
            op_type_configs={
                "conv": optimization.OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype="int8", granularity="per_channel"
                )
            }
        ),
    )
    # The conversion toolchain owns this MIL program. It is used only while
    # authoring the package, never as a private device-execution interface.
    program = quantized._mil_program
    if program is None:
        raise RuntimeError(
            "The installed conversion toolchain did not retain its MIL program"
        )
    groups = quantize_projection_inputs(program, scale)
    model = ct.convert(
        program,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    verify_activation_operators(model)
    clone(args.target, args.output)
    (args.output / "manifest.json").unlink()
    package = args.output / "w8a8_head.mlpackage"
    compiled = args.output / "w8a8_head.mlmodelc"
    model.save(str(package))
    ct.models.utils.compile_model(str(package), destination_path=str(compiled))
    operations = Counter(op.op_type for op in program.functions["main"].operations)
    old_head = manifest["files"]["lm_head"]
    manifest["files"]["lm_head"] = compiled.name
    manifest["schema_version"] = 2
    manifest["head_output"] = {"kind": "logits", "token_batch_size": 1}
    manifest["validation_status"] = "unvalidated"
    compression = manifest.get("weight_compression", {})
    compression["roles"] = [
        role for role in compression.get("roles", []) if role != "lm_head"
    ]
    compression["compressed_files"] = [
        row
        for row in compression.get("compressed_files", [])
        if Path(row["file"]).stem != Path(old_head).stem
    ]
    manifest["head_quantization"] = {
        "scheme": "linear_symmetric_w8a8",
        "weight_granularity": "per_channel",
        "activation_scale": float(scale),
        "activation_ranges": ranges,
        "calibration_archive_sha256": digest(args.calibration),
        "calibration_manifest_sha256": capture["calibration_manifest_sha256"],
        "calibration_method": "FP16 source RMSNorm across all real captured head inputs",
        "quantized_input_groups": groups,
        "operations": dict(operations),
        "normalization_and_logits_dtype": "fp16",
        "parent_manifest_sha256": digest(args.target / "manifest.json"),
        "fp16_head_source_sha256": actual_source,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Retain the source package for inspecting quantized compute, not just loadability.
    print(
        json.dumps(
            {"output": str(args.output), "quantization": manifest["head_quantization"]}
        )
    )


if __name__ == "__main__":
    main()
