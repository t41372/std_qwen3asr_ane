"""Compare selective W8A8 with LUT8 on a real four-layer generation snapshot.

Q/K/V and MLP gate/up projections get per-channel W8 weights and calibrated A8
inputs. Attention output and MLP down projections retain LUT8; all nonlinear
math and KV state remain FP16. This is a bounded feasibility probe, not ASR
quality evidence or a deployable bundle.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import coremltools as ct
import coremltools.optimize.coreml as optimization
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.compress import (
    compress_model,
    compression_config,
    weight_bytes,
)
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    convert_partition,
    load_partition,
)
from std_qwen3asr_ane.conversion.passes import verify_activation_operators
from std_qwen3asr_ane.diagnostics import inspect_compute_plan
from std_qwen3asr_ane.runtime import PersistentInputModel

SELECTED = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}


def projection_id(operation):
    match = re.match(
        r"layers_(\d+)_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)_weight",
        operation.weight.name,
    )
    if match is None:
        raise ValueError(f"Unrecognized decoder projection: {operation.weight.name}")
    return f"layers_{match[1]}_{match[2]}", match[2]


def build_candidate(source_package, output, calibration):
    original = ct.models.MLModel(
        str(source_package),
        skip_model_load=True,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    # Obtain the SDK's authoring IR without changing the source weight values.
    original = optimization.decompress_weights(original)
    projections = [
        op
        for op in original._mil_program.functions["main"].operations
        if op.op_type == "conv"
    ]
    selected_names = {op.name for op in projections if projection_id(op)[1] in SELECTED}
    palette = compression_config("palette", 8, 32)
    for name in selected_names:
        palette.set_op_name(name, None)
    mixed = optimization.palettize_weights(original, config=palette)
    linear = optimization.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"
    )
    mixed = optimization.linear_quantize_weights(
        mixed,
        config=optimization.OptimizationConfig(
            op_name_configs={name: linear for name in selected_names}
        ),
    )
    program = mixed._mil_program
    function = program.functions["main"]
    projections = [op for op in function.operations if op.op_type == "conv"]
    shared, scales = {}, {}
    with function:
        for operation in projections:
            logical, kind = projection_id(operation)
            if kind not in SELECTED:
                continue
            tensor = calibration["projections"][logical]["tensor"]
            bounds = calibration["ranges"][tensor]
            if operation.x.shape[1] != bounds["channels"]:
                raise ValueError(f"Calibration channels differ for {logical}")
            scale = np.float16(
                max(abs(bounds["minimum"]), abs(bounds["maximum"])) / 127
            )
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid calibration scale for {logical}")
            value = operation.x
            if value.name not in shared:
                q = mb.quantize(
                    input=value, scale=scale, output_dtype="int8", before_op=operation
                )
                shared[value.name] = (
                    scale,
                    mb.dequantize(input=q, scale=scale, before_op=operation),
                )
            if shared[value.name][0] != scale:
                raise ValueError(
                    "Shared projection input has inconsistent calibration ranges"
                )
            operation.set_inputs(x=shared[value.name][1])
            scales[logical] = float(scale)
    model = ct.convert(
        program,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    verify_activation_operators(model)
    model.save(str(output))
    counts = Counter(op.op_type for op in program.functions["main"].operations)
    if counts["quantize"] != calibration["layers"] * 2:
        raise RuntimeError("A8 projection-input groups were lost during conversion")
    return {"activation_scales": scales, "operations": dict(counts)}


def benchmark(paths, example):
    with np.load(example, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    inputs = {
        name: value for name, value in arrays.items() if not name.startswith("state__")
    }
    models, states, durations, last = {}, {}, {name: [] for name in paths}, {}
    state = None
    try:
        for name, path in paths.items():
            models[name] = PersistentInputModel(
                ct.models.CompiledMLModel(
                    str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
                )
            )
            state = models[name].make_state()
            for key, value in arrays.items():
                if key.startswith("state__"):
                    state.write_state(key.removeprefix("state__"), value)
            states[name] = state
        state = None
        for repeat in range(40):
            order = tuple(paths) if repeat % 2 == 0 else tuple(reversed(paths))
            for name in order:
                started = perf_counter()
                result = models[name].predict(inputs, state=states[name])[
                    "output_hidden_states"
                ]
                elapsed = perf_counter() - started
                if not np.isfinite(result).all():
                    raise RuntimeError("Decoder produced non-finite hidden states")
                last[name] = result
                if repeat >= 10:
                    durations[name].append(elapsed)
        return {
            "timings": {
                name: {
                    "median_ms": float(np.median(times) * 1000),
                    "p95_ms": float(np.percentile(times, 95) * 1000),
                }
                for name, times in durations.items()
            },
            "max_abs_hidden_difference": float(
                np.max(
                    np.abs(
                        last["baseline"].astype(np.float32)
                        - last["candidate"].astype(np.float32)
                    )
                )
            ),
            "valid_tokens": int((inputs["update_mask"].sum(axis=-1) != 0).sum()),
            "finite": True,
        }
    finally:
        states.clear()
        state = None
        PersistentInputModel.close_many(list(models.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    calibration = json.loads((args.calibration / "ranges.json").read_text())
    if not calibration.get("complete") or calibration[
        "target_manifest_sha256"
    ] != digest(args.target / "manifest.json"):
        raise ValueError("Calibration must be complete and bound to this target")
    example = args.calibration / "generation-example.npz"
    if digest(example) != calibration["example"]["archive_sha256"]:
        raise ValueError("Generation example changed")
    target = json.loads((args.target / "manifest.json").read_text())
    config = json.loads((args.source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    report = {
        "complete": False,
        "calibration_sha256": digest(args.calibration / "ranges.json"),
    }
    try:
        module = DecoderPartition(
            config,
            calibration["layers"],
            target["max_sequence_length"],
            residual_scale=target["residual_scale"],
            token_batch_size=target["token_batch_size"],
        )
        load_partition(module, SourceWeights(args.source), 0)
        fp16 = args.output / "fp16.mlpackage"
        convert_partition(module, config, fp16, target["max_sequence_length"])
        del module
        baseline = args.output / "lut8.mlpackage"
        compress_model(fp16, baseline, "palette", 8, 32)
        candidate = args.output / "w8a8.mlpackage"
        report.update(build_candidate(fp16, candidate, calibration))
        paths = {}
        for name, package in (("baseline", baseline), ("candidate", candidate)):
            paths[name] = package.with_suffix(".mlmodelc")
            ct.models.utils.compile_model(
                str(package), destination_path=str(paths[name])
            )
        report.update(benchmark(paths, example))
        report["weight_bytes"] = {
            name: weight_bytes(path) for name, path in paths.items()
        }
        placement = inspect_compute_plan(paths["candidate"])
        (args.output / "placement.json").write_text(
            json.dumps(placement, indent=2) + "\n"
        )
        report["placement_summary"] = placement["summary"]
        report["complete"] = True
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
