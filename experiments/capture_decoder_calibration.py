"""Observe real projection inputs in a stateful target partition for W8A8.

The capture graph adds outputs to the existing LUT8 model; it does not reconstruct
KV state or run the SDK's broken intermediate-range accumulator. Only extrema
are retained, across all valid prompt/generation rows. Artifacts and utterance
identities make this calibration reproducible and separate from quality data.
"""

import argparse
import json
import re
from pathlib import Path

import coremltools as ct
import numpy as np
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def add_capture_outputs(package, output, layers):
    model = ct.models.MLModel(str(package), skip_model_load=True)
    specification = model.get_spec()
    block = next(
        iter(specification.mlProgram.functions["main"].block_specializations.values())
    )
    variables = {
        value.name: value
        for operation in block.operations
        for value in operation.outputs
    }
    taps, projections = {}, {}
    for operation in block.operations:
        if operation.type != "conv":
            continue
        weight = operation.inputs["weight"].arguments[0].name
        match = re.match(
            r"layers_(\d+)_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)_weight",
            weight,
        )
        if match is None or int(match[1]) >= layers:
            continue
        logical = f"layers_{match[1]}_{match[2]}"
        tensor = operation.inputs["x"].arguments[0].name
        projections[logical] = {"tensor": tensor, "output": operation.outputs[0].name}
        taps.setdefault(tensor, []).append(logical)
    if len(projections) != layers * 7:
        raise ValueError(
            "Capture graph does not expose the expected seven projections per layer"
        )
    for name in taps:
        tensor = variables[name].type.tensorType
        shape = [dimension.constant.size for dimension in tensor.dimensions]
        if (
            tensor.dataType != ct.proto.MIL_pb2.FLOAT16
            or len(shape) != 4
            or min(shape) < 1
        ):
            raise ValueError(f"Expected a fixed FP16 BCHW activation: {name}")
        feature = specification.description.output.add()
        feature.name = name
        feature.type.multiArrayType.dataType = (
            ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
        )
        feature.type.multiArrayType.shape.extend(shape)
        block.outputs.append(name)
    captured = ct.models.MLModel(
        specification, weights_dir=model.weights_dir, skip_model_load=True
    )
    captured.save(str(output))
    return taps, projections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.layers <= 14:
        parser.error("Use a fresh output and 1..14 layers")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.target / "manifest.json").read_text())
    source_manifest = json.loads((args.source_bundle / "manifest.json").read_text())
    package = args.source_bundle / source_manifest["decoder_partitions"][0]
    record = next(
        row
        for row in manifest["compiled_from"]["models"]
        if row["compiled"] == manifest["decoder_partitions"][0]
    )
    actual = {
        str(path.relative_to(package)): digest(path)
        for path in package.rglob("*")
        if path.is_file()
    }
    if actual != record["source_sha256"]:
        raise ValueError("Capture source differs from target partition provenance")
    capture_package = args.output / "capture.mlpackage"
    taps, projections = add_capture_outputs(package, capture_package, args.layers)
    compiled = args.output / "capture.mlmodelc"
    ct.models.utils.compile_model(str(capture_package), destination_path=str(compiled))
    report = {
        "complete": False,
        "layers": args.layers,
        "target_manifest_sha256": digest(args.target / "manifest.json"),
        "calibration_manifest_sha256": digest(args.manifest),
        "capture_spec_sha256": digest(
            capture_package / "Data/com.apple.CoreML/model.mlmodel"
        ),
        "projections": projections,
        "ranges": {},
        "utterances": [],
        "method": "existing stateful target with projection-input outputs; global extrema over all valid rows",
        "compute_units": "cpu_and_ne",
        "actual_ane_placement_verified": False,
    }
    runtime = CoreMLRuntime(args.target)
    capture = PersistentInputModel(
        ct.models.CompiledMLModel(
            str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
    )
    runtime.decoders[0].close()
    runtime.decoders[0] = capture
    original = capture.predict
    original_head = runtime.lm_head.predict
    generation_started = False
    current_id = None

    def mark_generation(data):
        nonlocal generation_started
        generation_started = True
        return original_head(data)

    runtime.lm_head.predict = mark_generation

    def observe(data, *, state):
        if generation_started and "example" not in report:
            arrays = {name: np.array(value, copy=True) for name, value in data.items()}
            for layer in range(args.layers):
                for kind in ("key", "value"):
                    name = f"{kind}_{layer}"
                    arrays[f"state__{name}"] = np.array(
                        state.read_state(name), copy=True
                    )
            example = args.output / "generation-example.npz"
            np.savez_compressed(example, **arrays)
            report["example"] = {"id": current_id, "archive_sha256": digest(example)}
        output = original(data, state=state)
        valid = np.asarray(data["update_mask"])[0, 0].sum(axis=-1) != 0
        for name in taps:
            values = output[name][..., valid].astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite activation in {name}")
            minimum, maximum = float(values.min()), float(values.max())
            previous = report["ranges"].setdefault(
                name, {"minimum": minimum, "maximum": maximum, "rows": 0}
            )
            previous["minimum"] = min(previous["minimum"], minimum)
            previous["maximum"] = max(previous["maximum"], maximum)
            previous["rows"] += int(valid.sum())
            previous["channels"] = values.shape[1]
        return output

    capture.predict = observe
    try:
        for item in manifest_rows(args.manifest):
            generation_started = False
            current_id = item["id"]
            audio, audio_hash = audio_samples(Path(item["audio_path"]))
            if audio_hash != item["audio_sha256"]:
                raise ValueError("Calibration audio changed")
            runtime.transcribe(audio, language=None, max_new_tokens=256)
            report["utterances"].append(
                {
                    "id": item["id"],
                    "audio_sha256": audio_hash,
                    "language": item["language"],
                }
            )
            (args.output / "ranges.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps({"id": item["id"], "captured": len(report["ranges"])}),
                flush=True,
            )
        report["complete"] = True
    finally:
        capture.predict = original
        runtime.lm_head.predict = original_head
        original = None
        original_head = None
        try:
            runtime.close()
            report["close_succeeded"] = True
        finally:
            (args.output / "ranges.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
