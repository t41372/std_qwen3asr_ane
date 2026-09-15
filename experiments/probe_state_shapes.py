"""Isolate enumerated/stateful load failures from tracing and reshape hints.

Run each case in a separate process. This tiny accumulator proves state
continuity only; CPU_AND_NE is a device policy, not placement evidence.
"""

import argparse
import json
import warnings
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from std_qwen3asr_ane.runtime import PersistentInputModel


def build_mil(path, *, fixed=False):
    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, 128, 1, 16 if fixed else get_new_symbol()), dtype=types.fp16
            ),
            mb.StateTensorSpec(shape=(1, 128, 1, 64), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def program(x, cache):
        projected = mb.conv(
            x=x, weight=np.eye(128, dtype=np.float16).reshape(128, 128, 1, 1)
        )
        addition = mb.reduce_sum(x=projected, axes=[3], keep_dims=True)
        updated = mb.add(x=mb.read_state(input=cache), y=addition)
        result = mb.mul(x=updated, y=np.float16(2), name="y")
        mb.coreml_update_state(state=cache, value=updated)
        return result

    model = ct.convert(
        program,
        inputs=[
            ct.TensorType(
                name="x",
                dtype=np.float16,
                shape=(1, 128, 1, 16)
                if fixed
                else ct.EnumeratedShapes(
                    shapes=[(1, 128, 1, 1), (1, 128, 1, 16)], default=(1, 128, 1, 16)
                ),
            )
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    model.save(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=None)
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Compile to a stable path before loading",
    )
    parser.add_argument(
        "--fixed", action="store_true", help="Fixed T16 stateful control"
    )
    parser.add_argument(
        "--compute-units", choices=("cpu_only", "cpu_and_ne"), required=True
    )
    parser.add_argument(
        "--reshape-hint", choices=("default", "infrequent"), default="default"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "existing_torch_trace" if args.source_model else "direct_mil",
        "fixed_width_control": args.fixed,
        "explicit_compiled_load": args.compiled,
        "compute_units": args.compute_units,
        "reshape_hint": args.reshape_hint,
        "load_succeeded": False,
        "state_parity": False,
        "actual_ane_placement_verified": False,
        "predictions": [],
    }
    model, state, wrappers = None, None, {}
    try:
        path = args.source_model or args.output.with_suffix(".mlpackage")
        if args.source_model is None:
            build_mil(path, fixed=args.fixed)
        report["model_path"] = str(path.resolve())
        channels = (
            ct.models.utils.load_spec(str(path))
            .description.input[0]
            .type.multiArrayType.shape[1]
        )
        model_type = ct.models.MLModel
        if args.compiled:
            compiled_path = args.output.with_suffix(".mlmodelc")
            ct.models.utils.compile_model(
                str(path), destination_path=str(compiled_path)
            )
            path = compiled_path
            model_type = ct.models.CompiledMLModel
        hints = (
            {"reshapeFrequency": ct.ReshapeFrequency.Infrequent}
            if args.reshape_hint == "infrequent"
            else None
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = model_type(
                str(path),
                compute_units=ct.ComputeUnit.CPU_ONLY
                if args.compute_units == "cpu_only"
                else ct.ComputeUnit.CPU_AND_NE,
                optimization_hints=hints,
            )
        report["load_warnings"] = [str(warning.message) for warning in captured]
        wrappers = {width: PersistentInputModel(model) for width in (1, 16)}
        model = None
        state = wrappers[16].make_state()
        report["load_succeeded"] = True
        expected = 0
        for width, value in ((16, 1), (1, 2), (16, 3), (1, 4)):
            if args.fixed:
                width = 16
            expected += width * value
            actual = wrappers[width].predict(
                {"x": np.full((1, channels, 1, width), value, np.float32)}, state=state
            )["y"]
            passed = bool(np.all(actual == expected * 2))
            report["predictions"].append(
                {"width": width, "expected": expected * 2, "passed": passed}
            )
            if not passed:
                raise AssertionError(
                    "Shape switch did not preserve the same model's state"
                )
        report["state_parity"] = True
    except Exception as error:  # noqa: BLE001 — Core ML raises untyped Exception on load failure.
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        state, model = None, None
        try:
            PersistentInputModel.close_many(list(wrappers.values()))
            report["close_succeeded"] = True
        except RuntimeError as error:
            report["close_succeeded"] = False
            report["close_error"] = str(error)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["state_parity"] and report["close_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
