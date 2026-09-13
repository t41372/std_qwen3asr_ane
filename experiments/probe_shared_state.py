"""Test whether two functions in one Core ML asset can reuse the same MLState.

This small exact-sum probe is a prerequisite for different prefill/decode widths.
It does not establish ANE placement or transformer numerical accuracy.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from std_qwen3asr_ane.runtime import PersistentInputModel
from torch import nn


class Accumulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 8, 1, 1))
        self.projection = nn.Conv2d(8, 8, 1, bias=False)
        self.projection.weight.data.copy_(torch.eye(8).reshape(8, 8, 1, 1))

    def forward(self, x):
        self.cache.add_(self.projection(x).sum(dim=-1, keepdim=True))
        return self.cache * 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enumerated", action="store_true")
    parser.add_argument("--separate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    descriptor = ct.utils.MultiFunctionDescriptor()
    for width in (16,) if args.enumerated else (1, 16):
        module = Accumulator().eval()
        example = torch.zeros(1, 8, 1, width)
        model = ct.convert(
            torch.jit.trace(module, example, check_trace=False),
            inputs=[
                ct.TensorType(
                    name="x",
                    shape=ct.EnumeratedShapes(
                        shapes=[(1, 8, 1, 1), (1, 8, 1, 16)], default=(1, 8, 1, 16)
                    )
                    if args.enumerated
                    else example.shape,
                    dtype=np.float16,
                )
            ],
            states=[
                ct.StateType(
                    name="cache",
                    wrapped_type=ct.TensorType(shape=(1, 8, 1, 1), dtype=np.float16),
                )
            ],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            skip_model_load=True,
        )
        path = args.output / f"t{width}.mlpackage"
        model.save(str(path))
        descriptor.add_function(
            str(path), src_function_name="main", target_function_name=f"t{width}"
        )
    if args.separate:
        models = {
            width: PersistentInputModel(
                ct.models.MLModel(
                    str(args.output / f"t{width}.mlpackage"),
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                )
            )
            for width in (1, 16)
        }
    elif args.enumerated:
        shared_model = ct.models.MLModel(
            str(path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            optimization_hints={"reshapeFrequency": ct.ReshapeFrequency.Infrequent},
        )
        models = {width: PersistentInputModel(shared_model) for width in (1, 16)}
        del shared_model
    else:
        descriptor.default_function_name = "t1"
        combined = args.output / "combined.mlpackage"
        ct.utils.save_multifunction(descriptor, str(combined))
        models = {
            width: PersistentInputModel(
                ct.models.MLModel(
                    str(combined),
                    function_name=f"t{width}",
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                )
            )
            for width in (1, 16)
        }
    rows = []
    try:
        for origin in (1, 16):
            state = models[origin].make_state()
            expected = 0
            for width, value in ((16, 1), (1, 2), (16, 3), (1, 4)):
                expected += width * value
                result = models[width].predict(
                    {"x": np.full((1, 8, 1, width), value, np.float32)}, state=state
                )["y"]
                passed = bool(np.all(result == 2 * expected))
                rows.append(
                    {
                        "state_origin": origin,
                        "function_width": width,
                        "expected": 2 * expected,
                        "actual": result.reshape(-1).tolist(),
                        "passed": passed,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
                if not passed:
                    raise AssertionError(
                        "Cross-function state did not preserve the accumulated value"
                    )
    finally:
        state = None
        for model in models.values():
            model.close()
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
