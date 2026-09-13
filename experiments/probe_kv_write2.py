"""Second KV-write probe: is the dynamic slice_update model valid on CPU, and on ANE?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
import torch
from torch import nn

CACHE = 1024
HEADS, DIM = 8, 128


class SliceAssign(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, position):
        pos = position[0]
        self.cache[:, :, :, pos : pos + 1] = k
        return self.cache[:, :, :, 0:2].sum() + k.sum()


class IndexCopy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, position):
        self.cache.index_copy_(3, position.long(), k)
        return self.cache[:, :, :, 0:2].sum() + k.sum()


class SliceAssignAttend(nn.Module):
    """Write then read the whole cache through a matmul, like attention."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, position):
        pos = position[0]
        self.cache[:, :, :, pos : pos + 1] = k
        q = k.permute(0, 2, 3, 1)  # [8,1,1,128]
        scores = torch.matmul(q, self.cache.permute(0, 2, 1, 3))  # [8,1,1,1024]
        return scores.sum()


def convert(module, name, output: Path, target):
    k = torch.ones(HEADS, DIM, 1, 1)
    position = torch.tensor([3], dtype=torch.int32)
    module.eval()
    traced = torch.jit.trace(module, (k, position), check_trace=False)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="k", shape=k.shape, dtype=np.float16),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(HEADS, DIM, 1, CACHE), dtype=np.float16),
                name="cache",
            )
        ],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    path = output / f"{name}.mlpackage"
    model.save(str(path))
    return path


def op_types(path):
    spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
    counts: dict[str, int] = {}
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                counts[operation.type] = counts.get(operation.type, 0) + 1
    return counts


def run(path: Path, units) -> dict:
    model = ct.models.MLModel(str(path), compute_units=units)
    state = model.make_state()
    k = np.full((HEADS, DIM, 1, 1), 2.0, dtype=np.float32)
    checks = {}
    for pos in (0, 5, 1023):
        model.predict({"k": k, "position": np.array([pos], dtype=np.int32)}, state=state)
        cache = np.array(state.read_state("cache"))
        checks[str(pos)] = np.flatnonzero(cache[0, 0, 0] != 0).tolist()
    durations = []
    for i in range(40):
        start = perf_counter()
        model.predict({"k": k, "position": np.array([i % CACHE], dtype=np.int32)}, state=state)
        if i >= 10:
            durations.append(perf_counter() - start)
    result = {"written_positions": checks, "median_ms": float(np.median(durations)) * 1e3}
    if units == ct.ComputeUnit.CPU_AND_NE:
        from std_qwen3asr_ane.diagnostics import inspect_compute_plan

        summary = inspect_compute_plan(path, "cpu_and_ne")["summary"]
        result["plan"] = {d: c["operation_count"] for d, c in summary["by_preferred_device"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    targets = {"macOS15": ct.target.macOS15}
    if hasattr(ct.target, "macOS26"):
        targets["macOS26"] = ct.target.macOS26
    for name, module_type in (
        ("slice_assign", SliceAssign),
        ("slice_assign_attend", SliceAssignAttend),
    ):
        for target_name, target in targets.items():
            key = f"{name}-{target_name}"
            entry: dict = {}
            try:
                path = convert(module_type(), key, args.output, target)
                entry["ops"] = op_types(path)
                for units_name, units in (("cpu_only", ct.ComputeUnit.CPU_ONLY), ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE)):
                    try:
                        entry[units_name] = run(path, units)
                    except Exception as error:  # noqa: BLE001
                        entry[units_name] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            except Exception as error:  # noqa: BLE001
                entry["error"] = f"{type(error).__name__}: {str(error)[:400]}"
            report[key] = entry
            print(json.dumps({key: entry}, default=str), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
