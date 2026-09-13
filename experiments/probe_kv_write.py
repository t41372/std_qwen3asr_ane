"""Find a KV-cache write formulation that becomes a partial state update on ANE.

The production graph writes the cache with full-cache mask multiplies, which
read and write the entire state twice per layer per step. This toy probe checks
which PyTorch formulations coremltools lowers to `slice_update` (or another
partial write) with a dynamic position, and whether that op is planned on ANE.
"""

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
    """cache[..., pos:pos+1] = k with a tensor position (jit.trace path)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, position):
        pos = position[0]
        self.cache[:, :, :, pos : pos + 1] = k
        return self.cache[:, :, :, 0:2].sum() + k.sum()


class IndexCopy(nn.Module):
    """cache.index_copy_(3, position, k)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, position):
        self.cache.index_copy_(3, position, k)
        return self.cache[:, :, :, 0:2].sum() + k.sum()


def op_types(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    spec = model.get_spec()
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                counts[operation.type] = counts.get(operation.type, 0) + 1
    return counts


def plan_devices(model) -> dict[str, int]:
    from std_qwen3asr_ane.diagnostics import inspect_compute_plan

    return inspect_compute_plan(model, "cpu_and_ne")["summary"]


def convert(module, name, output: Path, *, exported: bool):
    k = torch.ones(HEADS, DIM, 1, 1)
    position = torch.tensor([3], dtype=torch.int32)
    module.eval()
    if exported:
        program = torch.export.export(module, (k, position))
        source = program
    else:
        source = torch.jit.trace(module, (k, position), check_trace=False)
    model = ct.convert(
        source,
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
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    path = output / f"{name}.mlpackage"
    model.save(str(path))
    return path


def run(path: Path) -> dict:
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    k = np.full((HEADS, DIM, 1, 1), 2.0, dtype=np.float32)
    checks = {}
    for pos in (0, 5, 1023):
        model.predict({"k": k, "position": np.array([pos], dtype=np.int32)}, state=state)
        cache = np.array(state.read_state("cache"))
        written = np.flatnonzero(cache[0, 0, 0] != 0).tolist()
        checks[str(pos)] = written
    durations = []
    for i in range(40):
        start = perf_counter()
        model.predict({"k": k, "position": np.array([i % CACHE], dtype=np.int32)}, state=state)
        if i >= 10:
            durations.append(perf_counter() - start)
    return {"written_positions": checks, "median_ms": float(np.median(durations)) * 1e3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for name, module_type, exported in (
        ("slice_assign_trace", SliceAssign, False),
        ("slice_assign_export", SliceAssign, True),
        ("index_copy_trace", IndexCopy, False),
        ("index_copy_export", IndexCopy, True),
    ):
        entry: dict = {}
        try:
            path = convert(module_type(), name, args.output, exported=exported)
            model = ct.models.MLModel(str(path), skip_model_load=True)
            entry["ops"] = op_types(model)
            entry["plan"] = plan_devices(path)
            entry.update(run(path))
        except Exception as error:  # noqa: BLE001 - diagnostic probe
            entry["error"] = f"{type(error).__name__}: {str(error)[:500]}"
        report[name] = entry
        print(json.dumps({name: entry}, default=str), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
