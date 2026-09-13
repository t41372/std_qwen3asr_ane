"""Third KV-write probe: scatter-based partial writes and their ANE placement."""

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


class ScatterWrite(nn.Module):
    """cache.scatter_(3, index, k) with a broadcast integer index input."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, index):
        self.cache.scatter_(3, index, k)
        return self.cache[:, :, :, 0:2].sum() + k.sum()


class SelectWrite(nn.Module):
    """where(mask, k, cache): one read and one write of the whole cache."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, mask):
        self.cache.copy_(torch.where(mask > 0.5, k, self.cache))
        return self.cache[:, :, :, 0:2].sum() + k.sum()


class MaskWrite(nn.Module):
    """The production formulation: mul by (1-mask) then add k*mask."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(HEADS, DIM, 1, CACHE))

    def forward(self, k, mask):
        self.cache.mul_(1 - mask)
        self.cache.add_(k * mask)
        return self.cache[:, :, :, 0:2].sum() + k.sum()


def convert(module, name, output: Path, second_input):
    k = torch.ones(HEADS, DIM, 1, 1)
    module.eval()
    traced = torch.jit.trace(module, (k, second_input), check_trace=False)
    if second_input.dtype == torch.int64:
        second = ct.TensorType(name="index", shape=tuple(second_input.shape), dtype=np.int32)
    else:
        second = ct.TensorType(name="mask", shape=tuple(second_input.shape), dtype=np.float16)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="k", shape=k.shape, dtype=np.float16), second],
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


def op_types(path):
    spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
    counts: dict[str, int] = {}
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                counts[operation.type] = counts.get(operation.type, 0) + 1
    return counts


def feed_for(name, pos):
    k = np.full((HEADS, DIM, 1, 1), 2.0, dtype=np.float32)
    if name == "scatter":
        return {"k": k, "index": np.full((HEADS, DIM, 1, 1), pos, dtype=np.int32)}
    mask = np.zeros((1, 1, 1, CACHE), dtype=np.float32)
    mask[..., pos] = 1
    return {"k": k, "mask": mask}


def run(name, path: Path) -> dict:
    from std_qwen3asr_ane.diagnostics import inspect_compute_plan

    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    checks = {}
    for pos in (0, 5, 1023):
        model.predict(feed_for(name, pos), state=state)
        cache = np.array(state.read_state("cache"))
        checks[str(pos)] = np.flatnonzero(cache[0, 0, 0] != 0).tolist()
    durations = []
    for i in range(50):
        feed = feed_for(name, i % CACHE)
        start = perf_counter()
        model.predict(feed, state=state)
        if i >= 10:
            durations.append(perf_counter() - start)
    summary = inspect_compute_plan(path, "cpu_and_ne")["summary"]
    return {
        "written_positions": checks,
        "median_ms": float(np.median(durations)) * 1e3,
        "plan": {d: c["operation_count"] for d, c in summary["by_preferred_device"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    cases = (
        ("scatter", ScatterWrite, torch.full((HEADS, DIM, 1, 1), 3, dtype=torch.int64)),
        ("select", SelectWrite, torch.zeros(1, 1, 1, CACHE)),
        ("mask", MaskWrite, torch.zeros(1, 1, 1, CACHE)),
    )
    for name, module_type, second in cases:
        entry: dict = {}
        try:
            path = convert(module_type(), name, args.output, second)
            entry["ops"] = op_types(path)
            entry.update(run(name, path))
        except Exception as error:  # noqa: BLE001
            entry["error"] = f"{type(error).__name__}: {str(error)[:400]}"
        report[name] = entry
        print(json.dumps({name: entry}, default=str), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
