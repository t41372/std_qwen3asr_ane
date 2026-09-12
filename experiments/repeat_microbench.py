"""Bounded, stateful decoder-layer workload for hardware tracing.

This is an execution/placement probe, not an ASR quality or latency benchmark.
"""

import argparse
import json
import os
import time
from pathlib import Path

import coremltools as ct
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "artifacts/probes/decoder-layer-0-scale1.mlpackage",
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=root / "artifacts/qwen3-asr-1.7b/embedding.npy",
    )
    parser.add_argument(
        "--compute-units", choices=("cpu_and_ne", "cpu_only"), default="cpu_and_ne"
    )
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 20:
        parser.error("--seconds must be between 0 and 20")

    load_start = time.monotonic_ns()
    model = ct.models.MLModel(
        str(args.model),
        compute_units=getattr(ct.ComputeUnit, args.compute_units.upper()),
    )
    load_end = time.monotonic_ns()
    shapes = {
        item.name: tuple(item.type.multiArrayType.shape)
        for item in model.get_spec().description.input
    }
    context = shapes["attention_mask"][-1]
    half_dimension = shapes["cosine"][1]
    embeddings = np.load(args.embedding, mmap_mode="r")
    hidden = np.asarray(embeddings[8948], dtype=np.float16).reshape(
        shapes["hidden_states"]
    )
    frequency = 1_000_000.0 ** (np.arange(half_dimension) / half_dimension)
    inputs = []
    for position in range(context):
        mask = np.full(shapes["attention_mask"], -10000, dtype=np.float16)
        mask[..., : position + 1] = 0
        update = np.zeros(shapes["update_mask"], dtype=np.float16)
        update[..., position] = 1
        angle = position / frequency
        inputs.append(
            {
                "hidden_states": hidden,
                "cosine": np.cos(angle).astype(np.float16).reshape(shapes["cosine"]),
                "sine": np.sin(angle).astype(np.float16).reshape(shapes["sine"]),
                "attention_mask": mask,
                "update_mask": update,
            }
        )

    state = model.make_state()
    rows = []
    started = time.monotonic_ns()
    deadline = started + int(args.seconds * 1e9)
    while time.monotonic_ns() < deadline:
        position = len(rows) % context
        if position == 0 and rows:
            state = model.make_state()
        begin = time.monotonic_ns()
        result = model.predict(inputs[position], state=state)
        end = time.monotonic_ns()
        if not np.isfinite(result["output_hidden_states"]).all():
            raise RuntimeError("Non-finite decoder output during placement probe")
        rows.append({"start_ns": begin, "end_ns": end, "position": position})
    finished = time.monotonic_ns()
    report = {
        "purpose": "bounded hardware placement workload; not quality or performance certification",
        "pid": os.getpid(),
        "model": args.model.name,
        "compute_units": args.compute_units,
        "load_start_monotonic_ns": load_start,
        "load_end_monotonic_ns": load_end,
        "predict_start_monotonic_ns": started,
        "predict_end_monotonic_ns": finished,
        "prediction_count": len(rows),
        "all_outputs_finite": True,
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "predictions"}
        )
    )


if __name__ == "__main__":
    main()
