"""One four-layer T16 fusion hypothesis with real initial KV and hidden fixtures."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
import torch

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.compress import compress_model
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    convert_partition,
    load_partition,
)
from std_qwen3asr_ane.diagnostics import inspect_compute_plan
from std_qwen3asr_ane.runtime import PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("qkv", "mlp", "both", "grouped", "sdpa", "all"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path("artifacts/source/Qwen3-ASR-1.7B")
    control = Path("artifacts/evaluation/round2/w8a8-decoder-probe/lut8.mlmodelc")
    fixture = Path("artifacts/evaluation/round2/decoder-calibration/generation-example.npz")
    calibration = json.loads(fixture.with_name("ranges.json").read_text())
    if digest(fixture) != calibration["example"]["archive_sha256"]:
        raise ValueError("Real-state fixture changed")
    config = json.loads((source / "config.json").read_text())["thinker_config"]["text_config"]
    report = {
        "complete": False,
        "mode": args.mode,
        "fixture_sha256": digest(fixture),
        "source_revision": json.loads((source / "source.json").read_text())["revision"],
        "baseline_files": {
            str(p.relative_to(control)): digest(p)
            for p in sorted(control.rglob("*"))
            if p.is_file()
        },
    }
    models, states = {}, {}
    torch.set_num_threads(4)
    try:
        module = DecoderPartition(config, 4, 1024, token_batch_size=16).eval()
        load_partition(module, SourceWeights(source), 0)
        for layer in module.layers:
            if args.mode in ("grouped", "all"):
                layer.enable_grouped_attention()
            if args.mode in ("sdpa", "all"):
                layer.fused_attention = True
            layer.fuse_projections(
                attention=args.mode in ("qkv", "both", "all"),
                mlp=args.mode in ("mlp", "both", "all"),
            )
        dense, compressed = args.output / "dense.mlpackage", args.output / "lut8.mlpackage"
        convert_partition(module, config, dense, 1024)
        report["compression"] = compress_model(dense, compressed, "palette", 8, 32)
        compiled = args.output / "lut8.mlmodelc"
        ct.models.utils.compile_model(str(compressed), destination_path=str(compiled))
        plan = inspect_compute_plan(compiled)
        (args.output / "placement.json").write_text(json.dumps(plan, indent=2) + "\n")
        non_ane = [
            row for row in plan["operations"] if row["preferred_device"] not in (None, "ane")
        ]
        report["non_ane_operations"] = len(non_ane)
        with np.load(fixture) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        inputs = {key: value for key, value in arrays.items() if not key.startswith("state__")}
        initial = {
            key.removeprefix("state__"): value
            for key, value in arrays.items()
            if key.startswith("state__")
        }
        for name, path in (("baseline", control), ("candidate", compiled)):
            models[name] = PersistentInputModel(
                ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            )
            states[name] = models[name].make_state()
        last, records = {}, []
        with (args.output / "raw.jsonl").open("x") as stream:
            for repeat in range(-10, 50):
                order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                for name in order:
                    for key, value in initial.items():
                        states[name].write_state(key, value)
                    started = perf_counter()
                    last[name] = models[name].predict(inputs, state=states[name])[
                        "output_hidden_states"
                    ]
                    elapsed = perf_counter() - started
                    if not np.isfinite(last[name]).all():
                        raise ValueError("Non-finite decoder hidden state")
                    record = {"repeat": repeat, "mode": name, "seconds": elapsed}
                    stream.write(json.dumps(record) + "\n")
                    records.append(record)
                stream.flush()
        report["timings"] = {
            name: {
                f"p{p}": float(
                    np.percentile(
                        [
                            row["seconds"]
                            for row in records
                            if row["mode"] == name and row["repeat"] >= 0
                        ],
                        p,
                    )
                )
                for p in (50, 95)
            }
            for name in models
        }
        report["hidden_max_abs_error"] = float(
            np.max(
                np.abs(last["baseline"].astype(np.float32) - last["candidate"].astype(np.float32))
            )
        )
        used = int(np.flatnonzero((inputs["attention_mask"] == 0).any(axis=(0, 1, 2)))[-1]) + 1
        report["consumed_positions"] = used
        report["kv"] = {}
        for key in initial:
            a = np.array(states["baseline"].read_state(key), copy=True)[..., :used]
            b = np.array(states["candidate"].read_state(key), copy=True)[..., :used]
            report["kv"][key] = {
                "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
                "equal": bool(np.array_equal(a, b)),
                "max_abs_error": float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))),
            }
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        states.clear()
        try:
            PersistentInputModel.close_many(list(models.values()))
            report["close_succeeded"] = True
        finally:
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
