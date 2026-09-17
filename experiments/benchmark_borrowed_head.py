"""Compare owned and synchronously consumed logits on identical real hidden rows.

Uses one compiled head and randomized ABBA blocks. Prediction includes the SDK
bridge; consumer/copy time is measured separately. Stress mode retains periodic
RSS and weak-reference observations, not one output per prediction.
"""

import argparse
import gc
import json
import resource
import weakref
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import PersistentInputModel, logits_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--fixtures", type=Path, default=Path("artifacts/evaluation/round2/head-calibration.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--stress-calls", type=int, default=0)
    args = parser.parse_args()
    if args.blocks < 1 or args.stress_calls < 0:
        parser.error("Positive blocks and nonnegative stress calls required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    path = args.bundle / manifest["files"]["lm_head"]
    embedding = np.load(args.bundle / manifest["files"]["embedding"], mmap_mode="r")
    vocabulary = embedding.shape[0]
    fixture = np.load(args.fixtures)["hidden_states"][:60]
    hidden = [np.ascontiguousarray(row[None, :, None, None], dtype=np.float16) for row in fixture]
    report = {
        "complete": False,
        "close_succeeded": False,
        "bundle_manifest_sha256": digest(args.bundle / "manifest.json"),
        "fixtures_sha256": digest(args.fixtures),
        "script_sha256": digest(Path(__file__)),
        "rows": len(hidden),
        "stress_calls": args.stress_calls,
        "prediction_scope": "host wall including persistent input preparation and SDK bridge",
    }
    model = None
    records = []
    try:
        model = PersistentInputModel(
            ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        )
        expected = [
            logits_token(model.predict({"hidden_states": row}), vocabulary_size=vocabulary)
            for row in hidden
        ]
        rng = np.random.default_rng(20260915)
        with (args.output / "raw.jsonl").open("x") as output:
            for block in range(args.blocks):
                order = ("owned", "borrowed", "borrowed", "owned")
                if rng.integers(2):
                    order = ("borrowed", "owned", "owned", "borrowed")
                for slot, mode in enumerate(order):
                    for index, row in enumerate(hidden):
                        data = {"hidden_states": row}
                        start = perf_counter()
                        # This diagnostic intentionally splits the wrapper's
                        # prediction and consumption, preserving local ownership.
                        outputs = model._predict_outputs(data)
                        predicted = perf_counter()
                        if mode == "owned":
                            outputs = {
                                name: np.array(value, copy=True, order="C")
                                for name, value in outputs.items()
                            }
                        copied = perf_counter()
                        token = logits_token(outputs, vocabulary_size=vocabulary)
                        consumed = perf_counter()
                        record = {
                            "block": block,
                            "slot": slot,
                            "mode": mode,
                            "row": index,
                            "prediction_seconds": predicted - start,
                            "copy_seconds": copied - predicted,
                            "argmax_seconds": consumed - copied,
                            "total_seconds": consumed - start,
                            "exact": token == expected[index],
                        }
                        records.append(record)
                        output.write(json.dumps(record) + "\n")
                        if not record["exact"]:
                            raise AssertionError("Output consumption changed greedy token")
                        del outputs
                    output.flush()
        refs = []
        with (args.output / "stress.jsonl").open("x") as output:
            for index in range(args.stress_calls):
                row = index % len(hidden)

                def consume(outputs, call_index=index):
                    if call_index % 1000 == 0:
                        refs.extend(weakref.ref(value) for value in outputs.values())
                    return logits_token(outputs, vocabulary_size=vocabulary)

                token = model.predict_consumed({"hidden_states": hidden[row]}, consume)
                if token != expected[row]:
                    raise AssertionError(f"Stress token changed at call {index}")
                if index % 1000 == 0 or index + 1 == args.stress_calls:
                    gc.collect()
                    record = {
                        "calls": index + 1,
                        "live_sampled_arrays": sum(r() is not None for r in refs),
                        "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    refs.clear()
        report["medians"] = {
            mode: {
                metric: float(np.median([row[metric] for row in records if row["mode"] == mode]))
                for metric in (
                    "prediction_seconds",
                    "copy_seconds",
                    "argmax_seconds",
                    "total_seconds",
                )
            }
            for mode in ("owned", "borrowed")
        }
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if model is not None:
                model.close()
            report["close_succeeded"] = True
        finally:
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
