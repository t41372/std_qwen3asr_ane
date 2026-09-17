"""Compare a changed head graph with baseline using real rows and borrowed scans."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.diagnostics import inspect_compute_plan
from std_qwen3asr_ane.runtime import PersistentInputModel, logits_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = Path("artifacts/evaluation/round2/head-calibration.npz")
    hidden = np.load(fixtures)["hidden_states"][:60]
    paths, models, manifests = {}, {}, {}
    report = {
        "complete": False,
        "close_succeeded": False,
        "fixtures_sha256": digest(fixtures),
        "models": {},
    }
    records = []
    try:
        for name, bundle in (("baseline", args.baseline), ("candidate", args.candidate)):
            manifest = manifests[name] = json.loads((bundle / "manifest.json").read_text())
            paths[name] = bundle / manifest["files"]["lm_head"]
            models[name] = PersistentInputModel(
                ct.models.CompiledMLModel(str(paths[name]), compute_units=ct.ComputeUnit.CPU_AND_NE)
            )
            report["models"][name] = {
                "path": str(paths[name]),
                "manifest_sha256": digest(bundle / "manifest.json"),
            }
        # Schema-3 bundles store INT8 rows under another name; the manifest knows it.
        embedding = args.baseline / manifests["baseline"]["files"]["embedding"]
        vocabulary = np.load(embedding, mmap_mode="r", allow_pickle=False).shape[0]

        def consume(outputs):
            return logits_token(outputs, vocabulary_size=vocabulary)

        with (args.output / "raw.jsonl").open("x") as output:
            for repeat in range(-2, 10):
                for index, row in enumerate(hidden):
                    data = {
                        "hidden_states": np.ascontiguousarray(
                            row[None, :, None, None], dtype=np.float16
                        )
                    }
                    order = (
                        ("baseline", "candidate")
                        if (repeat + index) % 2 == 0
                        else ("candidate", "baseline")
                    )
                    tokens = {}
                    for name in order:
                        started = perf_counter()
                        token = models[name].predict_consumed(data, consume)
                        elapsed = perf_counter() - started
                        record = {
                            "repeat": repeat,
                            "row": index,
                            "mode": name,
                            "seconds": elapsed,
                            "token": token,
                        }
                        records.append(record)
                        output.write(json.dumps(record) + "\n")
                        tokens[name] = token
                    if tokens["baseline"] != tokens["candidate"]:
                        report.setdefault("token_mismatches", []).append({"row": index, **tokens})
                output.flush()
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
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            PersistentInputModel.close_many(list(models.values()))
            report["close_succeeded"] = True
        finally:
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plan = inspect_compute_plan(paths["candidate"])
    (args.output / "placement.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
