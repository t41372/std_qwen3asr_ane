"""Replay real audio graph inputs with equal useful work and explicit batch tails."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
from evaluate import audio_samples

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.diagnostics import inspect_compute_plan
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


class Recorder:
    def __init__(self, model):
        self.model, self.inputs = model, []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, data, **kwargs):
        self.inputs.append({key: np.array(value, copy=True) for key, value in data.items()})
        return self.model.predict(data, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.candidate / "manifest.json").read_text())
    role, batch = manifest["audio_candidate"]["role"], manifest["audio_candidate"]["batch"]
    output_name = "chunk_embeddings" if role == "frontend" else "audio_embeddings"
    runtime = CoreMLRuntime(args.baseline)
    recorder = Recorder(getattr(runtime, role))
    setattr(runtime, role, recorder)
    try:
        for language in ("en", "zh"):
            samples, _ = audio_samples(
                Path(f"artifacts/evaluation/smoke/qwen_official_{language}.wav")
            )
            runtime.prepare_prompt(samples, language=None, max_new_tokens=256)
        inputs = recorder.inputs
    finally:
        runtime.close()
    baseline_manifest = json.loads((args.baseline / "manifest.json").read_text())
    candidate_role = f"{role}_batched" if batch > 1 else role
    paths = {
        "baseline": args.baseline / baseline_manifest["files"][role],
        "candidate": args.candidate / manifest["files"][candidate_role],
    }
    report = {
        "complete": False,
        "role": role,
        "batch": batch,
        "real_b1_inputs": len(inputs),
        "close_succeeded": False,
        "baseline_manifest_sha256": digest(args.baseline / "manifest.json"),
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
    }
    models, records = {}, []
    absolute_sum, elements = 0.0, 0
    try:
        for name, path in paths.items():
            models[name] = PersistentInputModel(
                ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            )
        groups = [
            inputs[offset : offset + batch] for offset in range(0, len(inputs) - batch + 1, batch)
        ]
        if not groups:
            raise ValueError("Insufficient real inputs for one full batch")
        # Only complete groups are timed; the remainder is reported, not measured.
        report["timed_inputs"] = len(groups) * batch
        with (args.output / "raw.jsonl").open("x") as stream:
            for repeat in range(-3, 30):
                for index, group in enumerate(groups):
                    packed = {
                        key: np.concatenate([row[key] for row in group], axis=0) for key in group[0]
                    }
                    results = {}
                    order = (
                        ("baseline", "candidate")
                        if (repeat + index) % 2 == 0
                        else ("candidate", "baseline")
                    )
                    for name in order:
                        started = perf_counter()
                        if name == "baseline":
                            outputs = [models[name].predict(row)[output_name] for row in group]
                        else:
                            outputs = [models[name].predict(packed)[output_name]]
                        elapsed = perf_counter() - started
                        results[name] = np.concatenate(outputs, axis=0)
                        record = {
                            "repeat": repeat,
                            "group": index,
                            "mode": name,
                            "seconds": elapsed,
                        }
                        records.append(record)
                        stream.write(json.dumps(record) + "\n")
                    for output in results.values():
                        if not np.isfinite(output).all():
                            raise ValueError("Non-finite audio embeddings")
                    if repeat == 0:
                        error = np.abs(results["baseline"] - results["candidate"])
                        report["max_abs_error"] = max(report["max_abs_error"], float(error.max()))
                        absolute_sum += float(error.sum(dtype=np.float64))
                        elements += error.size
                stream.flush()
        report["mean_abs_error"] = absolute_sum / elements
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
