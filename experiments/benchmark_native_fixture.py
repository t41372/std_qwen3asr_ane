"""Python baseline for exactly the decoder steps replayed by DecoderBench.swift."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Use a fresh output and positive repeats")
    fixture = json.loads((args.fixture / "fixture.json").read_text())
    if not fixture["complete"] or fixture["model_manifest_sha256"] != digest(
        args.model_dir / "manifest.json"
    ):
        raise ValueError("Incomplete fixture or different model bundle")

    def load(descriptor):
        path = (args.fixture / descriptor["file"]).resolve()
        if (
            not path.is_relative_to(args.fixture.resolve())
            or digest(path) != descriptor["sha256"]
        ):
            raise ValueError("Fixture tensor path or hash mismatch")
        if descriptor["dtype"] != "float16-le":
            raise ValueError("Unsupported fixture dtype")
        return np.fromfile(path, dtype="<f2").reshape(descriptor["shape"])

    segments = [
        {
            "states": [
                {name: load(value) for name, value in state.items()}
                for state in segment["states"]
            ],
            "steps": [
                {
                    "inputs": {
                        name: load(value) for name, value in step["inputs"].items()
                    },
                    "expected": load(step["expected"]),
                }
                for step in segment["steps"]
            ],
        }
        for segment in fixture["segments"]
    ]
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    models = []
    states, state = [], None
    report = {
        "complete": False,
        "steps": fixture["step_count"],
        "fixture_sha256": digest(args.fixture / "fixture.json"),
        "timing_scope": "decoder predictions and Python staging only; excludes fixture loading, KV restoration, output verification and LM head",
        "seconds": [],
    }
    try:
        models = [
            PersistentInputModel(
                ct.models.CompiledMLModel(
                    str(args.model_dir / path), compute_units=ct.ComputeUnit.CPU_AND_NE
                )
            )
            for path in manifest["decoder_partitions"]
        ]
        for repeat in range(-1, args.repeats):
            elapsed = 0.0
            for segment in segments:
                states = [model.make_state() for model in models]
                for state, values in zip(states, segment["states"], strict=True):
                    for name, value in values.items():
                        # The public Python state bridge accepts float32 input;
                        # every saved FP16 value converts exactly before storage.
                        state.write_state(name, value.astype(np.float32))
                results = []
                started = perf_counter()
                for step in segment["steps"]:
                    inputs = dict(step["inputs"])
                    for model, state in zip(models, states, strict=True):
                        hidden = model.predict(inputs, state=state)[
                            "output_hidden_states"
                        ]
                        inputs["hidden_states"] = hidden
                    results.append(hidden)
                elapsed += perf_counter() - started
                for result, step in zip(results, segment["steps"], strict=True):
                    if not np.array_equal(result, step["expected"]):
                        raise AssertionError(
                            "Replayed decoder hidden states differ from the export"
                        )
            if repeat >= 0:
                report["seconds"].append(elapsed)
            print(json.dumps({"repeat": repeat, "seconds": elapsed}), flush=True)
        report.update(
            complete=True,
            exact_hidden_parity=True,
            median_seconds=float(np.median(report["seconds"])),
        )
    finally:
        states.clear()
        state = None
        PersistentInputModel.close_many(models)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
