"""Measure load, stages and per-model synchronous prediction costs serially.

This instrumented diagnostic identifies bottlenecks; use evaluate.py for final
uninstrumented performance comparisons. Input audio is decoded before timing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.runtime import CoreMLRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Positive repeats and a fresh output path are required")
    inputs = [
        (row, audio_samples(Path(row["audio_path"]))[0])
        for row in manifest_rows(args.manifest)
    ]
    started = perf_counter()
    runtime = CoreMLRuntime(args.model_dir)
    loaded = perf_counter()
    model_calls = defaultdict(list)
    models = {
        "frontend": runtime.frontend,
        "encoder": runtime.encoder,
        **{f"decoder_{i}": model for i, model in enumerate(runtime.decoders)},
        "lm_head": runtime.lm_head,
    }
    originals = {name: model.predict for name, model in models.items()}

    def instrument(name, original):
        def predict(*positional, **keywords):
            begin = perf_counter()
            result = original(*positional, **keywords)
            model_calls[name].append(perf_counter() - begin)
            return result

        return predict

    for name, model in models.items():
        model.predict = instrument(name, originals[name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w") as stream:
            for row, samples in inputs:
                for repeat in range(-1, args.repeats):
                    model_calls.clear()
                    result = runtime.transcribe(
                        samples, language=None, max_new_tokens=256
                    )
                    record = {
                        "id": row["id"],
                        "repeat": repeat,
                        "load_seconds": loaded - started,
                        "text": result.text,
                        "tokens": len(result.token_ids),
                        "timings": result.timings,
                        "models": {
                            name: {"calls": len(calls), "seconds": sum(calls)}
                            for name, calls in model_calls.items()
                        },
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        for name, model in models.items():
            model.predict = originals[name]
        originals.clear()
        runtime.close()


if __name__ == "__main__":
    main()
