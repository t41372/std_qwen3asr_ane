"""Export 100 real generation steps and initial KV state for native bridge replay.

Replay is teacher-forced and decoder-only. Prefill, audio processing and output
selection happen during export, outside the later Python/Swift timing scopes.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 1000:
        parser.error("Use a fresh output and 1..1000 steps")
    args.output.mkdir(parents=True)
    fixture = {
        "schema_version": 1,
        "complete": False,
        "step_count": 0,
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "segments": [],
    }
    runtime = CoreMLRuntime(args.model_dir)
    layers = 28 // len(runtime.decoders)
    if layers * len(runtime.decoders) != 28:
        raise ValueError("Fixture export requires uniform decoder partitions")
    originals = [model.predict for model in runtime.decoders]
    original_step, original_head = runtime._decode_step, runtime._next_token
    generating, capture_step = False, False
    segment, step = None, None

    def save(relative, array):
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.ascontiguousarray(array, dtype="<f2")
        if not np.isfinite(array).all():
            raise ValueError("Fixture contains non-finite data")
        path.write_bytes(array.tobytes())
        return {
            "file": relative,
            "shape": list(array.shape),
            "dtype": "float16-le",
            "sha256": digest(path),
        }

    def head(hidden):
        nonlocal generating
        generating = True
        return original_head(hidden)

    def decode(hidden, position, states, **kwargs):
        nonlocal capture_step, step
        capture_step = generating and fixture["step_count"] < args.steps
        if capture_step:
            prefix = f"segment-{len(fixture['segments']) - 1}"
            if not segment["states"]:
                for partition, state in enumerate(states):
                    tensors = {}
                    for layer in range(layers):
                        for kind in ("key", "value"):
                            name = f"{kind}_{layer}"
                            tensors[name] = save(
                                f"{prefix}/state-{partition}-{name}.bin",
                                state.read_state(name),
                            )
                    segment["states"].append(tensors)
            step = {"inputs": {}, "position": position}
        result = original_step(hidden, position, states, **kwargs)
        if capture_step:
            segment["steps"].append(step)
            fixture["step_count"] += 1
        capture_step = False
        return result

    def instrument(index, original):
        def predict(data, *, state=None):
            prefix = (
                f"segment-{len(fixture['segments']) - 1}/step-{fixture['step_count']}"
            )
            if capture_step and index == 0:
                step["inputs"] = {
                    name: save(f"{prefix}-{name}.bin", value)
                    for name, value in data.items()
                }
            output = original(data, state=state)
            if capture_step and index == len(runtime.decoders) - 1:
                step["expected"] = save(
                    f"{prefix}-expected.bin", output["output_hidden_states"]
                )
            return output

        return predict

    runtime._decode_step, runtime._next_token = decode, head
    for index, model in enumerate(runtime.decoders):
        model.predict = instrument(index, originals[index])
    try:
        inputs = [
            (row, *audio_samples(Path(row["audio_path"])))
            for row in manifest_rows(args.manifest)
        ]
        for ordinal, (row, audio, audio_hash) in enumerate(itertools.cycle(inputs)):
            if ordinal > args.steps:
                raise RuntimeError(
                    "Not enough generation steps in the source utterances"
                )
            generating = False
            segment = {
                "id": row["id"],
                "audio_sha256": audio_hash,
                "states": [],
                "steps": [],
            }
            fixture["segments"].append(segment)
            runtime.transcribe(audio, language=None, max_new_tokens=256)
            if not segment["steps"]:
                fixture["segments"].pop()
            if fixture["step_count"] >= args.steps:
                break
        fixture["complete"] = True
    finally:
        runtime._decode_step, runtime._next_token = original_step, original_head
        for model, original in zip(runtime.decoders, originals, strict=True):
            model.predict = original
        originals.clear()
        original_step, original_head, original = None, None, None
        try:
            runtime.close()
            fixture["close_succeeded"] = True
        finally:
            fixture["complete"] = fixture["complete"] and fixture.get(
                "close_succeeded", False
            )
            (args.output / "fixture.json").write_text(
                json.dumps(fixture, indent=2) + "\n"
            )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "steps": fixture["step_count"],
                "segments": len(fixture["segments"]),
            }
        )
    )


if __name__ == "__main__":
    main()
