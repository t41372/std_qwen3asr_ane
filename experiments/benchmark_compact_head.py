"""Check compact-head greedy choices and latency on actual speech decoder states."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
from evaluate import audio_samples
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


class RecordingHead:
    def __init__(self, model):
        self.model = model
        self.hidden = []

    def predict(self, data):
        self.hidden.append(np.array(data["hidden_states"], copy=True))
        return self.model.predict(data)

    def __getattr__(self, name):
        return getattr(self.model, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = CoreMLRuntime(Path("artifacts/qwen3-asr-1.7b-compiled"))
    recorder = RecordingHead(runtime.lm_head)
    runtime.lm_head = recorder
    compact = None
    report = {}
    try:
        for language in ("en", "zh"):
            samples, _ = audio_samples(
                Path(f"artifacts/evaluation/smoke/qwen_official_{language}.wav")
            )
            runtime.transcribe(samples, language=None, max_new_tokens=256)
        compact = PersistentInputModel(
            ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
        )
        width = compact.get_spec().description.input[0].type.multiArrayType.shape[-1]
        elapsed, mismatches = {"original": [], "compact": []}, []
        expected_tokens = []
        for hidden in recorder.hidden:
            data = {"hidden_states": hidden}
            start = perf_counter()
            original = recorder.model.predict(data)
            elapsed["original"].append(perf_counter() - start)
            logits = np.concatenate(
                [original[f"logits_{i}"].reshape(-1) for i in range(len(original))]
            )
            expected_tokens.append(int(np.argmax(logits)))
        for offset in range(0, len(recorder.hidden), width):
            block = recorder.hidden[offset : offset + width]
            padded = np.zeros((1, runtime.embeddings.shape[1], 1, width), np.float32)
            padded[..., : len(block)] = np.concatenate(block, axis=-1)
            start = perf_counter()
            selected = compact.predict({"hidden_states": padded})
            elapsed["compact"].append(perf_counter() - start)
            values = selected["max_values"].reshape(1, -1, width)
            indices = selected["max_indices"].reshape(1, -1, width)
            for row in range(len(block)):
                chunk = int(np.argmax(values[0, :, row]))
                actual = chunk * 8192 + int(indices[0, chunk, row])
                expected = expected_tokens[offset + row]
                if actual != expected:
                    mismatches.append(
                        {"index": offset + row, "expected": expected, "actual": actual}
                    )
        report = {
            "states": len(recorder.hidden),
            "token_batch_size": width,
            "mismatches": mismatches,
            "median_seconds": {
                name: float(np.median(values[1:])) for name, values in elapsed.items()
            },
            "seconds_per_useful_token_after_first_call": {
                "original": sum(elapsed["original"][1:]) / (len(recorder.hidden) - 1),
                "compact": sum(elapsed["compact"][1:]) / (len(recorder.hidden) - width),
            },
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if compact is not None:
            compact.close()
        runtime.close()
    if report["mismatches"]:
        raise AssertionError("Compact head changed greedy token selection")


if __name__ == "__main__":
    main()
