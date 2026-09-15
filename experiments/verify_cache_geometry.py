"""Compare finite, consumed KV state across two cache geometries on real audio.

This runs the ordinary transcribe method, retaining its prepared MLState handles
only long enough to inspect the final state through Core ML's public API. No
state inspection is included in latency measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def capture(runtime, audio, budget):
    original = runtime.prepare_prompt
    prepared = None

    def prepare(*args, **kwargs):
        nonlocal prepared
        prepared = original(*args, **kwargs)
        return prepared

    runtime.prepare_prompt = prepare
    try:
        result = runtime.transcribe(audio, language=None, max_new_tokens=budget)
    finally:
        runtime.prepare_prompt = original
    consumed = len(prepared.token_ids) + len(result.token_ids)
    layers = 28 // len(prepared.states)
    if layers * len(prepared.states) != 28:
        raise ValueError("Expected uniform target partitions")
    arrays, descriptions = {}, {}
    for index, state in enumerate(prepared.states):
        for layer in range(layers):
            for kind in ("key", "value"):
                name = f"{kind}_{layer}"
                value = np.asarray(state.read_state(name))
                if not np.isfinite(value).all() or consumed > value.shape[-1]:
                    raise ValueError("Invalid or nonfinite KV state")
                prefix = np.array(value[..., :consumed], copy=True)
                key = f"partition_{index}/{name}"
                arrays[key] = prefix
                descriptions[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "consumed_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
                    "unused_nonzero_elements": int(
                        np.count_nonzero(value[..., consumed:])
                    ),
                }
    return arrays, {
        "consumed_positions": consumed,
        "prompt_token_ids": list(prepared.token_ids),
        "output_token_ids": list(result.token_ids),
        "eos_token_id": result.timings["eos_token_id"],
        "finite_states": True,
        "states": descriptions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.max_new_tokens < 1:
        parser.error("Use a fresh output and a positive token budget")
    report = {
        "complete": False,
        "baseline_manifest_sha256": digest(args.baseline / "manifest.json"),
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "corpus_sha256": digest(args.manifest),
        "script_sha256": digest(Path(__file__)),
        "cases": [],
    }
    runtimes = []
    try:
        for path in (args.baseline, args.candidate):
            runtimes.append(CoreMLRuntime(path))
        for item in manifest_rows(args.manifest):
            audio, audio_hash = audio_samples(Path(item["audio_path"]))
            baseline, left = capture(runtimes[0], audio, args.max_new_tokens)
            candidate, right = capture(runtimes[1], audio, args.max_new_tokens)
            if baseline.keys() != candidate.keys():
                raise ValueError("Different KV layouts")
            mismatches = {}
            for key, first in baseline.items():
                second = candidate[key]
                if first.shape != second.shape or first.dtype != second.dtype:
                    mismatches[key] = {
                        "baseline_shape": list(first.shape),
                        "candidate_shape": list(second.shape),
                        "baseline_dtype": str(first.dtype),
                        "candidate_dtype": str(second.dtype),
                    }
                elif not np.array_equal(first, second):
                    mismatches[key] = {
                        "elements": int(np.count_nonzero(first != second)),
                        "max_abs_difference": float(
                            np.max(np.abs(first.astype(np.float32) - second))
                        ),
                    }
            exact_decisions = all(
                left[key] == right[key]
                for key in (
                    "consumed_positions",
                    "prompt_token_ids",
                    "output_token_ids",
                    "eos_token_id",
                )
            )
            row = {
                "id": item["id"],
                "audio_sha256": audio_hash,
                "exact_decisions": exact_decisions,
                "exact_consumed_kv": not mismatches,
                "differences": mismatches,
                "baseline": left,
                "candidate": right,
            }
            report["cases"].append(row)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in ("id", "exact_decisions", "exact_consumed_kv")
                    }
                ),
                flush=True,
            )
        report["complete"] = True
        report["passed"] = all(
            row["exact_decisions"] and row["exact_consumed_kv"]
            for row in report["cases"]
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            PersistentInputModel.close_many(
                [
                    model
                    for runtime in runtimes
                    for model in (
                        runtime.frontend,
                        runtime.encoder,
                        *runtime.decoders,
                        runtime.lm_head,
                    )
                ]
            )
            report["close_succeeded"] = True
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
