"""Compare finite prefill hidden values and every consumed KV slot on real audio."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from baseline_runtime import load_baseline_runtime, prediction_models
from evaluate import audio_samples, manifest_rows

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    factories = {
        "baseline": load_baseline_runtime(args.baseline_source)
        if args.baseline_source
        else CoreMLRuntime,
        "candidate": CoreMLRuntime,
    }
    runtimes, prepared = {}, {}
    report = {
        "complete": False,
        "cases": [],
        "manifest_sha256": digest(args.manifest),
        "bundles": {
            name: digest(path / "manifest.json")
            for name, path in (("baseline", args.baseline), ("candidate", args.candidate))
        },
    }
    try:
        for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
            runtimes[name] = factories[name](path)
            if len(runtimes[name].decoders) != 2:
                raise ValueError("This diagnostic requires the two p14 decoder partitions")
        for row in manifest_rows(args.manifest):
            audio, audio_hash = audio_samples(Path(row["audio_path"]))
            for name, runtime in runtimes.items():
                prepared[name] = runtime.prepare_prompt(
                    audio, language=None, max_new_tokens=args.max_new_tokens
                )
            if prepared["baseline"].token_ids != prepared["candidate"].token_ids:
                raise ValueError("Different input prompt tokens")
            used = len(prepared["baseline"].token_ids)
            a = prepared["baseline"].hidden.astype(np.float32)
            b = prepared["candidate"].hidden.astype(np.float32)
            case = {
                "id": row["id"],
                "audio_sha256": audio_hash,
                "consumed_positions": used,
                "hidden_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
                "hidden_max_abs_error": float(np.max(np.abs(a - b))),
                "states": [],
            }
            for partition in range(2):
                for layer in range(14):
                    for kind in ("key", "value"):
                        key = f"{kind}_{layer}"
                        a = np.array(
                            prepared["baseline"].states[partition].read_state(key), copy=True
                        )[..., :used]
                        b = np.array(
                            prepared["candidate"].states[partition].read_state(key), copy=True
                        )[..., :used]
                        case["states"].append(
                            {
                                "partition": partition,
                                "name": key,
                                "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
                                "equal": bool(np.array_equal(a, b)),
                                "max_abs_error": float(
                                    np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))
                                ),
                                "baseline_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                                "candidate_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                            }
                        )
            case["all_used_kv_equal"] = all(state["equal"] for state in case["states"])
            if not case["hidden_finite"] or not all(state["finite"] for state in case["states"]):
                raise ValueError("Non-finite hidden/KV values")
            report["cases"].append(case)
            prepared.clear()
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        prepared.clear()
        try:
            PersistentInputModel.close_many(
                [model for runtime in runtimes.values() for model in prediction_models(runtime)]
            )
            report["close_succeeded"] = True
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
