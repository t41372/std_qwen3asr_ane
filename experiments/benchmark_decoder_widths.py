"""Probe a real partition's shape handling, state continuity and host latency.

The runtime's real masks and input ownership wrapper are used. This diagnostic
does not substitute for full-transcript parity or hardware placement tracing.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = ct.models.MLModel(str(args.model), skip_model_load=True).get_spec()
    cache_length = next(
        item.type.multiArrayType.shape[-1]
        for item in metadata.description.input
        if item.name == "attention_mask"
    )
    start = perf_counter()
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_seconds = perf_counter() - start
    wrappers = {width: PersistentInputModel(model) for width in args.widths}
    del model
    runtime = CoreMLRuntime.__new__(CoreMLRuntime)
    runtime.embeddings = np.load(
        "artifacts/qwen3-asr-1.7b-final/embedding.npy", mmap_mode="r"
    )
    runtime.cache_length = cache_length
    phases = (
        np.arange(cache_length, dtype=np.float32)[:, None]
        * (1e6 ** (-np.arange(64, dtype=np.float32) / 64))[None]
    )
    runtime.cosine, runtime.sine = np.cos(phases), np.sin(phases)
    reports = []
    try:
        for width, wrapper in wrappers.items():
            runtime.token_batch_size = width
            runtime.decoders = [wrapper]
            states = [wrapper.make_state()]
            hidden = np.repeat(
                np.asarray(runtime.embeddings[8948], np.float32)[None, :, None, None],
                width,
                axis=-1,
            )
            durations = []
            for repeat in range(40):
                start = perf_counter()
                output = runtime._decode_step(hidden, 0, states)
                elapsed = perf_counter() - start
                if repeat >= 10:
                    durations.append(elapsed)
            report = {
                "width": width,
                "load_seconds": load_seconds,
                "median_seconds": float(np.median(durations)),
                "p95_seconds": float(np.percentile(durations, 95)),
                "finite": bool(np.isfinite(output).all()),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
        if 1 in wrappers and max(wrappers) > 1:
            width = max(wrappers)
            runtime.token_batch_size, runtime.decoders = width, [wrappers[width]]
            states = [wrappers[width].make_state()]
            runtime._decode_step(hidden, 0, states)
            before = np.array(states[0].read_state("key_0"), copy=True)
            runtime.token_batch_size, runtime.decoders = 1, [wrappers[1]]
            runtime._decode_step(hidden[..., :1], width, states)
            after = np.array(states[0].read_state("key_0"), copy=True)
            preserved = bool(np.array_equal(before[..., :width], after[..., :width]))
            reports.append(
                {
                    "shape_switch_preserves_prefix_state": preserved,
                    "new_position_written": bool(np.any(after[..., width] != 0)),
                }
            )
            if not preserved:
                raise AssertionError("Shape switch changed prior KV positions")
    finally:
        states = None
        # These fixed-buffer wrappers share one underlying enumerated model.
        # Retire every handle before waiting for any native input borrowers.
        for wrapper in wrappers.values():
            wrapper._resources["model"] = None
        for wrapper in wrappers.values():
            wrapper.close()
        args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
