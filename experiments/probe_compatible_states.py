"""Test public MLState compatibility between real fixed-width decoder models.

Compare shared state against explicit read/write copies on this exact host.
This is an experiment, not a portability assumption or default runtime policy.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {
        16: PersistentInputModel(
            ct.models.MLModel(
                "artifacts/qwen3-asr-1.7b-final/decoder_00.mlpackage",
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
        ),
        1: PersistentInputModel(
            ct.models.MLModel(
                "artifacts/probes/decoder-static-t1.mlpackage",
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
        ),
    }
    runtime = CoreMLRuntime.__new__(CoreMLRuntime)
    runtime.embeddings = np.load(
        "artifacts/qwen3-asr-1.7b-final/embedding.npy", mmap_mode="r"
    )
    runtime.residual_scale = 1
    runtime.cache_length = 1024
    phase = np.arange(1024, dtype=np.float32)[:, None] / 1e6 ** (
        np.arange(64, dtype=np.float32)[None] / 64
    )
    runtime.cosine, runtime.sine = np.cos(phase), np.sin(phase)
    shared = copied = None
    report = {
        "hypothesis": "equal named state shapes may be compatible between fixed-width Core ML models",
        "rows": [],
    }
    try:
        shared = models[16].make_state()
        runtime.token_batch_size, runtime.decoders = 16, [models[16]]
        tokens = (151644, 8948, 198, 151645, 198, 151644, 872, 198)
        runtime._decode_step(
            np.concatenate([runtime._embedding(token) for token in tokens], axis=-1),
            0,
            [shared],
        )
        copied = models[1].make_state()
        names = [f"{kind}_{layer}" for layer in range(4) for kind in ("key", "value")]
        for name in names:
            copied.write_state(name, shared.read_state(name))
        runtime.token_batch_size, runtime.decoders = 1, [models[1]]
        for position, token in enumerate(
            (198, 8948, 872, 198, 151644), start=len(tokens)
        ):
            actual = runtime._decode_step(runtime._embedding(token), position, [shared])
            expected = runtime._decode_step(
                runtime._embedding(token), position, [copied]
            )
            error = float(
                np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
            )
            equal_states = all(
                np.array_equal(shared.read_state(name), copied.read_state(name))
                for name in names
            )
            report["rows"].append(
                {
                    "position": position,
                    "max_abs": error,
                    "all_state_arrays_equal": equal_states,
                }
            )
            if error != 0 or not equal_states:
                raise AssertionError("Shared and copied state differ")
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        shared = copied = None
        PersistentInputModel.close_many(list(models.values()))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
