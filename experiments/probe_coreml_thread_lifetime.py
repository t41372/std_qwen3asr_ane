"""Isolate Core ML NumPy lifetime behavior with one existing decoder layer."""

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/probes/decoder-layer-0.mlpackage")
    )
    parser.add_argument(
        "--mode", choices=("transient", "retain", "persistent"), required=True
    )
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--linger", type=float, default=3)
    parser.add_argument("--idle-seconds", type=float, default=0)
    parser.add_argument(
        "--prediction-thread", choices=("main", "worker"), default="worker"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.open("x")

    def record(event, **fields):
        line = json.dumps({"event": event, "time": time.monotonic(), **fields})
        log.write(line + "\n")
        log.flush()
        print(line, flush=True)

    import coremltools as ct

    record("load_start", mode=args.mode)
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
    if args.mode == "persistent":
        from std_qwen3asr_ane.runtime import PersistentInputModel

        model = PersistentInputModel(model)
    spec = model.get_spec()
    shapes = {
        item.name: tuple(item.type.multiArrayType.shape)
        for item in spec.description.input
    }
    types = {
        item.name: np.float16
        if item.type.multiArrayType.dataType == 65552
        else np.float32
        for item in spec.description.input
    }
    inputs = {
        name: np.zeros(shape, dtype=types[name]) for name, shape in shapes.items()
    }
    state = model.make_state() if spec.description.state else None
    retained = []
    record("loaded", shapes=shapes)

    def predictions():
        for index in range(args.iterations):
            fresh = {
                name: np.zeros(shape, dtype=types[name])
                for name, shape in shapes.items()
            }
            if "hidden_states" in fresh:
                fresh["hidden_states"].fill(0.001)
            if "cosine" in fresh:
                fresh["cosine"].fill(1)
                position = index % shapes["update_mask"][-1]
                fresh["attention_mask"][..., position + 1 :] = -1e4
                fresh["update_mask"][..., position] = 1
            for name in ("conv1_mask", "conv2_mask"):
                if name in fresh:
                    fresh[name].fill(1)
            if args.mode == "persistent":
                for name, value in fresh.items():
                    np.copyto(inputs[name], value)
                submitted = inputs
            else:
                submitted = fresh
            if args.mode == "retain":
                retained.append(submitted)
            prediction = (
                model.predict(submitted, state=state)
                if state is not None
                else model.predict(submitted)
            )
            output = np.array(next(iter(prediction.values())), copy=True)
            record("prediction", index=index, finite=bool(np.isfinite(output).all()))
            if args.idle_seconds:
                del fresh, submitted, prediction, output
                gc.collect()
                time.sleep(args.idle_seconds)
            if index % 8 == 0:
                gc.collect()
                time.sleep(0.005)

    if args.prediction_thread == "main":
        predictions()
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(predictions).result()
    record("predictions_finished")
    time.sleep(args.linger)
    record("pre_close_linger_finished")
    state = None
    if args.mode == "persistent":
        model.close()
    model = None
    gc.collect()
    record("model_released_inputs_retained")
    time.sleep(args.linger)
    retained.clear()
    inputs.clear()
    gc.collect()
    record("inputs_released")
    time.sleep(args.linger)
    record("completed")
    log.close()


if __name__ == "__main__":
    main()
