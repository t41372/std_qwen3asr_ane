"""Capture bounded real ASR head inputs from a disjoint calibration manifest."""

import argparse
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
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        parser.error("Use fresh output paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = CoreMLRuntime(args.model_dir)
    original = runtime.lm_head.predict
    utterance_inputs, selected, records = [], [], []
    summary = {
        "complete": False,
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "calibration_manifest_sha256": digest(args.manifest),
        "selection": "first, middle and final greedy head call of each utterance",
        "language_mode": "auto",
        "max_new_tokens": 256,
        "samples": records,
    }

    def record(data):
        utterance_inputs.append(
            np.array(data["hidden_states"], dtype=np.float32, copy=True)
        )
        return original(data)

    runtime.lm_head.predict = record
    try:
        for item in manifest_rows(args.manifest):
            audio, audio_digest = audio_samples(Path(item["audio_path"]))
            if audio_digest != item["audio_sha256"]:
                raise ValueError("Calibration audio hash mismatch")
            utterance_inputs.clear()
            runtime.transcribe(audio, language=None, max_new_tokens=256)
            for index in sorted(
                {0, len(utterance_inputs) // 2, len(utterance_inputs) - 1}
            ):
                selected.append(utterance_inputs[index].reshape(-1))
                records.append(
                    {
                        "id": item["id"],
                        "language": item["language"],
                        "head_call": index,
                        "audio_sha256": audio_digest,
                    }
                )
            print(
                json.dumps({"id": item["id"], "head_calls": len(utterance_inputs)}),
                flush=True,
            )
        np.savez_compressed(args.output, hidden_states=np.stack(selected))
        summary.update(
            complete=True,
            input_archive_sha256=digest(args.output),
            input_count=len(selected),
        )
    finally:
        runtime.lm_head.predict = original
        original = None
        try:
            runtime.close()
            summary["close_succeeded"] = True
        finally:
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
