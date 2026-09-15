"""Alternate serial head candidates while retaining identical decoder handles."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import coremltools as ct
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest, language_head_output
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Use fresh output and positive repeats")
    candidate_manifest = json.loads((args.candidate / "manifest.json").read_text())
    descriptor = language_head_output(candidate_manifest)
    parent = candidate_manifest.get(
        "head_quantization", candidate_manifest.get("compact_head_parent", {})
    )
    expected_parent = parent.get(
        "parent_manifest_sha256", parent.get("manifest_sha256")
    )
    if expected_parent != digest(args.baseline / "manifest.json"):
        raise ValueError("Candidate head is not bound to this baseline")
    path = (args.candidate / candidate_manifest["files"]["lm_head"]).resolve()
    if not path.is_relative_to(args.candidate.resolve()) or path.suffix != ".mlmodelc":
        raise ValueError("A compiled candidate head inside its bundle is required")
    summary = {
        "complete": False,
        "all_exact_token_parity": True,
        "baseline_manifest_sha256": digest(args.baseline / "manifest.json"),
        "candidate_manifest_sha256": digest(args.candidate / "manifest.json"),
        "method": "alternating heads on one unchanged decoder runtime; private states per utterance",
        "script_sha256": digest(Path(__file__)),
    }
    runtime, candidate_head, baseline_head = None, None, None
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime = CoreMLRuntime(args.baseline)
        baseline_head, baseline_descriptor = runtime.lm_head, runtime.head_output
        candidate_head = PersistentInputModel(
            ct.models.CompiledMLModel(
                str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        )
        with args.output.open("x") as output:
            for item in manifest_rows(args.manifest):
                audio, audio_hash = audio_samples(Path(item["audio_path"]))
                for repeat in range(-1, args.repeats):
                    tokens = {}
                    order = (
                        ("baseline", "candidate")
                        if repeat % 2 == 0
                        else ("candidate", "baseline")
                    )
                    for mode in order:
                        runtime.lm_head = (
                            baseline_head if mode == "baseline" else candidate_head
                        )
                        runtime.head_output = (
                            baseline_descriptor if mode == "baseline" else descriptor
                        )
                        started = perf_counter()
                        result = runtime.transcribe(
                            audio, language=None, max_new_tokens=256
                        )
                        record = {
                            "id": item["id"],
                            "repeat": repeat,
                            "mode": mode,
                            "audio_sha256": audio_hash,
                            "seconds": perf_counter() - started,
                            "tokens": list(result.token_ids),
                            "text": result.text,
                            "timings": result.timings,
                        }
                        records.append(record)
                        tokens[mode] = result.token_ids
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                    summary["all_exact_token_parity"] &= (
                        tokens["baseline"] == tokens["candidate"]
                    )
                    print(json.dumps({"id": item["id"], "repeat": repeat}), flush=True)
        summary["medians"] = {
            sample: {
                mode: {
                    metric: statistics.median(
                        row["timings"][metric]
                        for row in records
                        if row["id"] == sample
                        and row["mode"] == mode
                        and row["repeat"] >= 0
                    )
                    for metric in (
                        "total_seconds",
                        "head_seconds",
                        "generation_seconds",
                    )
                }
                for mode in ("baseline", "candidate")
            }
            for sample in sorted({row["id"] for row in records})
        }
        summary["complete"] = True
    finally:
        if runtime is not None and baseline_head is not None:
            runtime.lm_head = baseline_head
        try:
            if candidate_head is not None:
                candidate_head.close()
        finally:
            try:
                if runtime is not None:
                    runtime.close()
                summary["close_succeeded"] = True
            finally:
                args.output.with_suffix(".summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
