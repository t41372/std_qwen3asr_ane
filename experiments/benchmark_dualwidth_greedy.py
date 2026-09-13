"""Compare ordinary 1.7B greedy decoding with narrow versus wide prefill.

Both modes use the production greedy loop and the same generation model handles.
Only prompt preparation changes. There is no draft model or oracle transcript.
State sharing remains an experiment on these validated, compatible artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter

from benchmark_speculative import validate_state_transfer
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.diagnostics import inspect_environment
from std_qwen3asr_ane.runtime import CoreMLRuntime


class PrefillComparisonRuntime(CoreMLRuntime):
    """Route batch prompt preparation while retaining the normal greedy loop."""

    def __init__(self, generation_dir: Path, prefill_dir: Path):
        started = perf_counter()
        super().__init__(generation_dir)
        self.generation_load_seconds = perf_counter() - started
        self.use_wide_prefill = False
        try:
            started = perf_counter()
            self.wide_prefill = CoreMLRuntime(prefill_dir)
            self.prefill_load_seconds = perf_counter() - started
            validate_state_transfer(self.wide_prefill, self)
        except BaseException:
            if hasattr(self, "wide_prefill"):
                self.wide_prefill.close()
            super().close()
            raise

    def prepare_prompt(self, *args, **kwargs):
        if kwargs.get("decoder_context") is not None:
            raise ValueError(
                "This experiment covers batch decoding, not streaming contexts"
            )
        if self.use_wide_prefill:
            return self.wide_prefill.prepare_prompt(*args, **kwargs)
        return super().prepare_prompt(*args, **kwargs)

    def close(self, *, timeout=5.0):
        self.wide_prefill.close(timeout=timeout)
        super().close(timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", type=Path, default=Path("artifacts/qwen3-asr-1.7b-compiled")
    )
    parser.add_argument(
        "--prefill",
        type=Path,
        default=Path("artifacts/qwen3-asr-1.7b-prefill64-compiled"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Use positive repeats and a fresh output path")
    inputs = [
        (row, *audio_samples(Path(row["audio_path"])))
        for row in manifest_rows(args.manifest)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = PrefillComparisonRuntime(args.generation, args.prefill)
    records = []
    summary = {
        "purpose": "ordinary Qwen3-ASR-1.7B greedy decoding; no draft model",
        "environment": inspect_environment(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(
            Path(__file__)
            .parents[1]
            .joinpath("std_qwen3asr_ane/src/std_qwen3asr_ane/runtime.py")
            .read_bytes()
        ).hexdigest(),
        "generation_manifest_sha256": hashlib.sha256(
            (args.generation / "manifest.json").read_bytes()
        ).hexdigest(),
        "prefill_manifest_sha256": hashlib.sha256(
            (args.prefill / "manifest.json").read_bytes()
        ).hexdigest(),
        "generation_load_seconds": runtime.generation_load_seconds,
        "additional_prefill_load_seconds": runtime.prefill_load_seconds,
        "all_exact_token_parity": True,
        "expected_pairs": len(inputs) * (args.repeats + 1),
        "verified_pairs": 0,
        "complete": False,
        "close_succeeded": False,
    }
    try:
        with args.output.open("x") as output:
            for row, samples, digest in inputs:
                for repeat in range(-1, args.repeats):
                    results = {}
                    modes = (
                        ("narrow", "wide") if repeat % 2 == 0 else ("wide", "narrow")
                    )
                    for mode in modes:
                        runtime.use_wide_prefill = mode == "wide"
                        result = runtime.transcribe(
                            samples, language=None, max_new_tokens=256
                        )
                        results[mode] = result.token_ids
                        record = {
                            "id": row["id"],
                            "repeat": repeat,
                            "mode": mode,
                            "audio_sha256": digest,
                            "audio_seconds": len(samples) / 16000,
                            "text": result.text,
                            "tokens": list(result.token_ids),
                            "timings": result.timings,
                        }
                        records.append(record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        print(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "repeat": repeat,
                                    "mode": mode,
                                    "seconds": result.timings["total_seconds"],
                                }
                            ),
                            flush=True,
                        )
                    if results["narrow"] != results["wide"]:
                        summary["all_exact_token_parity"] = False
                        raise AssertionError(
                            "Wide prefill changed serial target token IDs"
                        )
                    summary["verified_pairs"] += 1
        summary["medians"] = {
            row["id"]: {
                mode: {
                    metric: statistics.median(
                        record["timings"][metric]
                        for record in records
                        if record["id"] == row["id"]
                        and record["mode"] == mode
                        and record["repeat"] >= 0
                    )
                    for metric in (
                        "total_seconds",
                        "prefill_seconds",
                        "generation_seconds",
                    )
                }
                for mode in ("narrow", "wide")
            }
            for row, _, _ in inputs
        }
        summary["complete"] = summary["verified_pairs"] == summary["expected_pairs"]
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            runtime.close()
            summary["close_succeeded"] = True
        finally:
            args.output.with_suffix(".summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
