"""Compare exact audio caching with cumulative replay at every streaming update.

Both sides retain separate decoder prefix contexts and identical rollback rules.
Inputs arrive without wall-clock pacing: timings measure inference work, not
microphone waiting. Full Standard ASR event checks live in verify_streaming_runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.diagnostics import inspect_environment
from std_qwen3asr_ane.runtime import CoreMLRuntime, rollback_prefix


def compare_stream(runtime, samples, *, chunk_samples, reverse, max_new_tokens):
    contexts = {mode: runtime.new_decoder_context() for mode in ("fresh", "cached")}
    audio_context = runtime.new_audio_context()
    previous = {"fresh": "", "cached": ""}
    boundaries = list(range(chunk_samples, len(samples), chunk_samples)) + [
        len(samples)
    ]
    for index, boundary in enumerate(boundaries):
        results = {}
        modes = ("cached", "fresh") if (index + reverse) % 2 else ("fresh", "cached")
        for mode in modes:
            prefix = (
                rollback_prefix(runtime.tokenizer, previous[mode], 5)
                if index >= 2
                else ""
            )
            started = perf_counter()
            result = runtime.transcribe(
                samples[:boundary],
                language=None,
                max_new_tokens=max_new_tokens,
                prefix_text=prefix,
                decoder_context=contexts[mode],
                audio_context=audio_context if mode == "cached" else None,
            )
            results[mode] = result
            previous[mode] = result.raw_text
            yield {
                "mode": mode,
                "audio_prefix_seconds": boundary / 16000,
                "seconds": perf_counter() - started,
                "tokens": list(result.token_ids),
                "raw_text": result.raw_text,
                "timings": result.timings,
            }
        if (
            results["fresh"].token_ids != results["cached"].token_ids
            or results["fresh"].raw_text != results["cached"].raw_text
            or results["fresh"].timings["eos_token_id"]
            != results["cached"].timings["eos_token_id"]
        ):
            raise AssertionError(
                f"Audio cache changed streaming output at sample {boundary}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--silence-prefix-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.repeats < 1
        or not np.isfinite(args.chunk_seconds)
        or args.chunk_seconds < 1 / 16000
        or not np.isfinite(args.silence_prefix_seconds)
        or args.silence_prefix_seconds < 0
    ):
        parser.error(
            "Use a fresh output, positive repeats/chunk width and nonnegative prefix"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = None
    records = []
    summary = {
        "complete": False,
        "exact_token_and_raw_text_parity": False,
        "exact_eos_parity": False,
        "environment": inspect_environment(),
        "model_path": str(args.model_dir.resolve()),
        "manifest_sha256": digest(args.model_dir / "manifest.json"),
        "source_sha256": {
            path.name: digest(path)
            for path in sorted(
                (
                    Path(__file__).parents[1] / "std_qwen3asr_ane/src/std_qwen3asr_ane"
                ).glob("*.py")
            )
        },
        "script_sha256": digest(Path(__file__)),
        "chunk_seconds": args.chunk_seconds,
        "silence_prefix_seconds": args.silence_prefix_seconds,
        "max_new_tokens": args.max_new_tokens,
        "realtime_feed": False,
        "close_succeeded": False,
    }
    try:
        runtime = CoreMLRuntime(args.model_dir)
        with args.output.open("x") as output:
            for item in manifest_rows(args.manifest):
                samples, audio_digest = audio_samples(Path(item["audio_path"]))
                samples = np.pad(
                    samples, (round(args.silence_prefix_seconds * 16000), 0)
                )
                pcm_digest = hashlib.sha256(samples.tobytes()).hexdigest()
                for repeat in range(-1, args.repeats):
                    for record in compare_stream(
                        runtime,
                        samples,
                        chunk_samples=round(args.chunk_seconds * 16000),
                        reverse=repeat % 2,
                        max_new_tokens=args.max_new_tokens,
                    ):
                        record.update(
                            id=item["id"],
                            repeat=repeat,
                            audio_sha256=audio_digest,
                            input_pcm_sha256=pcm_digest,
                        )
                        records.append(record)
                        output.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        output.flush()
                    print(
                        json.dumps(
                            {"id": item["id"], "repeat": repeat, "parity": True}
                        ),
                        flush=True,
                    )
        measured = [record for record in records if record["repeat"] >= 0]
        summary["by_sample"] = {
            sample_id: {
                mode: {
                    "median_cumulative_seconds": statistics.median(
                        sum(
                            row["seconds"]
                            for row in measured
                            if row["id"] == sample_id
                            and row["mode"] == mode
                            and row["repeat"] == repeat
                        )
                        for repeat in range(args.repeats)
                    ),
                    "partial_p50_seconds": float(
                        np.median(
                            [
                                row["seconds"]
                                for row in measured
                                if row["id"] == sample_id and row["mode"] == mode
                            ]
                        )
                    ),
                    "partial_p95_seconds": float(
                        np.percentile(
                            [
                                row["seconds"]
                                for row in measured
                                if row["id"] == sample_id and row["mode"] == mode
                            ],
                            95,
                        )
                    ),
                }
                for mode in ("fresh", "cached")
            }
            for sample_id in sorted({row["id"] for row in measured})
        }
        summary.update(
            complete=True, exact_token_and_raw_text_parity=True, exact_eos_parity=True
        )
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
            summary["close_succeeded"] = True
        finally:
            args.output.with_suffix(".summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


if __name__ == "__main__":
    main()
