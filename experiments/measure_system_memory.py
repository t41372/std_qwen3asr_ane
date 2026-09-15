"""System-wide memory residency of loading a bundle and transcribing once.

Process footprints miss Neural Engine allocations, so this samples the kernel's
page counters (vm_stat) before load, after load and after inference. The delta
is an upper bound that includes any concurrent activity; run on a quiet machine
and report the idle drift measured by the same script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from evaluate import audio_samples, manifest_rows

PAGE_KEYS = (
    "Pages active",
    "Pages wired down",
    "Pages occupied by compressor",
    "Pages free",
)


def vm_stat() -> dict[str, int]:
    text = subprocess.run(
        ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True
    ).stdout
    page_size = int(text.splitlines()[0].split("page size of")[1].split()[0])
    values = {}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        key, count = line.split(":", 1)
        values[key.strip()] = int(count.strip().rstrip(".")) * page_size
    values["page_size"] = page_size
    return values


def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, float]:
    return {key: (after[key] - before[key]) / 2**20 for key in PAGE_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("coreml", "mlx"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--draft-dir",
        type=Path,
        default=None,
        help="coreml: GPU-draft bundle (speculative path)",
    )
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if args.output.exists() or args.max_new_tokens < 1:
        parser.error("Use a fresh output and positive token budget")
    inputs = [
        audio_samples(Path(row["audio_path"]))[0]
        for row in manifest_rows(args.manifest)
    ]
    time.sleep(3)
    idle_start = vm_stat()
    time.sleep(5)
    before = vm_stat()
    report = {
        "backend": args.backend,
        "model_dir": str(args.model_dir),
        "draft_dir": None if args.draft_dir is None else str(args.draft_dir),
        "idle_drift_mib": delta(before, idle_start),
        "max_new_tokens": args.max_new_tokens,
    }
    if args.backend == "coreml":
        from std_qwen3asr_ane.runtime import CoreMLRuntime

        runtime = CoreMLRuntime(args.model_dir)
        if args.draft_dir is not None:
            from std_qwen3asr_ane.draft import DraftRuntime

            draft = DraftRuntime(args.draft_dir, runtime, quantize_bits=args.draft_bits)
            transcribe = lambda s: (
                runtime.transcribe_speculative(
                    s,
                    draft,
                    language=None,
                    max_new_tokens=args.max_new_tokens,
                    lookahead=15,
                ).text
            )

            def close():
                draft.close()
                runtime.close()
        else:
            transcribe = lambda s: (
                runtime.transcribe(
                    s, language=None, max_new_tokens=args.max_new_tokens
                ).text
            )
            close = runtime.close
    else:
        from benchmark_mlx import configure_local_caches, load_backend

        configure_local_caches()
        predict, _ = load_backend(args.model_dir.resolve(), 20260912)
        transcribe = lambda s: predict(s, None, args.max_new_tokens)["hypothesis"]
        close = None
    loaded = vm_stat()
    report["after_load_mib"] = delta(loaded, before)
    for samples in inputs:
        transcribe(samples)
    inferred = vm_stat()
    report["after_inference_mib"] = delta(inferred, before)
    for samples in inputs:
        transcribe(samples)
    report["after_second_pass_mib"] = delta(vm_stat(), before)
    if close is not None:
        close()
        report["close_succeeded"] = True
        report["after_close_mib"] = delta(vm_stat(), before)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
