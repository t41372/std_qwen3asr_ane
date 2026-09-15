"""Measure process and system residency across load, inference and explicit close."""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

from evaluate import audio_samples, manifest_rows
from measure_system_memory import vm_stat

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime

PAGE_NAMES = (
    "Pages free",
    "Pages active",
    "Pages inactive",
    "Pages speculative",
    "Pages wired down",
    "Pages purgeable",
    "File-backed pages",
    "Anonymous pages",
    "Pages stored in compressor",
    "Pages occupied by compressor",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--sampler", type=Path, default=Path("artifacts/evaluation/round3/process-memory")
    )
    args = parser.parse_args()
    if args.output.exists() or args.passes < 2:
        parser.error("Use a new output and at least two passes")
    inputs = [audio_samples(Path(row["audio_path"]))[0] for row in manifest_rows(args.manifest)]
    report = {
        "complete": False,
        "bundle_manifest_sha256": digest(args.bundle / "manifest.json"),
        "audio_manifest_sha256": digest(args.manifest),
        "pid": os.getpid(),
        "sampler_sha256": digest(args.sampler),
        "passes": args.passes,
        "snapshots": [],
        "max_new_tokens": args.max_new_tokens,
    }

    def sample(stage):
        pages = vm_stat()
        process = json.loads(
            subprocess.check_output([str(args.sampler), str(os.getpid())], text=True)
        )
        record = {
            "stage": stage,
            "monotonic_seconds": time.monotonic(),
            **process,
            "system_bytes": {key: pages.get(key) for key in PAGE_NAMES},
        }
        report["snapshots"].append(record)
        print(json.dumps(record), flush=True)

    runtime = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sample("idle_start")
        time.sleep(3)
        sample("before_load")
        runtime = CoreMLRuntime(args.bundle)
        sample("after_load")
        for repeat in range(args.passes):
            for audio in inputs:
                runtime.transcribe(audio, language=None, max_new_tokens=args.max_new_tokens)
            if repeat < 2 or repeat + 1 == args.passes:
                sample(f"after_pass_{repeat + 1}")
        runtime.close()
        runtime = None
        gc.collect()
        report["close_succeeded"] = True
        report["close_includes_host_reference_release"] = True
        sample("after_close")
        time.sleep(2)
        sample("after_close_2s")
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
