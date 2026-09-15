"""Measure loaded residency, idle power and first-request latency after real idle.

Run this on a quiet machine after other model work ends. Each requested interval
starts after the preceding inference has finished, so 60/600/3600 seconds are
independent idle ages rather than timestamps since process launch. PSTR is a
whole-machine estimate; process CPU and system memory drift are reported too.
"""

import argparse
import hashlib
import json
import resource
import statistics
import subprocess
import time
from pathlib import Path

from evaluate import audio_samples, manifest_rows
from measure_system_memory import delta, vm_stat
from power_v2.whole_machine import SystemPower, battery_sample
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.plugin import Qwen3ASREngine


def process_cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def observe_idle(seconds, phase, power, output):
    started, cpu_started, memory = time.monotonic(), process_cpu(), vm_stat()
    watts = []
    next_sample = started
    while True:
        now = time.monotonic()
        if now >= started + seconds:
            break
        sample = {"phase": phase, "monotonic_seconds": now, "wall_time": time.time()}
        if power is not None:
            sample["pstr_w_estimate"] = power.watts()
            watts.append(sample["pstr_w_estimate"])
        output.write(json.dumps(sample, allow_nan=False) + "\n")
        output.flush()
        next_sample += 1
        time.sleep(max(0, min(next_sample, started + seconds) - time.monotonic()))
    elapsed = time.monotonic() - started
    result = {
        "phase": phase,
        "requested_seconds": seconds,
        "actual_seconds": elapsed,
        "power_samples": len(watts),
        "mean_w_estimate": statistics.mean(watts) if watts else None,
        "min_w_estimate": min(watts) if watts else None,
        "max_w_estimate": max(watts) if watts else None,
        "process_cpu_seconds": process_cpu() - cpu_started,
        "system_memory_drift_mib": delta(vm_stat(), memory),
    }
    try:
        result["battery_at_end"] = battery_sample()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        result["battery_error"] = str(error)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--idle-seconds", type=float, nargs="+", default=[60, 600, 3600]
    )
    parser.add_argument("--bracket-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.output.exists() or any(
        not 0 < seconds <= 3600
        for seconds in [*args.idle_seconds, args.bracket_seconds]
    ):
        parser.error("Use a fresh output and intervals in (0, 3600] seconds")
    args.output.mkdir(parents=True)
    item = manifest_rows(args.manifest)[0]
    audio, audio_hash = audio_samples(Path(item["audio_path"]))
    report = {
        "complete": False,
        "model_manifest_sha256": digest(args.model_dir / "manifest.json"),
        "audio_sha256": audio_hash,
        "audio_id": item["id"],
        "draft_configured": args.draft_dir is not None,
        "script_sha256": digest(Path(__file__)),
        "method": "one warmed engine; first request after each independent idle interval",
        "idle": [],
        "wake": [],
        "close_succeeded": False,
    }
    engine, power = None, None
    try:
        try:
            power = SystemPower()
        except Exception as error:  # noqa: BLE001 — unavailable telemetry is not zero power.
            report["power_error"] = str(error)
        try:
            report["battery_before"] = battery_sample()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            report["battery_error"] = str(error)
        with (args.output / "power.jsonl").open("x") as output:
            report["unloaded"] = observe_idle(
                args.bracket_seconds, "unloaded", power, output
            )
            before = vm_stat()
            engine = Qwen3ASREngine(model_dir=args.model_dir, draft_dir=args.draft_dir)
            started = time.monotonic()
            engine.prepare()
            report["prepare_seconds"] = time.monotonic() - started
            report["draft_loaded_by_prepare"] = engine._draft is not None
            report["after_prepare_mib"] = delta(vm_stat(), before)
            started = time.monotonic()
            expected = engine.transcribe((audio, 16000)).text
            report["first_batch_seconds"] = time.monotonic() - started
            report["draft_loaded_after_warmup"] = engine._draft is not None
            started = time.monotonic()
            warmed = engine.transcribe((audio, 16000))
            last_request_finished = time.monotonic()
            report["warmed_request_seconds"] = last_request_finished - started
            if warmed.text != expected:
                raise RuntimeError("Warm requests produced different text")
            report["after_warmup_mib"] = delta(vm_stat(), before)
            for seconds in args.idle_seconds:
                idle = observe_idle(seconds, f"loaded_{seconds:g}s", power, output)
                report["idle"].append(idle)
                started = time.monotonic()
                result = engine.transcribe((audio, 16000))
                finished = time.monotonic()
                wake = {
                    "requested_idle_seconds": seconds,
                    "idle_seconds": started - last_request_finished,
                    "request_seconds": finished - started,
                    "text_matches_warmup": result.text == expected,
                    "text_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                    "system_memory_mib": delta(vm_stat(), before),
                }
                last_request_finished = finished
                report["wake"].append(wake)
                (args.output / "summary.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps(wake), flush=True)
            engine.close()
            report["close_succeeded"] = True
            report["released"] = observe_idle(
                args.bracket_seconds, "released", power, output
            )
            report["after_close_mib"] = delta(vm_stat(), before)
        report["complete"] = all(row["text_matches_warmup"] for row in report["wake"])
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if engine is not None and not report["close_succeeded"]:
                engine.close()
                report["close_succeeded"] = True
        finally:
            if power is not None:
                power.close()
            (args.output / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
