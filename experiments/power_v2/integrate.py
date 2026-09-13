"""Integrate PSTR samples over explicit Python-monotonic workload boundaries.

Example: python3 experiments/power_v2/integrate.py trace.jsonl --start 123 --end 723 --audio-seconds 600
Optional idle watts are an exploratory subtraction, never device attribution.
"""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path


def integrate(rows, start, end, max_gap=2.5):
    samples = [
        (r["monotonic_s"], r["pstr_w_estimate"]) for r in rows if "pstr_w_estimate" in r
    ]
    if not all(math.isfinite(x) for x in (start, end, max_gap)) or max_gap <= 0:
        raise ValueError(
            "Boundaries and maximum gap must be finite; gap must be positive"
        )
    if end <= start or len(samples) < 2:
        raise ValueError("Need positive interval and at least two PSTR samples")
    if any(not math.isfinite(t) or not math.isfinite(p) or p < 0 for t, p in samples):
        raise ValueError("Invalid PSTR sample")
    if any(b[0] <= a[0] for a, b in pairwise(samples)):
        raise ValueError("Samples must be strictly ordered")
    if samples[0][0] > start or samples[-1][0] < end:
        raise ValueError("Collector must bracket both workload boundaries")
    joules = 0.0
    used_gaps = []
    for (t0, p0), (t1, p1) in pairwise(samples):
        left, right = max(start, t0), min(end, t1)
        if right <= left:
            continue
        gap = t1 - t0
        if gap > max_gap:
            raise ValueError(f"Unobserved PSTR gap {gap:.3f}s exceeds {max_gap}s")
        slope = (p1 - p0) / gap
        joules += (
            ((p0 + slope * (left - t0)) + (p0 + slope * (right - t0)))
            * (right - left)
            / 2
        )
        used_gaps.append(gap)
    battery_rows = [
        r for r in rows if start <= r.get("monotonic_s", -1) <= end and "battery" in r
    ]
    battery_only = bool(battery_rows) and all(
        r["battery"].get("ExternalConnected") is False
        and r["battery"].get("IsCharging") is False
        for r in battery_rows
    )
    return {
        "measurement": "whole-machine SMC PSTR estimate",
        "gross_j_estimate": joules,
        "duration_s": end - start,
        "mean_w_estimate": joules / (end - start),
        "max_sample_gap_s": max(used_gaps),
        "battery_only_observed": battery_only,
        "device_attribution_valid": False,
        "calibrated_energy_gate_valid": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--audio-seconds", type=float, required=True)
    parser.add_argument("--idle-watts", type=float)
    parser.add_argument("--max-gap", type=float, default=2.5)
    args = parser.parse_args()
    if args.audio_seconds <= 0 or not math.isfinite(args.audio_seconds):
        parser.error("audio-seconds must be finite and positive")
    if args.idle_watts is not None and (
        args.idle_watts < 0 or not math.isfinite(args.idle_watts)
    ):
        parser.error("idle-watts must be finite and nonnegative")
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    result = integrate(rows, args.start, args.end, args.max_gap)
    result["audio_seconds"] = args.audio_seconds
    result["gross_j_per_audio_second_estimate"] = (
        result["gross_j_estimate"] / args.audio_seconds
    )
    if args.idle_watts is not None:
        adjusted = result["gross_j_estimate"] - args.idle_watts * result["duration_s"]
        result.update(
            idle_watts=args.idle_watts,
            idle_adjusted_j_exploratory=adjusted,
            idle_adjusted_j_per_audio_second_exploratory=adjusted / args.audio_seconds,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
