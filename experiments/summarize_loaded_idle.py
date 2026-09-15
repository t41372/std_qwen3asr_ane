"""Compare completed idle runs, retaining timing coverage and global-memory limits."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder):
    source, power = folder / "summary.json", folder / "power.jsonl"
    report = json.loads(source.read_text())
    if not report["complete"] or not report["close_succeeded"]:
        raise ValueError(f"Run is unfinished or failed: {folder}")
    if [row["requested_idle_seconds"] for row in report["wake"]] != [60, 600, 3600]:
        raise ValueError("Expected the actual independent 1/10/60-minute ages")
    if (
        report["draft_loaded_by_prepare"]
        or report["draft_loaded_after_warmup"] != report["draft_configured"]
    ):
        raise ValueError("Observed draft lifecycle differs from the lazy-load policy")
    groups = defaultdict(list)
    for line in power.read_text().splitlines():
        row = json.loads(line)
        if "pstr_w_estimate" in row:
            groups[row["phase"]].append(row)
    coverage = {}
    for phase in (report["unloaded"], *report["idle"], report["released"]):
        rows = groups[phase["phase"]]
        if len(rows) < 2 or not all(
            math.isfinite(row["pstr_w_estimate"]) and row["pstr_w_estimate"] >= 0
            for row in rows
        ):
            raise ValueError("Power samples are missing or invalid")
        gaps = [
            right["monotonic_seconds"] - left["monotonic_seconds"]
            for left, right in pairwise(rows)
        ]
        if min(gaps) <= 0:
            raise ValueError("Power timestamps are not ordered")
        span = rows[-1]["monotonic_seconds"] - rows[0]["monotonic_seconds"]
        coverage[phase["phase"]] = {
            "samples": len(rows),
            "sample_span_seconds": span,
            "max_gap_seconds": max(gaps),
            "wall_minus_monotonic_span_seconds": rows[-1]["wall_time"]
            - rows[0]["wall_time"]
            - span,
            "mean_w_estimate": phase["mean_w_estimate"],
            "process_cpu_seconds_including_sampler": phase["process_cpu_seconds"],
        }
    return {
        "source_sha256": digest(source),
        "power_sha256": digest(power),
        "model_manifest_sha256": report["model_manifest_sha256"],
        "audio_sha256": report["audio_sha256"],
        "draft_configured": report["draft_configured"],
        "draft_loaded_by_prepare": report["draft_loaded_by_prepare"],
        "draft_loaded_after_warmup": report["draft_loaded_after_warmup"],
        "first_batch_seconds": report["first_batch_seconds"],
        "warmed_request_seconds": report["warmed_request_seconds"],
        "wake": [
            {
                **row,
                "seconds_above_warm_reference": row["request_seconds"]
                - report["warmed_request_seconds"],
            }
            for row in report["wake"]
        ],
        "power_and_coverage": coverage,
        "after_prepare_mib": report["after_prepare_mib"],
        "after_warmup_mib": report["after_warmup_mib"],
        "after_close_mib": report["after_close_mib"],
        "wired_drop_across_close_mib": report["wake"][-1]["system_memory_mib"][
            "Pages wired down"
        ]
        - report["after_close_mib"]["Pages wired down"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {
        name: summarize(args.root / f"idle-{name}") for name in ("target", "draft")
    }
    if (
        conditions["target"]["draft_configured"]
        or not conditions["draft"]["draft_configured"]
    ):
        raise ValueError("Expected one target-only and one draft-loaded condition")
    for key in ("model_manifest_sha256", "audio_sha256"):
        if conditions["target"][key] != conditions["draft"][key]:
            raise ValueError(f"Conditions differ in {key}")
    result = {
        "complete": True,
        "all_wake_texts_match": all(
            row["text_matches_warmup"]
            for condition in conditions.values()
            for row in condition["wake"]
        ),
        "all_actual_idle_ages_at_least_requested": all(
            row["idle_seconds"] >= row["requested_idle_seconds"]
            for condition in conditions.values()
            for row in condition["wake"]
        ),
        "conditions": conditions,
        "limitations": [
            "One fixed clip and one wake per age/condition; these are observations, not latency percentiles.",
            "Whole-machine PSTR includes changing desktop activity; differences are not isolated model idle power.",
            "vm_stat is global. Long-interval drift and the change across close cannot be assigned entirely to this process.",
            "Sampler CPU is included in this process CPU measurement; Core ML daemons and kernel work are not.",
            "No automatic eviction or ASR+LLM contention experiment was added.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
