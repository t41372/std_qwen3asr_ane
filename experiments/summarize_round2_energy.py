"""Summarize equal-work ABBA energy without hiding idle drift or sensor limits."""

import argparse
import json
import statistics
from pathlib import Path

from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blocks, identities, expected = {}, {}, None
    for name in ("A1", "B1", "B2", "A2"):
        folder = args.root / name
        source = folder / "summary.json"
        report = json.loads(source.read_text())
        predictions = [
            json.loads(line)
            for line in (folder / "predictions.jsonl").read_text().splitlines()
        ]
        identity = [
            (
                row["id"],
                row["repeat"],
                row["audio_sha256"],
                row["audio_seconds"],
                row["text"],
            )
            for row in predictions
        ]
        if expected is None:
            expected = identity
        if (
            report.get("error")
            or report["completed"] != 120
            or len(predictions) != 120
            or identity != expected
        ):
            raise ValueError(
                "Energy blocks differ in work, output, coverage or success"
            )
        model_hash = report["model_manifest_sha256"]
        identities.setdefault(name[0], model_hash)
        if identities[name[0]] != model_hash or report["max_new_tokens"] != 128:
            raise ValueError("Model identity or output budget changed between blocks")
        rows = [
            json.loads(line)
            for line in (folder / "power.jsonl").read_text().splitlines()
        ]
        before = [
            row["pstr_w_estimate"]
            for row in rows
            if "pstr_w_estimate" in row
            and row["monotonic_s"] < report["start_monotonic_s"] - 2
        ]
        after = [
            row["pstr_w_estimate"]
            for row in rows
            if "pstr_w_estimate" in row
            and row["monotonic_s"] > report["end_monotonic_s"] + 2
        ]
        blocks[name] = {
            "source_sha256": digest(source),
            "power_sha256": digest(folder / "power.jsonl"),
            "predictions_sha256": digest(folder / "predictions.jsonl"),
            "gross_j_per_audio_second": report["gross_j_per_audio_second"],
            "workload_mean_w_estimate": report["power"]["mean_w_estimate"],
            "idle_before_mean_w_estimate": statistics.mean(before),
            "idle_after_mean_w_estimate": statistics.mean(after),
            "boundary_j_per_audio_second": {
                shift: value / report["audio_seconds"]
                for shift, value in report["boundary_shift_j"].items()
            },
            "process_cpu": report["process_cpu"],
            "audio_seconds": report["audio_seconds"],
            "work_seconds": report["power"]["duration_s"],
        }
    baseline = statistics.median(
        blocks[key]["gross_j_per_audio_second"] for key in ("A1", "A2")
    )
    candidate = statistics.median(
        blocks[key]["gross_j_per_audio_second"] for key in ("B1", "B2")
    )
    baseline_low = min(
        value
        for key in ("A1", "A2")
        for value in blocks[key]["boundary_j_per_audio_second"].values()
    )
    candidate_high = max(
        value
        for key in ("B1", "B2")
        for value in blocks[key]["boundary_j_per_audio_second"].values()
    )
    result = {
        "complete": True,
        "equal_work_and_output": True,
        "blocks": blocks,
        "model_manifest_sha256": identities,
        "baseline_median_j_per_audio_second": baseline,
        "candidate_median_j_per_audio_second": candidate,
        "reduction_fraction": 1 - candidate / baseline,
        "all_candidate_blocks_below_all_baseline_blocks_with_2s_boundary_shifts": candidate_high
        < baseline_low,
        "calibrated_measurement": False,
        "limitations": [
            "Whole-machine SMC PSTR estimate, not a wall meter or ANE-only energy.",
            "Desktop activity persisted and idle brackets varied; this is not a low-background system-idle measurement.",
            "Two blocks per model do not establish an energy confidence interval or causal per-device attribution.",
            "Do not compare directly with earlier runs using different audio workloads or desktop conditions.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
