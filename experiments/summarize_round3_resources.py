"""Summarize complete independent resource runs without treating missing data as zero."""

import argparse
import json
from pathlib import Path

import numpy as np


def memory_metrics(report):
    if not report.get("complete") or not report.get("close_succeeded"):
        raise ValueError("Incomplete memory run")
    stages = {row["stage"]: row for row in report["snapshots"]}
    before, warm = stages["before_load"], stages["after_request_2"]
    result = {
        "rss_delta_mib": (warm["rss_bytes"] - before["rss_bytes"]) / 2**20,
        "footprint_delta_mib": (warm["phys_footprint_bytes"] - before["phys_footprint_bytes"])
        / 2**20,
        "peak_footprint_mib": max(row["peak_phys_footprint_bytes"] for row in stages.values())
        / 2**20,
        "after_close_footprint_mib": stages["after_close_2s"]["phys_footprint_bytes"] / 2**20,
    }
    for label, name in (
        ("wired", "Pages wired down"),
        ("compressed", "Pages occupied by compressor"),
        ("file_backed", "File-backed pages"),
        ("purgeable", "Pages purgeable"),
    ):
        first, second = before["system_bytes"][name], warm["system_bytes"][name]
        result[f"{label}_delta_mib"] = (
            None if first is None or second is None else (second - first) / 2**20
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    series = json.loads((args.root / "summary.json").read_text())
    if not series["complete"]:
        raise ValueError("Resource series did not complete")
    report = {"memory": None, "energy": None}
    if series["memory"]:
        runs = {}
        for row in series["memory"]:
            key = (row["pair"], row["mode"])
            if key in runs:
                raise ValueError("Duplicate memory run")
            runs[key] = memory_metrics(json.loads((args.root / row["report"]).read_text()))
        if set(runs) != {(pair, mode) for pair in range(5) for mode in ("baseline", "candidate")}:
            raise ValueError("Five complete memory pairs are required")
        names = next(iter(runs.values())).keys()
        report["memory"] = {"medians": {}, "paired_deltas_mib": {}}
        for mode in ("baseline", "candidate"):
            report["memory"]["medians"][mode] = {
                name: float(np.median([runs[pair, mode][name] for pair in range(5)]))
                if all(runs[pair, mode][name] is not None for pair in range(5))
                else None
                for name in names
            }
        for name in names:
            report["memory"]["paired_deltas_mib"][name] = (
                [runs[pair, "candidate"][name] - runs[pair, "baseline"][name] for pair in range(5)]
                if all(
                    runs[pair, mode][name] is not None
                    for pair in range(5)
                    for mode in ("baseline", "candidate")
                )
                else None
            )
    if series["energy"]:
        report["energy"] = {
            key: series[key]
            for key in (
                "energy_medians",
                "paired_energy_relative_changes",
                "paired_energy_median_relative_change",
                "paired_energy_median_ci95",
                "energy_interval_unit",
            )
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
