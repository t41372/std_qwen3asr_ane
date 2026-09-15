"""Five independent memory pairs and randomized ABBA whole-machine energy blocks."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--energy-repeats", type=int, default=40)
    parser.add_argument("--mode", choices=("memory", "energy", "both"), default="both")
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    utterances = sum(bool(line.strip()) for line in args.manifest.read_text().splitlines())
    if not utterances:
        parser.error("Energy/memory manifest must not be empty")
    report = {
        "complete": False,
        "energy": [],
        "memory": [],
        "seed": 20260915,
        "manifest_sha256": digest(args.manifest),
        "model_manifests": {name: digest(path / "manifest.json") for name, path in paths.items()},
        "baseline_runtime_sha256": digest(args.baseline_source / "runtime.py")
        if args.baseline_source
        else None,
        "energy_measurement": "uncalibrated whole-machine PSTR, not per-device attribution",
        "memory_protocol": "five independent pairs; first pair includes at least 100 requests; others use two passes; warm deltas sampled after request 2",
    }

    def run(label, script, options, *, baseline=False):
        environment = []
        for command in (
            ["pmset", "-g", "batt"],
            ["pmset", "-g", "therm"],
            ["pmset", "-g", "custom"],
        ):
            status = subprocess.run(command, capture_output=True, text=True, check=False)
            environment.append(
                {
                    "command": command,
                    "returncode": status.returncode,
                    "stdout": status.stdout,
                    "stderr": status.stderr,
                }
            )
        (args.output / f"{label}-environment.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
        worker = [sys.executable, f"experiments/{script}", *options]
        if baseline and args.baseline_source:
            worker = [
                sys.executable,
                "experiments/baseline_runtime.py",
                "--source",
                str(args.baseline_source),
                "--",
                script,
                *options,
            ]
        subprocess.run(
            [
                sys.executable,
                "experiments/run_bounded.py",
                "--output",
                str(args.output / f"{label}-command"),
                "--timeout",
                "900",
                "--",
                *worker,
            ],
            check=True,
        )

    shared = ["--manifest", str(args.manifest), "--max-new-tokens", str(args.max_new_tokens)]
    try:
        if args.mode in ("memory", "both"):
            for pair in range(5):
                order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                for name in order:
                    label = f"memory-{pair}-{name}"
                    destination = args.output / f"{label}.json"
                    run(
                        label,
                        "measure_round3_memory.py",
                        [
                            "--bundle",
                            str(paths[name]),
                            "--output",
                            str(destination),
                            "--passes",
                            str(max(2, math.ceil(100 / utterances))) if pair == 0 else "2",
                            *shared,
                        ],
                        baseline=name == "baseline",
                    )
                    data = json.loads(destination.read_text())
                    report["memory"].append(
                        {
                            "pair": pair,
                            "mode": name,
                            "report": destination.name,
                            "sha256": digest(destination),
                            "complete": data["complete"],
                        }
                    )
        if args.mode in ("energy", "both"):
            rng = np.random.default_rng(report["seed"])
            for group in range(5):
                order = ("baseline", "candidate", "candidate", "baseline")
                if rng.integers(2):
                    order = ("candidate", "baseline", "baseline", "candidate")
                for slot, name in enumerate(order):
                    label = f"energy-{group}-{slot}-{name}"
                    destination = args.output / label
                    run(
                        label,
                        "benchmark_energy.py",
                        [
                            "--backend",
                            "coreml",
                            "--model-dir",
                            str(paths[name]),
                            "--output",
                            str(destination),
                            "--repeats",
                            str(args.energy_repeats),
                            *shared,
                        ],
                        baseline=name == "baseline",
                    )
                    data = json.loads((destination / "summary.json").read_text())
                    duration = data["power"]["duration_s"]
                    report["energy"].append(
                        {
                            "group": group,
                            "slot": slot,
                            "mode": name,
                            "report": str(destination.relative_to(args.output)),
                            "duration_seconds": duration,
                            "joules_per_audio_second": data["gross_j_per_audio_second"],
                            "boundary_shift_j": data["boundary_shift_j"],
                        }
                    )
                    if duration < 60:
                        raise ValueError(
                            "Energy block shorter than 60 seconds; retain and rerun with more fixed work"
                        )
                    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            report["energy_medians"] = {
                name: float(
                    np.median(
                        [
                            row["joules_per_audio_second"]
                            for row in report["energy"]
                            if row["mode"] == name
                        ]
                    )
                )
                for name in paths
            }
            changes = []
            for group in range(5):
                values = {
                    name: float(
                        np.mean(
                            [
                                row["joules_per_audio_second"]
                                for row in report["energy"]
                                if row["group"] == group and row["mode"] == name
                            ]
                        )
                    )
                    for name in paths
                }
                changes.append(values["candidate"] / values["baseline"] - 1)
            bootstrap = np.random.default_rng(20260916).choice(
                changes, size=(10000, 5), replace=True
            )
            report["paired_energy_relative_changes"] = changes
            report["paired_energy_median_relative_change"] = float(np.median(changes))
            report["paired_energy_median_ci95"] = np.percentile(
                np.median(bootstrap, axis=1), [2.5, 97.5]
            ).tolist()
            report["energy_interval_unit"] = (
                "five complete ABBA/BAAB groups; exploratory uncalibrated estimate"
            )
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
