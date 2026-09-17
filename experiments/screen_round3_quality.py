"""Sequential quality screening against frozen target controls, with fail-closed gates."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate import character_language

SETS = {
    "selection": (
        "artifacts/evaluation/silu-validation/selection-200.jsonl",
        "artifacts/evaluation/round3/baseline/selection.jsonl",
        256,
    ),
    "regression": (
        "artifacts/evaluation/round2/regression-400.jsonl",
        "artifacts/evaluation/round3/quality-controls/regression.jsonl",
        256,
    ),
    "multilingual": (
        "artifacts/evaluation/round2/multilingual/regression.jsonl",
        "artifacts/evaluation/round3/quality-controls/multilingual.jsonl",
        128,
    ),
    "robustness": (
        "artifacts/evaluation/round2/robustness/manifest.jsonl",
        "artifacts/evaluation/round3/quality-controls/robustness.jsonl",
        128,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--int8-embedding", type=Path)
    parser.add_argument("--sets", nargs="+", choices=tuple(SETS), default=list(SETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "score_gate_passed": True,
        "sets": {},
        "raw_text_review_required": True,
        "latency_claim": False,
    }
    prefix = [sys.executable, "experiments/round3_runtime.py"]
    if args.int8_embedding:
        prefix += ["--int8-embedding", str(args.int8_embedding)]
    try:
        for name in args.sets:
            manifest, control, budget = SETS[name]
            result = args.output / f"{name}.jsonl"
            command = [
                *prefix,
                "--",
                "evaluate.py",
                "--backend",
                "coreml",
                "--model-dir",
                str(args.bundle),
                "--manifest",
                manifest,
                "--output",
                str(result),
                "--warmups",
                "0",
                "--repeats",
                "1",
                "--max-new-tokens",
                str(budget),
            ]
            subprocess.run(
                [
                    sys.executable,
                    "experiments/run_bounded.py",
                    "--output",
                    str(args.output / f"{name}-command"),
                    "--timeout",
                    "1800",
                    "--",
                    *command,
                ],
                check=True,
            )
            comparison = args.output / f"{name}-comparison.json"
            subprocess.run(
                [
                    sys.executable,
                    "experiments/evaluate.py",
                    "--compare",
                    control,
                    str(result),
                    "--output",
                    str(comparison),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    sys.executable,
                    "experiments/review_transcript_changes.py",
                    "--baseline",
                    control,
                    "--candidate",
                    str(result),
                    "--output",
                    str(args.output / f"{name}-text-review.json"),
                ],
                check=True,
            )
            data = json.loads(comparison.read_text())
            gate = bool(data["valid_comparison"])
            languages = {}
            for language, metrics in data["by_language"].items():
                metric = "cer" if character_language(language) else "wer"
                value = metrics[metric]
                languages[language] = value
                if value.get("delta") is None:
                    # Empty-reference cases are checked below using error counts.
                    continue
                interval = value.get("ci95")
                # paired_bootstrap returns no interval for a single utterance or
                # zero reference units; that cannot pass a confidence gate.
                gate &= value["delta"] <= 0 and interval is not None and interval[1] <= 0.01
            original = {
                row["id"]: row for row in map(json.loads, Path(control).read_text().splitlines())
            }
            for row in map(json.loads, result.read_text().splitlines()):
                metric = "cer" if character_language(row["language"]) else "wer"
                if row.get("error"):
                    gate = False
                elif row["scores"][metric]["reference_units"] == 0:
                    gate &= (
                        row["scores"][metric]["errors"]
                        <= original[row["id"]]["scores"][metric]["errors"]
                    )
            report["sets"][name] = {
                "score_gate_passed": gate,
                "by_language": languages,
                "token_parity": data.get("token_parity"),
                "eos_parity": data.get("eos_parity"),
            }
            report["score_gate_passed"] &= gate
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"set": name, "score_gate_passed": gate, "by_language": languages}),
                flush=True,
            )
            if not gate:
                report["stop_reason"] = f"{name}_quality_gate_failed"
                break
        report["complete"] = True
    except Exception as error:
        report["score_gate_passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
