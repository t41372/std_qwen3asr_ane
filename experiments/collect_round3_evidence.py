"""Index retained raw evidence and copy compact, non-transcript measurement summaries."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/evaluation/round3"))
    parser.add_argument("--output", type=Path, default=Path("research/evidence/round3"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index, summaries = [], {}
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log", ".txt", ".patch"):
            continue
        relative = str(path.relative_to(args.root))
        index.append({"path": relative, "sha256": digest(path), "bytes": path.stat().st_size})
        if path.name != "summary.json":
            continue
        data = json.loads(path.read_text())
        summaries[relative] = {
            key: data[key]
            for key in (
                "complete",
                "close_succeeded",
                "error",
                "stop_reason",
                "score_gate_passed",
                "mode",
                "role",
                "batch",
                "rows",
                "stress_calls",
                "timings",
                "medians",
                "max_abs_error",
                "mean_abs_error",
                "hidden_max_abs_error",
                "non_ane_operations",
                "energy_medians",
                "bundle_manifest_sha256",
                "candidate_manifest_sha256",
            )
            if key in data
        }
        if "sets" in data:
            summaries[relative]["quality_sets"] = {
                name: {
                    "score_gate_passed": value["score_gate_passed"],
                    "by_language": value["by_language"],
                    "tokens_equal": (value.get("token_parity") or {}).get("all_equal"),
                }
                for name, value in data["sets"].items()
            }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "raw_root": str(args.root),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path): digest(path)
            for directory in (Path("std_qwen3asr_ane/src"), Path("experiments"))
            for path in sorted(directory.rglob("*"))
            if path.suffix in (".py", ".sh", ".c") and ".venv" not in path.parts
        },
        "files": index,
        "interpretation": "File presence is not a passed gate; consult completion flags and results-round3.md.",
    }
    (args.output / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "measurements.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"indexed_files": len(index), "summaries": len(summaries)}))


if __name__ == "__main__":
    main()
