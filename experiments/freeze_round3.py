"""Freeze local hardware, corpus identities and unused round-3 held-out rows.

This reads existing downloaded assets only. Held-out rows are selected from the
next pinned metadata-balanced prefix, excluding every previously evaluated ID
or audio hash. Candidate results must not influence this selection.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from evaluate import environment

from std_qwen3asr_ane.bundle import digest


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--offset", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = environment()
    report.update(python_executable=sys.executable, uv_lock_sha256=digest(Path("uv.lock")))
    report["commands"] = []
    for command in (
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain"],
        ["sw_vers"],
        ["uname", "-a"],
        ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "SPPowerDataType", "-json"],
        ["pmset", "-g", "custom"],
        ["pmset", "-g", "batt"],
        ["pmset", "-g", "therm"],
        ["xcodebuild", "-version"],
    ):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        report["commands"].append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    (args.output / "environment.json").write_text(json.dumps(report, indent=2) + "\n")

    used_ids, used_hashes = set(), set()
    # Evaluation rows, unlike input manifests, have prediction/outcome fields.
    evaluated_files = []
    for path in sorted(Path("artifacts/evaluation").rglob("*.jsonl")):
        if args.output in path.parents:
            continue
        evaluated = False
        for row in rows(path):
            if not isinstance(row, dict) or not any(
                key in row for key in ("hypothesis", "backend_timings", "token_ids")
            ):
                continue
            used_ids.add(row.get("id"))
            used_hashes.add(row.get("audio_sha256"))
            evaluated = True
        if evaluated:
            evaluated_files.append(str(path))
    # Calibration is excluded even if it did not produce evaluator-style rows.
    for path in (Path("artifacts/evaluation/round2/calibration-200.jsonl"),):
        for row in rows(path):
            used_ids.add(row["id"])
            used_hashes.add(row["audio_sha256"])
    selected, sources = [], {}
    for path in args.source_manifests:
        sources[str(path)] = digest(path)
        eligible = []
        for row in rows(path)[args.offset :]:
            if row["id"] in used_ids or row["audio_sha256"] in used_hashes:
                continue
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            if digest(audio_path) != row["audio_sha256"]:
                raise ValueError(f"Changed audio: {row['id']}")
            eligible.append({**row, "audio_path": str(audio_path)})
        if len(eligible) < 100:
            raise ValueError(f"Only {len(eligible)} unused rows in {path}; extend pinned corpus")
        selected.extend(eligible[:100])
    manifest = args.output / "heldout-200.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    report = {
        "selection": "Next 100 unused rows per pinned balanced source, excluding evaluated/calibration IDs and audio hashes",
        "source_offset": args.offset,
        "source_manifests": sources,
        "evaluated_files_checked": evaluated_files,
        "manifest_sha256": digest(manifest),
        "count": len(selected),
        "candidate_results_opened": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "freeze.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"heldout_rows": len(selected), "sha256": digest(manifest)}))


if __name__ == "__main__":
    main()
