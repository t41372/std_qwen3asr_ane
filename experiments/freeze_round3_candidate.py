"""Freeze a committed candidate and evaluation inputs before opening held-out data."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evaluate import environment
from fingerprint_bundles import fingerprint

from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "std_qwen3asr_ane/src", "experiments"], text=True
    )
    if status:
        raise ValueError("Commit inference and experiment source changes before freezing")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "bundle": fingerprint(args.bundle),
        "environment": environment(),
        "uv_lock_sha256": digest(Path("uv.lock")),
        "promotion_gates_sha256": digest(Path("research/evidence/round3/promotion-gates.json")),
        "manifests": {str(path): digest(path) for path in args.manifests},
        "heldout_results_seen": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"commit": report["git_commit"], "manifest_sha256": report["bundle"]["manifest_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
