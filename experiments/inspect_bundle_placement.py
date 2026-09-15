"""Save anticipated placement and exact package provenance for a complete bundle."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from std_qwen3asr_ane.diagnostics import inspect_compute_plan


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    packages = [
        manifest["files"]["frontend"],
        *(
            [manifest["files"]["frontend_batched"]]
            if "frontend_batched" in manifest["files"]
            else []
        ),
        manifest["files"]["encoder"],
        *manifest["decoder_partitions"],
        manifest["files"]["lm_head"],
    ]
    if len(packages) != len(set(packages)):
        raise ValueError("Bundle manifest contains duplicate model packages")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "evidence_kind": "anticipated_compute_plan",
        "actual_execution_verified": False,
        "bundle_name": args.bundle.name,
        "source_revision": manifest.get("source_revision"),
        "manifest_sha256": digest(manifest_path),
        "token_batch_size": manifest.get("token_batch_size", 1),
        "compute_units": "cpu_and_ne",
        "models": [],
    }
    for package_name in packages:
        package = args.bundle / package_name
        files = [
            {
                "path": str(path.relative_to(args.bundle)),
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(package.rglob("*"))
            if path.is_file()
        ]
        plan = inspect_compute_plan(package)
        name = package.stem
        destination = args.output / f"{name}.json"
        destination.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        unknown = Counter(
            row["operator"] for row in plan["operations"] if row["preferred_device"] is None
        )
        non_ane = Counter(
            row["operator"]
            for row in plan["operations"]
            if row["preferred_device"] not in (None, "ane")
        )
        report["models"].append(
            {
                "model": package_name,
                "plan_file": destination.name,
                "summary": plan["summary"],
                "unknown_device_operators": dict(unknown),
                "non_ane_preferred_operators": dict(non_ane),
                "files": files,
            }
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": package_name,
                    "summary": plan["summary"],
                    "unknown_operators": dict(unknown),
                    "non_ane_operators": dict(non_ane),
                }
            ),
            flush=True,
        )
    print(f"Saved all {len(report['models'])} model plans", flush=True)


if __name__ == "__main__":
    main()
