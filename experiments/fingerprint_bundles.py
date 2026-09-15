"""Hash actual inference payloads before timing, including arrays and tokenizers."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from std_qwen3asr_ane.bundle import digest, validate_bundle_paths


def fingerprint(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relatives = set(manifest["files"].values()) | set(manifest["decoder_partitions"])
    validate_bundle_paths(root, relatives)
    paths = {manifest_path}
    for relative in relatives:
        path = root / relative
        if path.is_dir():
            paths.update(child for child in path.rglob("*") if child.is_file())
        else:
            paths.add(path)
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Bundle payload escapes its root: {path}")
    return {
        "path": str(root),
        "manifest_sha256": digest(manifest_path),
        "files": {
            str(path.relative_to(root)): {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(paths)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    root = Path(__file__).resolve().parents[1]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            )
        ),
        "source_sha256": {
            str(path.relative_to(root)): digest(path)
            for directory in (root / "std_qwen3asr_ane/src", root / "experiments")
            for pattern in ("*.py", "*.swift", "*.sh")
            for path in sorted(directory.rglob(pattern))
            if ".venv" not in path.parts
        },
        "bundles": [fingerprint(path) for path in args.bundles],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "bundles": len(report["bundles"])}))


if __name__ == "__main__":
    main()
