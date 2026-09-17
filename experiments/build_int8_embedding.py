"""Quantize only host embedding rows, retaining immutable reconstruction evidence."""

import argparse
import json
from pathlib import Path

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.embedding import write_int8_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    report = write_int8_embedding(args.bundle / manifest["files"]["embedding"], args.output)
    report.update(
        parent_manifest_sha256=digest(args.bundle / "manifest.json"),
        builder_sha256=digest(Path(__file__)),
        validation_status="unvalidated",
    )
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
