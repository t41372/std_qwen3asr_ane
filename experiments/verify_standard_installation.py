"""Exercise an installed CLI from unrelated working directories, fully offline.

Run after a successful ordinary tool install and standard-asr pull. This checks
installation/lifecycle behavior and records real results; it is not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="std-qwen3asr-ane/1.7b")
    args = parser.parse_args()
    cli, output = args.cli.absolute(), args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        STANDARD_ASR_MODEL_DIR=str(args.model_root.absolute()),
        STANDARD_ASR_ALLOW_DOWNLOAD="0",
    )

    def run(caller: Path, *arguments: str) -> dict:
        result = subprocess.run(
            [str(cli), *arguments, "--json"],
            cwd=caller,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        (caller / f"{arguments[0]}.log").write_text(result.stderr)
        return json.loads(result.stdout)

    records = []
    for index, audio in enumerate(args.audio):
        caller = output / f"caller-{index}"
        caller.mkdir(exist_ok=True)
        status = run(caller, "status", args.model, "--require-ready")
        pull = run(caller, "pull", args.model)
        assert status == pull, "Offline pull changed a ready immutable artifact report"
        transcript = run(caller, "transcribe", args.model, str(audio.absolute()))
        assert transcript["text"], "Expected nonempty recognition for this speech fixture"
        record = {
            "audio": str(audio.absolute()),
            "cwd": str(caller),
            "status": status,
            "pull": pull,
            "result": transcript,
        }
        records.append(record)
        (output / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    locations = {record["status"]["requirements"][0]["location"] for record in records}
    assert len(locations) == 1, "Changing cwd changed the resolved model"
    print(json.dumps({"passed": True, "recordings": len(records), "locations": sorted(locations)}))


if __name__ == "__main__":
    main()
