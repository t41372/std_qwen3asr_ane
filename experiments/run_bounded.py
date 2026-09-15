"""Run one experiment with a process-group timeout and retained command/log evidence."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("A command and positive timeout are required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "command": command,
        "cwd": str(Path.cwd()),
        "timeout_seconds": args.timeout,
        "started_at": datetime.now(UTC).isoformat(),
        "complete": False,
    }
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report["source_sha256"] = {}
    for root in ("std_qwen3asr_ane/src", "experiments"):
        for directory, directories, files in os.walk(root):
            directories[:] = [
                name for name in directories if name not in (".venv", "__pycache__", ".cache")
            ]
            for name in files:
                path = Path(directory) / name
                if path.suffix in (".py", ".sh", ".c"):
                    report["source_sha256"][str(path)] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    started = time.monotonic()
    with (
        (args.output / "stdout.log").open("x") as stdout,
        (args.output / "stderr.log").open("x") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        report["pid"] = process.pid
        try:
            report["returncode"] = process.wait(timeout=args.timeout)
            report["complete"] = report["returncode"] == 0
        except subprocess.TimeoutExpired:
            report["error"] = "process_group_timeout"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            report["returncode"] = process.returncode
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            (args.output / "command.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "source_sha256"}),
        flush=True,
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
