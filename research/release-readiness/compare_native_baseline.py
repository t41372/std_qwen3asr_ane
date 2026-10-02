"""Compare default greedy decoding from one source tree against another.

The script has no model download, conversion, or artifact-writing path. In
``compare`` mode it spawns two fresh Python processes so each imports exactly
one ``std_qwen3asr_ane.runtime`` implementation against the same compiled
bundle and decoded NumPy audio arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/qwen3-asr-1.7b"
BASELINE_SOURCE = ROOT / ".cache/native-baseline-884c22e/std_qwen3asr_ane/src"
CURRENT_SOURCE = ROOT / "std_qwen3asr_ane/src"
EVIDENCE = ROOT / "research/release-readiness/native-baseline-884c22e-2026-10-02.json"
HELDOUT_MANIFESTS = (
    ROOT / "artifacts/evaluation/librispeech-balanced-100/manifest.jsonl",
    ROOT / "artifacts/evaluation/fleurs-zh-balanced-100/manifest.jsonl",
)
SMOKE = (
    ("smoke-en", ROOT / "artifacts/evaluation/smoke/qwen_official_en.wav"),
    ("smoke-zh", ROOT / "artifacts/evaluation/smoke/qwen_official_zh.wav"),
)
HELDOUT_PER_MANIFEST = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs() -> list[tuple[str, Path, dict[str, object]]]:
    selected = []
    for item_id, path in SMOKE:
        selected.append((item_id, path, {"selection": "official smoke"}))
    for manifest_path in HELDOUT_MANIFESTS:
        for index, line in enumerate(manifest_path.read_text().splitlines()[:HELDOUT_PER_MANIFEST]):
            row = json.loads(line)
            path = manifest_path.parent / row["audio_path"]
            selected.append(
                (
                    f"{manifest_path.parent.name}-{index}-{row['id']}",
                    path,
                    {
                        "selection": f"first {HELDOUT_PER_MANIFEST} rows of {manifest_path.relative_to(ROOT)}",
                        "manifest_audio_sha256": row["audio_sha256"],
                        "manifest_language": row["language"],
                    },
                )
            )
    return selected


def load(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] != 1:
        raise ValueError(f"Expected mono audio: {path}")
    samples = data[:, 0]
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor).astype(
            np.float32
        )
    return samples, {
        "path": str(path),
        "sha256": sha256(path),
        "source_sample_rate": sample_rate,
        "native_samples": len(samples),
        "native_seconds": len(samples) / 16000,
    }


def run_single(label: str) -> None:
    from std_qwen3asr_ane.runtime import CoreMLRuntime

    runtime_module = importlib.import_module("std_qwen3asr_ane.runtime")
    runtime = CoreMLRuntime(BUNDLE)
    try:
        rows = []
        for item_id, path, selection in inputs():
            samples, audio = load(path)
            result = runtime.transcribe(samples, language=None, max_new_tokens=256)
            rows.append(
                {
                    "id": item_id,
                    "selection": selection,
                    "audio": audio,
                    "text": result.text,
                    "raw_text": result.raw_text,
                    "language": result.language,
                    "token_ids": list(result.token_ids),
                    "eos_token_id": result.timings["eos_token_id"],
                }
            )
        print(
            json.dumps(
                {
                    "label": label,
                    "runtime_module": runtime_module.__file__,
                    "runtime_sha256": sha256(Path(runtime_module.__file__)),
                    "bundle_manifest_sha256": sha256(BUNDLE / "manifest.json"),
                    "heldout_per_manifest": HELDOUT_PER_MANIFEST,
                    "rows": rows,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
    finally:
        runtime.close()


def run_child(label: str, source: Path | None) -> dict:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if source is not None:
        environment["PYTHONPATH"] = str(source)
    process = subprocess.run(
        [sys.executable, str(Path(__file__)), "single", "--label", label],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"{label} exited {process.returncode}: stderr={process.stderr} stdout={process.stdout}"
        )
    return json.loads(process.stdout)


def compare() -> dict:
    baseline = run_child("baseline-884c22e", BASELINE_SOURCE)
    current = run_child("current-installed-editable", None)
    baseline_rows = {row["id"]: row for row in baseline["rows"]}
    current_rows = {row["id"]: row for row in current["rows"]}
    comparison = []
    for item_id, older in baseline_rows.items():
        newer = current_rows[item_id]
        comparison.append(
            {
                "id": item_id,
                "text_equal": older["text"] == newer["text"],
                "raw_text_equal": older["raw_text"] == newer["raw_text"],
                "token_ids_equal": older["token_ids"] == newer["token_ids"],
                "eos_equal": older["eos_token_id"] == newer["eos_token_id"],
                "baseline": older,
                "current": newer,
            }
        )
    equality = {
        "all_text_equal": all(item["text_equal"] for item in comparison),
        "all_raw_text_equal": all(item["raw_text_equal"] for item in comparison),
        "all_token_ids_equal": all(item["token_ids_equal"] for item in comparison),
        "all_eos_equal": all(item["eos_equal"] for item in comparison),
    }
    return {
        "schema_version": 1,
        "date": "2026-10-02",
        "status": "passed" if all(equality.values()) else "failed",
        "command": "./.venv/bin/python research/release-readiness/compare_native_baseline.py compare",
        "environment": {
            "executable": sys.executable,
            "current_source": "installed editable package; PYTHONPATH removed",
            "baseline_source_override": str(BASELINE_SOURCE),
            "standard_asr_commit": "5f6eef25e35e5e66e9010474e6dee531021e61f1",
        },
        "source_sha256": {
            "verifier": sha256(Path(__file__)),
            "current_runtime": sha256(CURRENT_SOURCE / "std_qwen3asr_ane/runtime.py"),
        },
        "baseline": baseline,
        "current": current,
        "comparison": comparison,
        **equality,
        "failures": [] if all(equality.values()) else ["one or more exact parity gates failed"],
        "performance_claim": False,
    }


def write_evidence(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="mode", required=True)
    single = subcommands.add_parser("single")
    single.add_argument("--label", required=True)
    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("--output", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    if args.mode == "single":
        run_single(args.label)
    else:
        try:
            document = compare()
        except Exception as error:
            document = {
                "schema_version": 1,
                "date": "2026-10-02",
                "status": "failed",
                "command": "./.venv/bin/python research/release-readiness/compare_native_baseline.py compare",
                "source_sha256": {"verifier": sha256(Path(__file__))},
                "failures": [{"type": type(error).__name__, "message": str(error)}],
                "performance_claim": False,
            }
            write_evidence(document, args.output)
            raise
        write_evidence(document, args.output)
        print(json.dumps({"status": document["status"], "output": str(args.output)}))
        if document["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
