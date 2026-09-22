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
        samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor).astype(np.float32)
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


def run_child(label: str, source: Path) -> dict:
    environment = os.environ | {"PYTHONPATH": str(source)}
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


def compare() -> None:
    baseline = run_child("baseline-884c22e", BASELINE_SOURCE)
    current = run_child("current", CURRENT_SOURCE)
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
    print(
        json.dumps(
            {
                "command": "./.venv/bin/python research/release-readiness/compare_native_baseline.py compare",
                "baseline": baseline,
                "current": current,
                "comparison": comparison,
                "all_text_equal": all(item["text_equal"] for item in comparison),
                "all_raw_text_equal": all(item["raw_text_equal"] for item in comparison),
                "all_token_ids_equal": all(item["token_ids_equal"] for item in comparison),
                "all_eos_equal": all(item["eos_equal"] for item in comparison),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="mode", required=True)
    single = subcommands.add_parser("single")
    single.add_argument("--label", required=True)
    subcommands.add_parser("compare")
    args = parser.parse_args()
    if args.mode == "single":
        run_single(args.label)
    else:
        compare()


if __name__ == "__main__":
    main()
