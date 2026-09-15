#!/bin/bash
# This takes about 72 minutes per condition. Keep other model work stopped.
# Run under one caffeinate -di assertion for constant system/display policy.
set -euo pipefail
cd "$(dirname "$0")/../.."
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
manifest=artifacts/evaluation/smoke/manifest.jsonl
root=artifacts/evaluation/round2

observe() {
  local mode=$1 suffix=$2
  shift 2
  local interpreter=.venv/bin/python
  if [[ "$mode" == draft ]]; then
    interpreter=.venv-draft/bin/python
  fi
  local idle_command=("$interpreter" experiments/benchmark_loaded_idle.py
    --model-dir "$baseline" --manifest "$manifest"
    --output "$root/idle-$mode$suffix")
  if [[ "$mode" == draft ]]; then
    idle_command+=(--draft-dir artifacts/qwen3-asr-1.7b-draft)
  fi
  "${idle_command[@]}" "$@" \
    > "$root/idle-$mode$suffix.log" 2>&1
}

for mode in target draft; do
  observe "$mode" -quick --idle-seconds 1 2 --bracket-seconds 1
done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for mode in ("target", "draft"):
    path = Path(f"artifacts/evaluation/round2/idle-{mode}-quick/summary.json")
    report = json.loads(path.read_text())
    checks = (
        report["complete"] and report["close_succeeded"],
        report["draft_loaded_by_prepare"] is False,
        report["draft_loaded_after_warmup"] is (mode == "draft"),
        len(report["idle"]) == 2 and all(row["power_samples"] > 0 for row in report["idle"]),
    )
    if not all(checks):
        raise RuntimeError(f"{mode} idle harness check failed: {report}")
PY

observe target ""
observe draft ""
