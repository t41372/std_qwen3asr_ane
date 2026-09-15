#!/bin/bash
# Run under one `caffeinate -di` assertion on an otherwise quiet machine.
set -euo pipefail
cd "$(dirname "$0")/../.."
output_root=${1:?Provide a fresh output directory}
benchmark_python=.venv/bin/python
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
candidate=artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled
manifest=artifacts/evaluation/round2/short-smoke.jsonl
mkdir -p "$output_root"
for block in A1 B1 B2 A2; do
  case "$block" in
    A*) model=$baseline ;;
    B*) model=$candidate ;;
  esac
  "$benchmark_python" experiments/benchmark_energy.py \
    --backend coreml --model-dir "$model" --manifest "$manifest" \
    --max-new-tokens 128 --repeats 60 --output "$output_root/$block" \
    > "$output_root/$block.log" 2>&1
done
