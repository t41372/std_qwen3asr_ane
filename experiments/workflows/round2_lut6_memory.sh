#!/bin/bash
# Independent process residency, alternating order, matched useful work.
set -euo pipefail
cd "$(dirname "$0")/../.."
output_root=${1:?Provide a fresh output directory}
if [[ -e "$output_root" ]]; then
  exit 1
fi
mkdir -p "$output_root"
benchmark_python=.venv/bin/python
for block in A1 B1 B2 A2 A3 B3; do
  case "$block" in
    A*) model=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled ;;
    B*) model=artifacts/qwen3-asr-1.7b-r2-lut6-g32-compiled ;;
  esac
  /usr/bin/time -l "$benchmark_python" experiments/measure_system_memory.py \
    --backend coreml --model-dir "$model" \
    --manifest artifacts/evaluation/round2/short-smoke.jsonl \
    --max-new-tokens 128 --output "$output_root/$block.json" \
    > "$output_root/$block.log" 2>&1
done
