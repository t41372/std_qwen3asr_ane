#!/bin/bash
# Frozen optional memory candidate; run alone under a caffeinate assertion.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
candidate=artifacts/qwen3-asr-1.7b-r2-lut6-g32-compiled
root=artifacts/evaluation/round2

paired() {
  local name=$1 manifest=$2 repeats=$3
  "$benchmark_python" experiments/benchmark_paired_corpus.py \
    --baseline "$baseline" --candidate "$candidate" --manifest "$manifest" \
    --max-new-tokens 256 --repeats "$repeats" --output "$root/$name" \
    > "$root/$name.log" 2>&1
  "$benchmark_python" experiments/evaluate.py \
    --compare "$root/$name/baseline.jsonl" "$root/$name/candidate.jsonl" \
    --output "$root/$name/comparison.json" > "$root/$name-comparison.log" 2>&1
}

paired lut6-paired-3 "$root/regression-400.jsonl" 3
paired lut6-heldout "$root/lut6-heldout-200.jsonl" 1
paired lut6-multilingual "$root/multilingual/regression.jsonl" 1
"$benchmark_python" experiments/inspect_bundle_placement.py \
  --bundle "$candidate" --output "$root/lut6-placement" \
  > "$root/lut6-placement.log" 2>&1
