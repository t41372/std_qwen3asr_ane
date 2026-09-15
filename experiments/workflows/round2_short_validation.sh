#!/bin/bash
# Run serially, under one caffeinate assertion, with no concurrent model work.
# Candidate weights are frozen; the EOS pass only adds previously missing evidence.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
candidate=artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled
root=artifacts/evaluation/round2

paired() {
  local name=$1 manifest=$2
  "$benchmark_python" experiments/benchmark_paired_corpus.py \
    --baseline "$baseline" --candidate "$candidate" --manifest "$manifest" \
    --max-new-tokens 128 --repeats 1 --output "$root/$name" \
    > "$root/$name.log" 2>&1
  "$benchmark_python" experiments/evaluate.py \
    --compare "$root/$name/baseline.jsonl" "$root/$name/candidate.jsonl" \
    --output "$root/$name/comparison.json" > "$root/$name-comparison.log" 2>&1
}

paired short-eos-parity "$root/short-eos-diagnostic.jsonl"
paired multilingual-paired "$root/multilingual/regression.jsonl"
paired robustness-paired "$root/robustness/manifest.jsonl"
"$benchmark_python" experiments/benchmark_streaming_cache.py \
  --model-dir "$candidate" --manifest "$root/robustness/manifest.jsonl" \
  --output "$root/robustness-streaming.jsonl" --max-new-tokens 128 --repeats 1 \
  > "$root/robustness-streaming.log" 2>&1
