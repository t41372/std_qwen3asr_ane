#!/bin/bash
# All final paired regression work finishes before fresh held-out is opened.
set -euo pipefail
cd "$(dirname "$0")/../.."
candidate=${1:?Provide the frozen combined candidate bundle}
output=${2:?Provide a new output root}
benchmark_python=.venv/bin/python
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
for cohort in regression multilingual robustness short-supplement; do
  repeats=3
  case "$cohort" in
    regression) manifest=artifacts/evaluation/round2/regression-400.jsonl; budget=256 ;;
    multilingual) manifest=artifacts/evaluation/round2/multilingual/regression.jsonl; budget=128 ;;
    robustness) manifest=artifacts/evaluation/round2/robustness/manifest.jsonl; budget=128 ;;
    short-supplement)
      manifest=artifacts/evaluation/round3/voice-command/short-supplement/manifest.jsonl
      budget=128
      repeats=1 # Additional quality coverage; no latency promotion claim from this cohort.
      ;;
  esac
  "$benchmark_python" experiments/run_bounded.py --output "$output/$cohort-command" \
    --timeout 5400 -- "$benchmark_python" experiments/benchmark_paired_corpus.py \
    --baseline "$baseline" --baseline-source artifacts/evaluation/round3/baseline/source \
    --candidate "$candidate" --manifest "$manifest" --output "$output/$cohort" \
    --repeats "$repeats" --max-new-tokens "$budget"
  "$benchmark_python" experiments/evaluate.py --compare "$output/$cohort/baseline.jsonl" \
    "$output/$cohort/candidate.jsonl" --output "$output/$cohort/comparison.json" \
    > "$output/$cohort/compare.log" 2>&1
  "$benchmark_python" experiments/review_transcript_changes.py --baseline "$output/$cohort/baseline.jsonl" \
    --candidate "$output/$cohort/candidate.jsonl" --output "$output/$cohort/text-review.json"
done
