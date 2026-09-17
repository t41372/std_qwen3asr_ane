#!/bin/bash
# Independent candidates first; no fresh held-out is opened by this workflow.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
base=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
smoke=artifacts/evaluation/smoke/manifest.jsonl
root=artifacts/evaluation/round3
run() {
  local label=$1 timeout=$2
  shift 2
  "$benchmark_python" experiments/run_bounded.py --output "$root/$label-command" \
    --timeout "$timeout" -- "$benchmark_python" "$@"
}
compare() {
  "$benchmark_python" experiments/evaluate.py --compare "$root/$1/baseline.jsonl" \
    "$root/$1/candidate.jsonl" --output "$root/$1/comparison.json" > "$root/$1/compare.log" 2>&1
}
run instrumentation 300 experiments/round3_runtime.py --profile \
  --candidate-only artifacts/qwen3-asr-1.7b-r3-instrumentation -- benchmark_paired_corpus.py \
  --baseline "$base" --candidate artifacts/qwen3-asr-1.7b-r3-instrumentation \
  --manifest "$smoke" --output "$root/instrumentation" --repeats 3
compare instrumentation
run host-smoke 300 experiments/benchmark_paired_corpus.py --baseline "$base" \
  --baseline-source "$root/baseline/source" --candidate artifacts/qwen3-asr-1.7b-r3-host-combined \
  --manifest "$smoke" --output "$root/host-smoke" --repeats 3
compare host-smoke
run voice-command/supplement 1800 experiments/benchmark_paired_corpus.py \
  --baseline artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled \
  --candidate artifacts/qwen3-asr-1.7b-r3-voice-command \
  --manifest "$root/voice-command/short-supplement/manifest.jsonl" \
  --output "$root/voice-command/supplement" --repeats 3 --max-new-tokens 64
compare voice-command/supplement
run voice-command/robustness 300 experiments/benchmark_paired_corpus.py \
  --baseline artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled \
  --candidate artifacts/qwen3-asr-1.7b-r3-voice-command \
  --manifest "$root/voice-command/robustness.jsonl" \
  --output "$root/voice-command/robustness" --repeats 3 --max-new-tokens 64
compare voice-command/robustness
"$benchmark_python" experiments/screen_round3_quality.py \
  --bundle artifacts/qwen3-asr-1.7b-r3-encoder-g16-v2 --output "$root/encoder-g16-quality" \
  --sets regression multilingual robustness
if ! "$benchmark_python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["score_gate_passed"] else 1)' \
    "$root/encoder-g16-quality/summary.json"; then
  "$benchmark_python" experiments/screen_round3_quality.py \
    --bundle artifacts/qwen3-asr-1.7b-r3-encoder-g8-v2 --output "$root/encoder-g8-quality" \
    --sets regression multilingual robustness
fi
