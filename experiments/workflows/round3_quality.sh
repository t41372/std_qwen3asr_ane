#!/bin/bash
# Screening uses historical target text controls, not historical latency claims.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
"$benchmark_python" experiments/screen_round3_quality.py \
  --bundle artifacts/qwen3-asr-1.7b-r3-frontend-b4-v2 \
  --output artifacts/evaluation/round3/frontend-quality \
  --sets regression multilingual robustness
"$benchmark_python" experiments/screen_round3_quality.py \
  --bundle artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled \
  --int8-embedding artifacts/evaluation/round3/int8-embedding/table \
  --output artifacts/evaluation/round3/int8-quality \
  --sets regression multilingual robustness
"$benchmark_python" experiments/screen_round3_quality.py \
  --bundle artifacts/qwen3-asr-1.7b-r3-encoder-g32-v2 \
  --output artifacts/evaluation/round3/encoder-g32-quality
