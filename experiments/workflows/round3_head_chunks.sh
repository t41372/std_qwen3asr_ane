#!/bin/bash
# Each build and measurement is isolated and bounded; failed builds retain logs.
set -uo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
root=artifacts/evaluation/round3/head-chunks
for chunk in 2048 4096 6144; do
  candidate=artifacts/qwen3-asr-1.7b-r3-head-$chunk
  if "$benchmark_python" experiments/run_bounded.py --output "$root/build-$chunk" \
      --timeout 1800 -- "$benchmark_python" experiments/build_head_chunk.py \
      --baseline "$baseline" --output "$candidate" --chunk "$chunk"; then
    "$benchmark_python" experiments/run_bounded.py --output "$root/run-$chunk" \
      --timeout 600 -- "$benchmark_python" experiments/benchmark_head_chunks.py \
      --baseline "$baseline" --candidate "$candidate" --output "$root/measure-$chunk"
  fi
done
