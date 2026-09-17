#!/bin/bash
# A failure or insufficient gain stays a bounded four-layer result.
set -uo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
for mode in qkv mlp both grouped sdpa all; do
  "$benchmark_python" experiments/run_bounded.py \
    --output "artifacts/evaluation/round3/fusion/$mode-command" --timeout 900 -- \
    "$benchmark_python" experiments/probe_round3_fusion.py --mode "$mode" \
    --output "artifacts/evaluation/round3/fusion/$mode"
done
