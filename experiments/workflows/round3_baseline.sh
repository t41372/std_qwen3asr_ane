#!/bin/bash
# Baseline must finish before changing the inference implementation.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
bundle=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
output=artifacts/evaluation/round3/baseline
smoke=artifacts/evaluation/smoke/manifest.jsonl
mkdir -p "$output"
"$benchmark_python" experiments/evaluate.py --backend coreml --model-dir "$bundle" \
  --manifest "$smoke" --output "$output/smoke.jsonl" --warmups 1 --repeats 3 \
  > "$output/smoke.log" 2>&1
"$benchmark_python" experiments/evaluate.py --backend coreml --model-dir "$bundle" \
  --manifest artifacts/evaluation/silu-validation/selection-200.jsonl \
  --output "$output/selection.jsonl" --warmups 1 --repeats 1 \
  > "$output/selection.log" 2>&1
for language in en zh; do
  "$benchmark_python" experiments/verify_streaming_runtime.py --model-dir "$bundle" \
    --audio "artifacts/evaluation/smoke/qwen_official_$language.wav" \
    --output "$output/streaming-$language.json" --realtime --chunk-seconds 2 \
    > "$output/streaming-$language.log" 2>&1
done
STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR="$bundle" \
  .venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b \
  > "$output/compliance.txt" 2>&1
"$benchmark_python" experiments/inspect_bundle_placement.py --bundle "$bundle" \
  --output "$output/placement" > "$output/placement.log" 2>&1
"$benchmark_python" experiments/measure_system_memory.py --backend coreml \
  --model-dir "$bundle" --manifest "$smoke" --output "$output/memory.json" \
  > "$output/memory.log" 2>&1
"$benchmark_python" experiments/benchmark_energy.py --backend coreml \
  --model-dir "$bundle" --manifest "$smoke" --repeats 60 --output "$output/energy" \
  > "$output/energy.log" 2>&1
