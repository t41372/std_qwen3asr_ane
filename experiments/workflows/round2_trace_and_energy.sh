#!/bin/bash
# Run after all corpus/conversion work, under one caffeinate -di assertion.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python="$PWD/.venv/bin/python"
baseline="$PWD/artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled"
short="$PWD/artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled"
lut6="$PWD/artifacts/qwen3-asr-1.7b-r2-lut6-g32-compiled"
root="$PWD/artifacts/evaluation/round2"

for name in short-smoke robustness; do
  case "$name" in
    short-smoke) manifest="$root/short-smoke.jsonl" ;;
    robustness) manifest="$root/robustness/manifest.jsonl" ;;
  esac
  "$benchmark_python" experiments/verify_cache_geometry.py \
    --baseline "$baseline" --candidate "$short" --manifest "$manifest" \
    --output "$root/$name-kv-parity.json" --max-new-tokens 128 \
    > "$root/$name-kv-parity.log" 2>&1
done

"$benchmark_python" experiments/verify_streaming_runtime.py \
  --model-dir "$short" --profile short-dictation \
  --audio artifacts/evaluation/fleurs-zh-balanced-200/audio/000587.wav \
  --output "$root/short-streaming-zh.json" --realtime --max-new-tokens 128 \
  > "$root/short-streaming-zh.log" 2>&1

# Quiet reverse-order repeat of the small native diagnostic.
for block in A1 B1 B2 A2; do
  case "$block" in
    A*)
      "$benchmark_python" experiments/benchmark_native_fixture.py \
        --model-dir "$baseline" --fixture "$root/native-fixture" \
        --output "$root/native-quiet-$block.json" \
        > "$root/native-quiet-$block.log" 2>&1 ;;
    B*)
      "$root/native-decoder-bench" "$baseline" "$root/native-fixture" \
        "$root/native-quiet-$block.json" > "$root/native-quiet-$block.log" 2>&1 ;;
  esac
done

for name in short lut6; do
  case "$name" in
    short) model=$short ;;
    lut6) model=$lut6 ;;
  esac
  "$benchmark_python" experiments/trace_ane.py record \
    --prefix "$root/$name-trace" \
    --developer-dir /Applications/Xcode.app/Contents/Developer --seconds 60 -- \
    "$benchmark_python" "$PWD/experiments/evaluate.py" --backend coreml \
    --model-dir "$model" --manifest "$root/short-smoke.jsonl" \
    --output "$root/$name-trace-workload.jsonl" \
    --max-new-tokens 128 --warmups 0 --repeats 1 \
    > "$root/$name-trace.log" 2>&1
  "$benchmark_python" experiments/bind_trace_evidence.py \
    --prefix "$root/$name-trace" --bundle "$model" \
    --placement-dir "$root/$name-placement" \
    --output "$root/$name-trace-attribution.json" \
    > "$root/$name-trace-attribution.log" 2>&1
done

/bin/bash experiments/workflows/round2_short_energy.sh "$root/energy-short"
