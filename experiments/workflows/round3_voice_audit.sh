#!/bin/bash
# Complete diagnostic provenance for the unpromoted cache256 candidate.
set -euo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
bundle=artifacts/qwen3-asr-1.7b-r3-voice-command
root=artifacts/evaluation/round3/voice-command
"$benchmark_python" experiments/run_bounded.py --output "$root/old-multilingual-command" \
  --timeout 300 -- "$benchmark_python" experiments/benchmark_paired_corpus.py \
  --baseline artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled --candidate "$bundle" \
  --manifest "$root/multilingual.jsonl" --output "$root/old-multilingual" --repeats 3 --max-new-tokens 64
"$benchmark_python" experiments/inspect_bundle_placement.py --bundle "$bundle" \
  --output "$root/placement" > "$root/placement.log" 2>&1
prefix="$root/hardware"
"$benchmark_python" experiments/trace_ane.py record --prefix "$prefix" \
  --developer-dir /Applications/Xcode.app/Contents/Developer --seconds 60 -- \
  "$PWD/$benchmark_python" "$PWD/experiments/evaluate.py" --backend coreml \
  --model-dir "$PWD/$bundle" --manifest "$PWD/$root/energy-manifest.jsonl" \
  --output "$PWD/$prefix-workload.jsonl" --warmups 0 --repeats 1 --max-new-tokens 64 \
  > "$root/hardware.log" 2>&1
"$benchmark_python" experiments/bind_trace_evidence.py --prefix "$prefix" --bundle "$bundle" \
  --placement-dir "$root/placement" --output "$root/hardware-attribution.json" \
  > "$root/hardware-attribution.log" 2>&1
