#!/bin/bash
# Serialize audio graph construction and real-input paired screening.
set -uo pipefail
cd "$(dirname "$0")/../.."
benchmark_python=.venv/bin/python
baseline=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
root=${1:-artifacts/evaluation/round3/audio}
suffix=${2:-}
variants=(frontend-b4 frontend-b8 encoder-b2 encoder-g32 encoder-g16 encoder-g8)
if [ "$#" -gt 2 ]; then
  variants=("${@:3}")
fi
"$benchmark_python" experiments/run_bounded.py --output "$root/inventory-command" \
  --timeout 120 -- "$benchmark_python" experiments/inventory_audio_weights.py \
  --output "$root/inventory.json"
for variant in "${variants[@]}"; do
  case "$variant" in
    frontend-b4) options=(--role frontend --batch 4) ;;
    frontend-b8) options=(--role frontend --batch 8) ;;
    encoder-b2) options=(--role encoder --batch 2) ;;
    encoder-g32) options=(--role encoder --lut8-group 32) ;;
    encoder-g16) options=(--role encoder --lut8-group 16) ;;
    encoder-g8) options=(--role encoder --lut8-group 8) ;;
  esac
  candidate=artifacts/qwen3-asr-1.7b-r3-$variant$suffix
  if "$benchmark_python" experiments/run_bounded.py --output "$root/build-$variant" \
      --timeout 1800 -- "$benchmark_python" experiments/build_audio_candidate.py \
      --baseline "$baseline" --output "$candidate" "${options[@]}"; then
    "$benchmark_python" experiments/run_bounded.py --output "$root/run-$variant" \
      --timeout 600 -- "$benchmark_python" experiments/benchmark_audio_component.py \
      --baseline "$baseline" --candidate "$candidate" --output "$root/measure-$variant"
  fi
done
