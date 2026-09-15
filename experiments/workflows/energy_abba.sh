#!/bin/bash
# Whole-machine energy: ABBA-style order, separate processes, serial.
# Usage: energy_abba.sh <bundle-name> [repeats]; the same repeat count is used for every backend
# so each block transcribes the same audio seconds (equal work).
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=.venv/bin/python
MLXPY=experiments/mlx_reference/.venv/bin/python
OUT=${ENERGY_OUT:-artifacts/power}
mkdir -p $OUT
REPEATS=${2:-40}
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
LUT=$1
run_block() { # name backend python model_dir repeats [extra args]
  local name=$1 backend=$2 python=$3 dir=$4 repeats=$5; shift 5
  echo "=== $name $(date +%H:%M:%S) $(pmset -g batt | tail -1 | tr -s ' ' | cut -c1-70) ==="
  $python experiments/benchmark_energy.py --backend $backend --model-dir $dir --manifest $SMOKE --repeats $repeats --output $OUT/$name "$@" > $OUT/$name.log 2>&1
  $PY -c "
import json; d=json.load(open('$OUT/$name/summary.json')); p=d['power']
print(json.dumps({'name':'$name','completed':d.get('completed'),'audio_s':round(d.get('audio_seconds',0),2),'duration_s':round(p['duration_s'],2),'mean_w':round(p['mean_w_estimate'],2),'gross_j':round(p['gross_j_estimate'],1),'j_per_audio_s':round(d.get('gross_j_per_audio_second',0),3),'error':d.get('error')}))"
  sleep 20
}
for order in 1 2; do
  if [ $order = 1 ]; then seq="fp16 lut mlxbf16 mlxq4"; else seq="mlxq4 mlxbf16 lut fp16"; fi
  for b in $seq; do
    case $b in
      fp16) run_block ane-fp16-$order coreml $PY artifacts/qwen3-asr-1.7b-compiled $REPEATS;;
      lut) run_block ane-$LUT-$order coreml $PY artifacts/qwen3-asr-1.7b-$LUT-compiled $REPEATS;;
      mlxbf16) run_block mlx-bf16-$order mlx $MLXPY artifacts/source/Qwen3-ASR-1.7B $REPEATS;;
      mlxq4) run_block mlx-q4-$order mlx $MLXPY artifacts/source/Qwen3-ASR-1.7B-MLX-4bit $REPEATS;;
    esac
  done
done
echo ALL_ENERGY_DONE
