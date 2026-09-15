#!/bin/bash
# One-session, equal-work whole-machine energy for every path in the results
# table, alternating order (A B C ... then reversed), separate processes, with
# a 30-second idle sample first. Usage: energy_matrix.sh [repeats]
# Backends: fp16 lut8 p14 mlxbf16 mlxq8 mlxq4 draft (set BACKENDS to override).
set -u
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=.venv/bin/python
DPY=.venv-draft/bin/python
MLXPY=experiments/mlx_reference/.venv/bin/python
OUT=${ENERGY_OUT:-artifacts/power-matrix}
REPEATS=${1:-40}
BACKENDS=${BACKENDS:-"fp16 lut8 p14 mlxbf16 mlxq8 mlxq4 draft"}
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
DRAFTB=artifacts/qwen3-asr-1.7b-draft
mkdir -p $OUT
echo "=== idle 30 s $(date +%H:%M:%S) ==="
sleep 20
$PY experiments/power_v2/whole_machine.py --seconds 30 --battery-every 5 > $OUT/idle-30s.jsonl 2> $OUT/idle.log
$PY -c "
import json; rows=[json.loads(l) for l in open('$OUT/idle-30s.jsonl') if l.strip()]
w=[r['pstr_w_estimate'] for r in rows if r.get('pstr_w_estimate') is not None]; print('idle mean W', round(sum(w)/len(w),2), 'samples', len(w))" 2>/dev/null || tail -2 $OUT/idle-30s.jsonl
run_block() { # name backend python model_dir [extra args]
  local name=$1 backend=$2 python=$3 dir=$4; shift 4
  echo "=== $name $(date +%H:%M:%S) $(pmset -g batt | tail -1 | tr -s ' ' | cut -c1-70) ==="
  $python experiments/benchmark_energy.py --backend $backend --model-dir $dir --manifest $SMOKE --repeats $REPEATS --output $OUT/$name "$@" > $OUT/$name.log 2>&1
  $PY -c "
import json; d=json.load(open('$OUT/$name/summary.json')); p=d['power']
print(json.dumps({'name':'$name','completed':d.get('completed'),'audio_s':round(d.get('audio_seconds',0),2),'duration_s':round(p['duration_s'],2),'mean_w':round(p['mean_w_estimate'],2),'j_per_audio_s':round(d.get('gross_j_per_audio_second',0),3),'error':d.get('error')}))"
  sleep 20
}
reverse() { local r=""; for b in $1; do r="$b $r"; done; echo $r; }
for order in 1 2; do
  if [ $order = 1 ]; then seq="$BACKENDS"; else seq=$(reverse "$BACKENDS"); fi
  for b in $seq; do
    case $b in
      fp16) run_block ane-fp16-$order coreml $PY artifacts/qwen3-asr-1.7b-compiled;;
      lut8) run_block ane-lut8-$order coreml $PY artifacts/qwen3-asr-1.7b-lut8-g32-compiled;;
      p14) run_block ane-p14-$order coreml $PY artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled;;
      mlxbf16) run_block mlx-bf16-$order mlx $MLXPY artifacts/source/Qwen3-ASR-1.7B;;
      mlxq8) run_block mlx-q8-$order mlx $MLXPY artifacts/source/Qwen3-ASR-1.7B-MLX-8bit;;
      mlxq4) run_block mlx-q4-$order mlx $MLXPY artifacts/source/Qwen3-ASR-1.7B-MLX-4bit;;
      draft) run_block draft-$order specdraft $DPY artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled --draft-dir $DRAFTB --draft-bits 4 --lookahead 15;;
    esac
  done
done
echo ALL_ENERGY_MATRIX_DONE
