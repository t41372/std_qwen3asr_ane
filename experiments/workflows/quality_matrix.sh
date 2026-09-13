#!/bin/bash
# Quality of every GPU reference on the selection and held-out sets, paired
# against the official CPU FP32 model, the FP16 ANE bundle and the 8-bit ANE
# bundle. The same manifests, normalizer, language mode and token budget as the
# ANE runs. Corpus timing from these runs is not reported.
set -u
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=std_qwen3asr_ane/.venv/bin/python
MLXPY=experiments/mlx_reference/.venv/bin/python
EV=experiments/evaluate.py
OUT=artifacts/evaluation/candidates
SV=artifacts/evaluation/silu-validation
mkdir -p $OUT
compare() { # set candidate-file label
  local set=$1 file=$2 label=$3
  case $set in
    selection) refs="official:$SV/diagnostic-baseline.jsonl fp16:$SV/diagnostic-precise.jsonl lut8:$OUT/selection-lut8-g32.jsonl";;
    heldout) refs="official:$SV/heldout-baseline.jsonl fp16:$SV/heldout-precise.jsonl lut8:$OUT/heldout-lut8-g32.jsonl";;
  esac
  for ref in $refs; do
    name=${ref%%:*}; rfile=${ref#*:}
    $PY $EV --compare $rfile $file --output $OUT/$set-$label-vs-$name.json --bootstrap-samples 2000 --seed 20260912 > /dev/null 2>&1
    $PY -c "
import json; d=json.load(open('$OUT/$set-$label-vs-$name.json')); print('$set $label vs $name', 'valid', d.get('valid_comparison'), 'problems', len(d.get('problems') or []))
for lang,v in d.get('by_language',{}).items():
    m='wer' if lang=='en' else 'cer'; q=v[m]; print('  ', lang, m, 'delta_pp=%.3f'%(q['delta']*100), 'ci95_pp=[%.3f, %.3f]'%(q['ci95'][0]*100,q['ci95'][1]*100))"
  done
}
for set in selection heldout; do
  MAN=$SV/$set-200.jsonl
  for spec in "bf16 artifacts/source/Qwen3-ASR-1.7B" "q8 artifacts/source/Qwen3-ASR-1.7B-MLX-8bit" "q4 artifacts/source/Qwen3-ASR-1.7B-MLX-4bit"; do
    set -- $spec; dt=$1; dir=$2
    echo "=== mlx $dt $set $(date +%H:%M:%S) ==="
    $MLXPY experiments/benchmark_mlx.py --weights-dtype $dt --model-dir $dir --manifest $MAN --output $OUT/$set-mlx-$dt.jsonl --warmups 0 --repeats 1 > $OUT/$set-mlx-$dt.log 2>&1
    $PY -c "
import json; d=json.load(open('$OUT/$set-mlx-$dt.jsonl.summary.json')); print('quality', json.dumps(d.get('quality'))[:400]); print('load_error', d.get('model_load_error'))"
    compare $set $OUT/$set-mlx-$dt.jsonl mlx-$dt
  done
  echo "=== official mps bf16 $set $(date +%H:%M:%S) ==="
  $PY $EV --backend official --device mps --dtype bfloat16 --attn-implementation sdpa --model-dir artifacts/source/Qwen3-ASR-1.7B --manifest $MAN --output $OUT/$set-official-mps-bf16.jsonl --warmups 0 --repeats 1 > $OUT/$set-official-mps-bf16.log 2>&1
  $PY -c "
import json; d=json.load(open('$OUT/$set-official-mps-bf16.jsonl.summary.json')); print('quality', json.dumps(d.get('quality'))[:400]); print('load_error', d.get('model_load_error'))"
  compare $set $OUT/$set-official-mps-bf16.jsonl official-mps-bf16
done
echo ALL_QUALITY_MATRIX_DONE
