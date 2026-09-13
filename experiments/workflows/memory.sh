#!/bin/bash
# Peak memory of "load bundle + transcribe both smoke utterances once" per backend.
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=std_qwen3asr_ane/.venv/bin/python
MLXPY=experiments/mlx_reference/.venv/bin/python
OUT=artifacts/evaluation/candidates/memory
mkdir -p $OUT
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
LUT=$1
measure() { # name command...
  local name=$1; shift
  /usr/bin/time -l "$@" > $OUT/$name.stdout 2> $OUT/$name.time
  echo "$name $(grep -E 'maximum resident set size|peak memory footprint' $OUT/$name.time | tr -s ' ' | tr '\n' ' ')"
}
measure ane-fp16 $PY experiments/evaluate.py --backend coreml --model-dir artifacts/qwen3-asr-1.7b-compiled --manifest $SMOKE --output $OUT/ane-fp16.jsonl --warmups 0 --repeats 1
measure ane-$LUT $PY experiments/evaluate.py --backend coreml --model-dir artifacts/qwen3-asr-1.7b-$LUT-compiled --manifest $SMOKE --output $OUT/ane-$LUT.jsonl --warmups 0 --repeats 1
measure mlx-bf16 $MLXPY experiments/benchmark_mlx.py --weights-dtype bf16 --model-dir artifacts/source/Qwen3-ASR-1.7B --manifest $SMOKE --output $OUT/mlx-bf16.jsonl --warmups 0 --repeats 1
measure mlx-q4 $MLXPY experiments/benchmark_mlx.py --weights-dtype q4 --model-dir artifacts/source/Qwen3-ASR-1.7B-MLX-4bit --manifest $SMOKE --output $OUT/mlx-q4.jsonl --warmups 0 --repeats 1
measure official-mps-bf16 $PY experiments/evaluate.py --backend official --device mps --dtype bfloat16 --attn-implementation sdpa --model-dir artifacts/source/Qwen3-ASR-1.7B --manifest $SMOKE --output $OUT/official-mps.jsonl --warmups 0 --repeats 1
du -sh artifacts/qwen3-asr-1.7b-compiled artifacts/qwen3-asr-1.7b-$LUT-compiled artifacts/source/Qwen3-ASR-1.7B artifacts/source/Qwen3-ASR-1.7B-MLX-4bit
echo ALL_MEMORY_DONE
