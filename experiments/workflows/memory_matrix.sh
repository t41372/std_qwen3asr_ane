#!/bin/bash
# System wired memory and process peak memory for the MLX 8-bit reference and
# the packaged draft path (the other rows were measured by memory.sh).
set -u
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=.venv/bin/python
DPY=.venv-draft/bin/python
MLXPY=experiments/mlx_reference/.venv/bin/python
OUT=artifacts/evaluation/candidates/memory
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
DRAFTB=artifacts/qwen3-asr-1.7b-draft
P14=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
mkdir -p $OUT
echo "=== sysmem mlx q8 $(date +%H:%M:%S) ==="
$MLXPY experiments/measure_system_memory.py --backend mlx --model-dir artifacts/source/Qwen3-ASR-1.7B-MLX-8bit --manifest $SMOKE --output $OUT/sysmem-mlx-q8.json 2>/dev/null | cut -c1-400
sleep 15
echo "=== sysmem p14 serial $(date +%H:%M:%S) ==="
$PY experiments/measure_system_memory.py --backend coreml --model-dir $P14 --manifest $SMOKE --output $OUT/sysmem-ane-p14.json 2>/dev/null | cut -c1-400
sleep 15
echo "=== sysmem packaged draft $(date +%H:%M:%S) ==="
$DPY experiments/measure_system_memory.py --backend coreml --model-dir $P14 --draft-dir $DRAFTB --manifest $SMOKE --output $OUT/sysmem-draft-pkg.json 2>/dev/null | cut -c1-400
sleep 15
measure() { local name=$1; shift; /usr/bin/time -l "$@" > $OUT/$name.stdout 2> $OUT/$name.time; echo "$name $(grep -E 'peak memory footprint' $OUT/$name.time | tr -s ' ')"; }
echo "=== process peak $(date +%H:%M:%S) ==="
measure mlx-q8 $MLXPY experiments/benchmark_mlx.py --weights-dtype q8 --model-dir artifacts/source/Qwen3-ASR-1.7B-MLX-8bit --manifest $SMOKE --output $OUT/mlx-q8.jsonl --warmups 0 --repeats 1
measure draft-pkg $DPY experiments/evaluate.py --backend coreml --model-dir $P14 --draft-dir $DRAFTB --manifest $SMOKE --output $OUT/draft-pkg.jsonl --warmups 0 --repeats 1
du -sh artifacts/source/Qwen3-ASR-1.7B-MLX-8bit $DRAFTB
echo ALL_MEMORY_MATRIX_DONE
