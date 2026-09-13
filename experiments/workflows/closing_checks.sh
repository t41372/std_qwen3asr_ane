#!/bin/bash
# After the p14/draft chain: trace the speculative smoke run (ANE vs GPU intervals for one PID),
# rebuild p14 through the CLI path and hash-compare, then the fresh-clone install check.
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
PY=std_qwen3asr_ane/.venv/bin/python
DPY=std_qwen3asr_ane/.venv-draft/bin/python
P14=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
DRAFTB=artifacts/qwen3-asr-1.7b-draft   # from: qwen3-asr-ane build-draft
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
while ! grep -q ALL_P14_DRAFT_DONE artifacts/probes/floor/p14_draft_chain.log; do sleep 30; done
sleep 20
echo "=== speculative trace $(date +%H:%M:%S) ==="
$PY experiments/trace_ane.py record --prefix artifacts/telemetry/specdraft-q4-k15-isolated --developer-dir /Applications/Xcode.app/Contents/Developer --seconds 90 -- $PWD/$DPY $PWD/experiments/benchmark_mlx_draft.py --target $PWD/$P14 --draft-bundle $PWD/$DRAFTB --draft-bits 4 --manifest $PWD/$SMOKE --output $PWD/artifacts/telemetry/specdraft-q4-k15-isolated-workload.jsonl --lookahead 15 --repeats 1 > artifacts/telemetry/specdraft-record.log 2>&1
grep -E '"ane_prediction_rows"|"gpu_hardware_rows_target_pid"|"target_pid"|"trace_duration_seconds"|Error' artifacts/telemetry/specdraft-record.log | head -6
echo "=== rebuild p14 through the CLI path $(date +%H:%M:%S) ==="
$PY -m std_qwen3asr_ane.cli build --source artifacts/source/Qwen3-ASR-1.7B --output artifacts/qwen3-asr-1.7b-cli-p14 --cache-length 1024 --token-batch-size 16 --layers-per-partition 14 > artifacts/cli-p14-build.log 2>&1; echo "build exit $?"
$PY -m std_qwen3asr_ane.cli compress --source artifacts/qwen3-asr-1.7b-cli-p14 --output artifacts/qwen3-asr-1.7b-cli-p14-lut8-g32 --scheme palette --bits 8 --group-size 32 > artifacts/cli-p14-compress.log 2>&1; echo "compress exit $?"
for f in decoder_00 decoder_14 lm_head; do
  a=$(shasum -a 256 artifacts/qwen3-asr-1.7b-cli-p14-lut8-g32/$f.mlpackage/Data/com.apple.CoreML/weights/weight.bin | cut -c1-16)
  b=$(shasum -a 256 artifacts/qwen3-asr-1.7b-p14-lut8-g32/$f.mlpackage/Data/com.apple.CoreML/weights/weight.bin | cut -c1-16)
  echo "$f cli=$a variant=$b $([ "$a" = "$b" ] && echo SAME || echo DIFFERENT)"
done
for f in frontend encoder; do
  a=$(shasum -a 256 artifacts/qwen3-asr-1.7b-cli-p14/$f.mlpackage/Data/com.apple.CoreML/weights/weight.bin | cut -c1-16)
  b=$(shasum -a 256 artifacts/qwen3-asr-1.7b-final/$f.mlpackage/Data/com.apple.CoreML/weights/weight.bin | cut -c1-16)
  echo "$f cli=$a final=$b $([ "$a" = "$b" ] && echo SAME || echo DIFFERENT)"
done
echo "=== fresh clone install check $(date +%H:%M:%S) ==="
SCRATCH="${SCRATCH:-/tmp/std-qwen3asr-ane-fresh-install}"
rm -rf $SCRATCH && git clone -q $PWD $SCRATCH && cd $SCRATCH && export UV_CACHE_DIR="$PWD/.cache/uv" && uv sync --project std_qwen3asr_ane --frozen --group convert --python 3.12 2>&1 | tail -3; echo "uv sync exit ${PIPESTATUS[0]}"
std_qwen3asr_ane/.venv/bin/python -c "import std_qwen3asr_ane, standard_asr, coremltools; print('import ok', coremltools.__version__)"
std_qwen3asr_ane/.venv/bin/pytest -q std_qwen3asr_ane/tests 2>&1 | tail -1
std_qwen3asr_ane/.venv/bin/standard-asr list 2>&1 | head -5
std_qwen3asr_ane/.venv/bin/qwen3-asr-ane --help | head -3
echo ALL_CLOSING_DONE
