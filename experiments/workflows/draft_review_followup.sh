#!/bin/bash
# Review follow-ups needing hardware, run serially: speculative-only trace, then the LUT8 compact-head probe.
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
PY=std_qwen3asr_ane/.venv/bin/python
DPY=std_qwen3asr_ane/.venv-draft/bin/python
P14=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
DRAFTB=artifacts/qwen3-asr-1.7b-draft   # from: qwen3-asr-ane build-draft
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
echo "=== speculative-only trace $(date +%H:%M:%S) ==="
$PY experiments/trace_ane.py record --prefix artifacts/telemetry/specdraft-only-q4-k15 --developer-dir /Applications/Xcode.app/Contents/Developer --seconds 90 -- $PWD/$DPY $PWD/experiments/benchmark_mlx_draft.py --target $PWD/$P14 --draft-bundle $PWD/$DRAFTB --draft-bits 4 --manifest $PWD/$SMOKE --output $PWD/artifacts/telemetry/specdraft-only-q4-k15-workload.jsonl --lookahead 15 --repeats 1 --skip-serial > artifacts/telemetry/specdraft-only-record.log 2>&1
echo "record exit $?"
grep -E '"target_pid"|"target_exit_status"|"ane_prediction_rows"|"gpu_hardware_rows_target_pid"|"ane_prediction_interval_union_ns"|"gpu_target_pid_interval_union_ns"|Traceback|Error' artifacts/telemetry/specdraft-only-record.log | head -8
sleep 15
echo "=== compact head probe on LUT8 (p14 bundle + LUT8 head) $(date +%H:%M:%S) ==="
$PY experiments/benchmark_compact_head.py --bundle $P14 --model $HEAD --output artifacts/probes/lm-head-compact-t16-lut8-g32-vs-p14-lut8.json > artifacts/probes/lm-head-compact-t16-lut8-g32-vs-p14-lut8.log 2>&1
echo "probe exit $?"
$PY -c "
import json; d=json.load(open('artifacts/probes/lm-head-compact-t16-lut8-g32-vs-p14-lut8.json')); print('states', d['states'], 'mismatches', len(d['mismatches']), 'chunk', d['vocabulary_chunk'], 'median', d['median_seconds'])" 2>&1 | tail -2
echo REVIEW_FOLLOWUP_DONE
