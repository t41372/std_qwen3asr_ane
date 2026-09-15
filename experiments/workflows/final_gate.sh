#!/bin/bash
# Final-candidate validation chain. Usage: final_gate.sh <bundle-name>
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=.venv/bin/python
EV=experiments/evaluate.py
OUT=artifacts/evaluation/candidates
LUT=$1
DIR=artifacts/qwen3-asr-1.7b-$LUT-compiled
HELD=artifacts/evaluation/silu-validation/heldout-200.jsonl
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
echo "=== held-out quality $LUT $(date +%H:%M:%S) ==="
$PY $EV --backend coreml --model-dir $DIR --manifest $HELD --output $OUT/heldout-$LUT.jsonl --warmups 0 --repeats 1 > $OUT/heldout-$LUT.log 2>&1
$PY -c "
import json; d=json.load(open('$OUT/heldout-$LUT.jsonl.summary.json')); print('quality', json.dumps(d.get('quality'))[:900]); print('latency', json.dumps(d.get('latency')))"
$PY $EV --compare artifacts/evaluation/silu-validation/heldout-precise.jsonl $OUT/heldout-$LUT.jsonl --output $OUT/heldout-$LUT-vs-fp16.json --bootstrap-samples 2000 --seed 20260912 > /dev/null 2>&1
$PY $EV --compare artifacts/evaluation/silu-validation/heldout-baseline.jsonl $OUT/heldout-$LUT.jsonl --output $OUT/heldout-$LUT-vs-official.json --bootstrap-samples 2000 --seed 20260912 > /dev/null 2>&1
$PY -c "
import json
for ref in ('fp16','official'):
    d=json.load(open('$OUT/heldout-$LUT-vs-'+ref+'.json'))
    print(ref, 'valid', d.get('valid_comparison'), 'problems', d.get('problems'))
    for lang,v in d.get('by_language',{}).items():
        m='wer' if lang=='en' else 'cer'
        q=v[m]; print(' ', lang, m, 'delta_pp=%.3f'%(q['delta']*100), 'ci95_pp=[%.3f, %.3f]'%(q['ci95'][0]*100,q['ci95'][1]*100))"
sleep 30
echo "=== energy ABBA $(date +%H:%M:%S) ==="
experiments/workflows/energy_abba.sh $LUT
sleep 20
echo "=== memory $(date +%H:%M:%S) ==="
experiments/workflows/memory.sh $LUT
sleep 20
echo "=== ANE trace $(date +%H:%M:%S) ==="
$PY experiments/trace_ane.py record --prefix artifacts/telemetry/$LUT-isolated --developer-dir /Applications/Xcode.app/Contents/Developer --seconds 60 -- $PWD/$PY $PWD/experiments/evaluate.py --backend coreml --model-dir $PWD/$DIR --manifest $PWD/$SMOKE --output $PWD/artifacts/telemetry/$LUT-isolated-workload.jsonl --warmups 0 --repeats 1 > artifacts/telemetry/$LUT-record.log 2>&1
grep -E '"ane_prediction_count"|"target_pid"|"trace_seconds"|Error|error' artifacts/telemetry/$LUT-record.log | head -6
echo "=== streaming realtime EN $(date +%H:%M:%S) ==="
$PY experiments/verify_streaming_runtime.py --model-dir $DIR --audio artifacts/evaluation/smoke/qwen_official_en.wav --output $OUT/streaming-$LUT-en.json --realtime --chunk-seconds 2 > $OUT/streaming-$LUT-en.log 2>&1
$PY -c "
import json; d=json.load(open('$OUT/streaming-$LUT-en.json')); print('events', len(d['events']), 'event_ok', d['event_compliance_passed'], 'result_ok', d['result_compliance_passed'], 'text', d['result']['text'][:120])"
echo "=== streaming silence $(date +%H:%M:%S) ==="
$PY experiments/verify_streaming_runtime.py --model-dir $DIR --audio artifacts/evaluation/smoke/qwen_official_zh.wav --output $OUT/streaming-$LUT-silence.json --silence > $OUT/streaming-$LUT-silence.log 2>&1
tail -2 $OUT/streaming-$LUT-silence.log | cut -c1-300
echo "=== plugin verify + compliance $(date +%H:%M:%S) ==="
$PY experiments/verify_plugin_runtime.py --model-dir $DIR --audio artifacts/evaluation/smoke/qwen_official_zh.wav --output $OUT/plugin-$LUT.json > /dev/null 2>&1; echo "plugin verify exit $?"
STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=$DIR .venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b > $OUT/compliance-$LUT.log 2>&1; echo "compliance exit $?"; tail -3 $OUT/compliance-$LUT.log | cut -c1-300
echo ALL_FINAL_DONE
