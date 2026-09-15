#!/bin/bash
# Smoke latency + selection-set quality for one candidate bundle.
set -u
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
PY=.venv/bin/python
EV=experiments/evaluate.py
OUT=artifacts/evaluation/candidates
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
SEL=artifacts/evaluation/silu-validation/selection-200.jsonl
NAME=${1:?bundle name, e.g. lut8-g32}
DIR=artifacts/qwen3-asr-1.7b-$NAME-compiled
mkdir -p $OUT
echo "=== smoke latency $NAME $(date +%H:%M:%S) ==="
$PY $EV --backend coreml --model-dir $DIR --manifest $SMOKE --output $OUT/smoke-$NAME.jsonl --warmups 1 --repeats 5 > $OUT/smoke-$NAME.log 2>&1
$PY - <<PYEOF
import json, statistics
rows=[json.loads(l) for l in open("$OUT/smoke-$NAME.jsonl")]
d=json.load(open("$OUT/smoke-$NAME.jsonl.summary.json")); print("load", d.get("model_load_seconds"), "err", d.get("model_load_error"))
for sid in sorted({r["id"] for r in rows}):
    m=[r for r in rows if r["id"]==sid and r["phase"]=="measured" and r["seconds"] is not None]
    if m: print(sid, "median=%.3f"%statistics.median(r["seconds"] for r in m), "hyps=%d"%len({r["hypothesis"] for r in m}))
PYEOF
echo "=== selection quality $NAME $(date +%H:%M:%S) ==="
$PY $EV --backend coreml --model-dir $DIR --manifest $SEL --output $OUT/selection-$NAME.jsonl --warmups 0 --repeats 1 > $OUT/selection-$NAME.log 2>&1
$PY -c "
import json; d=json.load(open('$OUT/selection-$NAME.jsonl.summary.json')); print('quality', json.dumps(d.get('quality'))[:700]); print('latency', json.dumps(d.get('latency')))"
for ref in fp16:artifacts/evaluation/silu-validation/diagnostic-precise.jsonl official:artifacts/evaluation/silu-validation/diagnostic-baseline.jsonl; do
  label=${ref%%:*}; file=${ref#*:}
  $PY $EV --compare $file $OUT/selection-$NAME.jsonl --output $OUT/selection-$NAME-vs-$label.json --bootstrap-samples 2000 --seed 20260912 > /dev/null 2>&1
  $PY -c "
import json; d=json.load(open('$OUT/selection-$NAME-vs-$label.json')); print('$label', 'valid', d.get('valid_comparison'), d.get('problems'))
for lang,v in d.get('by_language',{}).items():
    m='wer' if lang=='en' else 'cer'; q=v[m]; print(' ', lang, m, 'delta_pp=%.3f'%(q['delta']*100), 'ci95_pp=[%.3f, %.3f]'%(q['ci95'][0]*100,q['ci95'][1]*100))"
done
echo CANDIDATE_GATE_DONE
