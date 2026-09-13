#!/bin/bash
# The packaged draft path, end to end through the Standard ASR engine: a CLI
# transcription, then the selection and held-out sets with the draft on. The
# hypotheses must equal the serial default bundle's, sentence for sentence.
set -u
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
DPY=std_qwen3asr_ane/.venv-draft/bin/python
PY=std_qwen3asr_ane/.venv/bin/python
OUT=artifacts/evaluation/candidates
SV=artifacts/evaluation/silu-validation
DRAFTB=artifacts/qwen3-asr-1.7b-draft
echo "=== cli transcribe with draft $(date +%H:%M:%S) ==="
std_qwen3asr_ane/.venv-draft/bin/qwen3-asr-ane transcribe artifacts/evaluation/smoke/qwen_official_zh.wav --draft-dir $DRAFTB | head -5
for set in selection heldout; do
  echo "=== plugin draft $set $(date +%H:%M:%S) ==="
  $DPY experiments/evaluate.py --backend standard --model-key std-qwen3asr-ane/1.7b \
    --engine-config "{\"model_dir\": \"artifacts/qwen3-asr-1.7b\", \"draft_dir\": \"$DRAFTB\"}" \
    --manifest $SV/$set-200.jsonl --output $OUT/$set-plugin-draft.jsonl --warmups 0 --repeats 1 > $OUT/$set-plugin-draft.log 2>&1
  $PY - <<PYEOF
import json
draft = {r["id"]: r for r in map(json.loads, open("$OUT/$set-plugin-draft.jsonl")) if r["phase"] == "measured"}
serial = {r["id"]: r for r in map(json.loads, open("$OUT/$set-p14-lut8-g32.jsonl")) if r["phase"] == "measured"}
same = sum(draft[i]["hypothesis"] == serial[i]["hypothesis"] for i in serial)
errors = sum(r["error"] is not None for r in draft.values())
print("$set", "sentences", len(serial), "identical_text", same, "errors", errors)
PYEOF
  $PY experiments/evaluate.py --compare $SV/$( [ $set = selection ] && echo diagnostic || echo heldout )-baseline.jsonl $OUT/$set-plugin-draft.jsonl --output $OUT/$set-plugin-draft-vs-official.json --bootstrap-samples 2000 --seed 20260912 > /dev/null 2>&1
  $PY -c "
import json; d=json.load(open('$OUT/$set-plugin-draft-vs-official.json')); print('$set draft vs official valid', d['valid_comparison'])
for lang,v in d['by_language'].items():
    m='wer' if lang=='en' else 'cer'; q=v[m]; print('  ', lang, m, 'delta_pp=%.3f'%(q['delta']*100), 'ci95_pp=[%.3f, %.3f]'%(q['ci95'][0]*100,q['ci95'][1]*100))"
done
echo ALL_DRAFT_PLUGIN_PARITY_DONE
