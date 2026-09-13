#!/bin/bash
# p14 evidence (placement, compliance, streaming, trace) and the speculative path's parity + energy.
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
PY=std_qwen3asr_ane/.venv/bin/python
DPY=std_qwen3asr_ane/.venv-draft/bin/python
OUT=artifacts/evaluation/candidates
SMOKE=artifacts/evaluation/smoke/manifest.jsonl
SEL=artifacts/evaluation/silu-validation/selection-200.jsonl
P14=artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled
DRAFTB=artifacts/qwen3-asr-1.7b-draft   # from: qwen3-asr-ane build-draft
echo "=== p14 placement $(date +%H:%M:%S) ==="
mkdir -p artifacts/validation/p14-lut8-g32-placement
for g in frontend encoder decoder_00 decoder_14 lm_head; do
  $PY -m std_qwen3asr_ane.cli inspect artifacts/qwen3-asr-1.7b-p14-lut8-g32/$g.mlpackage > artifacts/validation/p14-lut8-g32-placement/$g.json 2>/dev/null
  $PY -c "
import json; d=json.load(open('artifacts/validation/p14-lut8-g32-placement/$g.json')); s=d['summary']
print('$g', json.dumps({k: v['operation_count'] for k, v in s['by_preferred_device'].items()}), 'known_cost_non_ane', sum(v['known_cost_count'] for k, v in s['by_preferred_device'].items() if k != 'ane'))"
done
echo "=== p14 streaming + silence + compliance $(date +%H:%M:%S) ==="
$PY experiments/verify_streaming_runtime.py --model-dir $P14 --audio artifacts/evaluation/smoke/qwen_official_en.wav --output $OUT/streaming-p14-en.json --realtime --chunk-seconds 2 > $OUT/streaming-p14-en.log 2>&1
$PY -c "
import json; d=json.load(open('$OUT/streaming-p14-en.json')); print('events', len(d['events']), 'event_ok', d['event_compliance_passed'], 'result_ok', d['result_compliance_passed'], d['result']['text'][:80])"
$PY experiments/verify_streaming_runtime.py --model-dir $P14 --audio artifacts/evaluation/smoke/qwen_official_zh.wav --output $OUT/streaming-p14-silence.json --silence > $OUT/streaming-p14-silence.log 2>&1
$PY -c "
import json; d=json.load(open('$OUT/streaming-p14-silence.json')); print('silence', d['silence'], 'stream_matches_batch', d.get('stream_matches_batch'))"
STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=$P14 std_qwen3asr_ane/.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b > $OUT/compliance-p14.log 2>&1; echo "compliance exit $?"
echo "=== p14 trace $(date +%H:%M:%S) ==="
$PY experiments/trace_ane.py record --prefix artifacts/telemetry/p14-lut8-g32-isolated --developer-dir /Applications/Xcode.app/Contents/Developer --seconds 60 -- $PWD/$PY $PWD/experiments/evaluate.py --backend coreml --model-dir $PWD/$P14 --manifest $PWD/$SMOKE --output $PWD/artifacts/telemetry/p14-lut8-g32-isolated-workload.jsonl --warmups 0 --repeats 1 > artifacts/telemetry/p14-record.log 2>&1
$PY experiments/bind_trace_evidence.py --prefix artifacts/telemetry/p14-lut8-g32-isolated --bundle artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled --placement-dir artifacts/validation/p14-lut8-g32-placement --output artifacts/telemetry/p14-lut8-g32-attribution.json 2>&1 | grep -E "ane_active_share|candidate_prediction|unattributed|target_pid|exit" 
$PY experiments/bind_trace_evidence.py --prefix artifacts/telemetry/lut8-g32-isolated --bundle artifacts/qwen3-asr-1.7b-lut8-g32-compiled --placement-dir artifacts/validation/lut8-g32-placement --output artifacts/telemetry/lut8-g32-attribution-bound.json 2>&1 | grep -E "ane_active_share|candidate_prediction|unattributed|target_pid"
sleep 30
echo "=== speculative selection parity (q4, k15) $(date +%H:%M:%S) ==="
$DPY experiments/benchmark_mlx_draft.py --target $P14 --draft-bundle $DRAFTB --draft-bits 4 --manifest $SEL --output $OUT/specdraft-q4-k15-selection.jsonl --lookahead 15 --repeats 1 > $OUT/specdraft-q4-k15-selection.log 2>&1
$PY - <<PYEOF
import json, statistics
rows=[json.loads(l) for l in open("$OUT/specdraft-q4-k15-selection.jsonl")]
m=[r for r in rows if r["repeat"]==0]
print("rows", len(m), "parity_all", all(r["exact_token_parity"] for r in m), "serial_total=%.1f"%sum(r["serial_seconds"] for r in m), "spec_total=%.1f"%sum(r["speculative_seconds"] for r in m),
      "audio=%.1f"%sum(r["audio_seconds"] for r in m), "accepted=%d/%d"%(sum(r["speculative"]["accepted_tokens"] for r in m), sum(r["speculative"]["proposed_tokens"] for r in m)))
print("tail log:", open("$OUT/specdraft-q4-k15-selection.log").read()[-400:] if len(m) < 200 else "ok")
PYEOF
sleep 30
echo "=== energy ABBA: p14 serial vs p14+q4 draft k15 $(date +%H:%M:%S) ==="
mkdir -p artifacts/power-v6
run_block() { local name=$1 python=$2; shift 2; echo "--- $name $(date +%H:%M:%S) $(pmset -g batt | tail -1 | tr -s ' ' | cut -c1-60)"; $python experiments/benchmark_energy.py --manifest $SMOKE --repeats 40 --output artifacts/power-v6/$name "$@" > artifacts/power-v6/$name.log 2>&1; $PY -c "
import json; d=json.load(open('artifacts/power-v6/$name/summary.json')); p=d['power']
print(json.dumps({'name':'$name','duration_s':round(p['duration_s'],2),'mean_w':round(p['mean_w_estimate'],2),'j_per_audio_s':round(d.get('gross_j_per_audio_second',0),3),'cpu':d.get('process_cpu'),'error':d.get('error')}))"; sleep 20; }
run_block serial-1 $PY --backend coreml --model-dir $P14
run_block spec-1 $DPY --backend specdraft --model-dir $P14 --draft-dir $DRAFTB --draft-bits 4 --lookahead 15
run_block spec-2 $DPY --backend specdraft --model-dir $P14 --draft-dir $DRAFTB --draft-bits 4 --lookahead 15
run_block serial-2 $PY --backend coreml --model-dir $P14
echo "--- sysmem spec"
$DPY - <<PYEOF
import json, subprocess, time, sys
sys.path.insert(0, "experiments")
from measure_system_memory import vm_stat, delta
from evaluate import audio_samples, manifest_rows
from pathlib import Path
import coremltools as ct
from benchmark_speculative import DecoderCursor
from mlx_draft import MLXDraft
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel
from std_qwen3asr_ane.speculative import greedy_speculative_decode
inputs=[audio_samples(Path(r["audio_path"]))[0] for r in manifest_rows(Path("$SMOKE"))]
time.sleep(5); before=vm_stat()
runtime=CoreMLRuntime(Path("$P14")); head=PersistentInputModel(ct.models.MLModel("$HEAD", compute_units=ct.ComputeUnit.CPU_AND_NE)); draft=MLXDraft(Path("$DRAFT"), quantize_bits=4)
loaded=vm_stat()
for s in inputs:
    p=runtime.prepare_prompt(s, language=None, max_new_tokens=256); draft.prepare(s, list(p.token_ids))
    greedy_speculative_decode(DecoderCursor(runtime, p, head), draft, p.hidden, target_position=len(p.token_ids), draft_position=len(p.token_ids), eos_token_ids=frozenset(runtime.eos_token_ids), max_new_tokens=256, lookahead=15)
after=vm_stat()
print(json.dumps({"after_load_mib": delta(loaded, before), "after_inference_mib": delta(after, before)}))
head.close(); runtime.close()
PYEOF
echo ALL_P14_DRAFT_DONE
