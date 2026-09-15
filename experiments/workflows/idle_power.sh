#!/bin/bash
cd "$(dirname "$0")/../.."
while ! grep -q ALL_CLOSING_DONE artifacts/probes/floor/closing_chain.log; do sleep 30; done
sleep 30
.venv/bin/python experiments/power_v2/whole_machine.py --seconds 30 --battery-every 5 > artifacts/power-v6/idle-30s.jsonl 2>/dev/null
python3 -c "
import json; rows=[json.loads(l) for l in open('artifacts/power-v6/idle-30s.jsonl')]; vals=[r['pstr_w_estimate'] for r in rows if 'pstr_w_estimate' in r]; print('idle samples', len(vals), 'mean_w', round(sum(vals)/len(vals),2))"
echo IDLE_DONE
