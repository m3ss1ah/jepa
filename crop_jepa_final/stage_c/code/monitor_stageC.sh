#!/bin/bash
# Status at a glance (read only). Run as often as you like:  bash monitor_stageC.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./stageC_common.sh
echo "=== queue ==="; squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10l %R" 2>/dev/null
echo; echo "=== state ==="; ls "$STATE" 2>/dev/null | tr '\n' ' '; echo
[ -f "$STATE/STOP" ] && echo "STOPPED: $(cat "$STATE/STOP")"
[ -f "$STATE/DONE" ] && echo "DONE at $(cat "$STATE/DONE")"
[ -f "$CJ_LOCKED_FILE" ] && python3 -c "import json;d=json.load(open('$CJ_LOCKED_FILE'));print('Phase 0: protocol', d['protocol'], '| context cap', d['ctx_tokens_cap'], '| cut points', [round(x,4) for x in d['dispersion_cut_points']])"
echo; echo "=== pipeline log (last 12) ==="; tail -n 12 "$LOGS/pipeline.log" 2>/dev/null
echo; echo "=== runs ==="
if [ -x venv/bin/python ]; then venv/bin/python check_complete_stageC.py all; else python3 check_complete_stageC.py all; fi
echo; echo "=== latest training lines ==="
for f in "$LOGS"/step2_phase0.log "$LOGS"/phase_*_worker*.log; do
  [ -f "$f" ] && { echo "-- $(basename "$f")"; grep -E " ep +[0-9]+/" "$f" | tail -n 1; }
done
echo; echo "=== GPU ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  || echo "(GPUs are visible inside the job: srun --jobid=<id> --overlap --pty nvidia-smi)"
