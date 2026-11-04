#!/bin/bash
# Status at a glance. Run as often as you like:  bash monitor.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./config.env
OUT="$CJ_OUTPUT_DIR"
echo "=== queue ==="; squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.10l %R" 2>/dev/null
echo; echo "=== state ==="; ls "$OUT/state" 2>/dev/null | tr '\n' ' '; echo
[ -f "$OUT/state/STOP" ] && { echo "STOPPED: $(cat "$OUT/state/STOP")"; }
[ -f "$OUT/state/DONE" ] && echo "DONE at $(cat "$OUT/state/DONE")"
echo; echo "=== pipeline log (last 12) ==="; tail -n 12 "$OUT/logs/pipeline.log" 2>/dev/null
echo; echo "=== runs ==="
if [ -x venv/bin/python ]; then venv/bin/python check_complete.py; else python3 check_complete.py; fi
echo; echo "=== latest training lines ==="
for f in "$OUT"/logs/phase2_pilot.log "$OUT"/logs/phase3_grid_worker*.log; do
  [ -f "$f" ] && { echo "-- $(basename "$f")"; grep -E " ep +[0-9]+/" "$f" | tail -n 1; }
done
echo; echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(run this inside the job: srun --jobid=<id> --pty nvidia-smi)"
