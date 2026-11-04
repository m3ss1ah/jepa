#!/bin/bash
# Status at a glance (read only):  bash monitor_followup.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./followup_common.sh
echo "=== queue ==="; squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10l %R" 2>/dev/null
echo; echo "=== state ==="; ls "$STATE" 2>/dev/null | tr '\n' ' '; echo
[ -f "$STATE/STOP" ] && echo "STOPPED: $(cat "$STATE/STOP")"
[ -f "$STATE/DONE" ] && echo "DONE at $(cat "$STATE/DONE")"
[ -f "$CJ_FOLLOWUP_FREEZE" ] && echo "follow-up plan frozen: $(cut -d' ' -f1 "$CJ_FOLLOWUP_FREEZE.sha256" 2>/dev/null | cut -c1-12)"
echo; echo "=== runs finished ==="; for PH in $FU_PHASES; do echo "$PH: $(n_done "$PH")/20"; done
echo; echo "=== pipeline log (last 10) ==="; tail -n 10 "$LOGS/pipeline.log" 2>/dev/null
echo; echo "=== latest training lines ==="
for f in "$LOGS"/*_worker*.log; do [ -f "$f" ] && { echo "-- $(basename "$f")"; grep -E " ep +[0-9]+/" "$f" | tail -n 1; }; done
