#!/bin/bash
# ONE command for Stage C:   bash run_stageC.sh
# Submits the CPU prep job (environment, read-only checks of the Stage B inputs, DINOv2 download) once, then the
# 2-GPU job. Safe to run again at any time: finished work is never repeated.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./stageC_common.sh

if [ -f "$STATE/DONE" ]; then
  echo "Stage C already DONE ($(cat "$STATE/DONE")). Read $OUT_ROOT/FINAL_REPORT.md"; exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then echo "sbatch not found - are you logged in to the cluster?"; exit 1; fi
if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "cjC\|cjC_prep"; then
  echo "A Stage C job is already queued or running:"; squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R"; exit 1
fi
# a previous STOP (gate failed / error / waiting for registration) is cleared so that you can continue after fixing it
rm -f "$STATE/STOP" "$STATE/attempts" "$STATE/last_progress"

if [ -f "$STATE/prep.ok" ] && [ -f venv/.ready ]; then
  JOB=$(bash ./submit_gpu.sh) || exit 1
  echo "Submitted Stage C GPU job $JOB (prep already done)."
else
  PREP=$(sbatch --parsable --partition="$PARTITION" stageC_prep.sbatch) || exit 1
  echo "Submitted CPU prep job $PREP."
  JOB=$(bash ./submit_gpu.sh --dependency=afterok:"$PREP" --kill-on-invalid-dep=yes) || exit 1
  echo "Submitted Stage C GPU job $JOB (starts after the prep job succeeds)."
fi
echo
echo "Watch progress:   bash monitor_stageC.sh        (or: squeue -u \$USER)"
