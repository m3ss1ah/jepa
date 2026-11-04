#!/bin/bash
# ONE command for the whole project:   bash run_all.sh
# Submits (1) a CPU job that builds the environment + downloads the data, then (2) the GPU job that runs
# everything. Safe to run again at any time: finished work is never repeated.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./config.env
mkdir -p "$CJ_OUTPUT_DIR/logs" "$CJ_OUTPUT_DIR/state"

if [ -f "$CJ_OUTPUT_DIR/state/DONE" ]; then
  echo "Already DONE ($(cat "$CJ_OUTPUT_DIR/state/DONE")). Results: $CJ_OUTPUT_DIR/results_bundle.zip"
  echo "To run the final report again anyway:  rm $CJ_OUTPUT_DIR/state/DONE && bash run_all.sh"; exit 0
fi
# a previous STOP (gate failed / error) is cleared so that you can try again after fixing the cause
rm -f "$CJ_OUTPUT_DIR/state/STOP" "$CJ_OUTPUT_DIR/state/attempts" "$CJ_OUTPUT_DIR/state/last_progress"

if ! command -v sbatch >/dev/null 2>&1; then echo "sbatch not found - are you logged in to the cluster?"; exit 1; fi
if [ -f "$CJ_OUTPUT_DIR/state/prep.ok" ] && [ -f venv/.ready ]; then
  echo "Environment + data already prepared -> submitting the GPU job only."
  sbatch gpu_job.sbatch
else
  PREP=$(sbatch --parsable prep_job.sbatch) || exit 1
  echo "Submitted CPU prep job $PREP (environment + dataset download)."
  sbatch --dependency=afterok:$PREP --kill-on-invalid-dep=yes gpu_job.sbatch
fi
echo
echo "Watch progress:   bash monitor.sh        (or: squeue -u \$USER)"
