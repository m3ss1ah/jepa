#!/bin/bash
# ONE command for the follow-up:   bash run_followup.sh
# Read-only checks of the Stage B / Stage C inputs, then submits the 1-GPU follow-up job. Safe to run again:
# finished phases and runs are skipped, unfinished runs resume from their checkpoint.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./followup_common.sh

if [ -f "$STATE/DONE" ]; then echo "Follow-up already DONE ($(cat "$STATE/DONE")). Read $OUT_ROOT/FOLLOWUP_2x2_REPORT.md"; exit 0; fi
command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found - are you logged in to the cluster?"; exit 1; }
if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "cjF"; then
  echo "A follow-up job is already queued or running:"; squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R"; exit 1
fi
missing=0
for f in "$CJ_CACHE_DIR/images_192.npy" "$CJ_CACHE_DIR/images_192.csv" "$STAGEB_SPLITS" "$CJ_LOCKED_FILE" \
         "$CJ_PREREG_FILE" "$CJ_PREREG_FILE.sha256" "$CJ_PREREG_CONFIRMED" "$STAGEC_VENV/bin/activate" \
         "$STAGEC_OUT/core/stats/seed_level_primary.csv" "$STAGEC_OUT/bridge/stats/seed_level_primary.csv" \
         "$STAGEC_OUT/core/stats/moderation.csv" "$STAGEC_OUT/bridge/stats/moderation.csv"; do
  [ -f "$f" ] || { echo "MISSING (read-only input): $f"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "Not submitting."; exit 1; }
echo "Stage C pre-registration sha256 (must be c6774dac...): $(cut -d' ' -f1 "$CJ_PREREG_FILE.sha256")"
rm -f "$STATE/STOP"
JOB=$(sbatch --parsable --partition="$PARTITION" --gres=gpu:"$NUM_GPUS" --cpus-per-task="$CPUS_PER_TASK" --mem="${MEM_GB}G" \
      --time="$WALL_TIME" followup_job.sbatch) || exit 1
echo "Submitted follow-up GPU job $JOB ($NUM_GPUS GPU(s), $CPUS_PER_TASK CPUs, ${MEM_GB} GB, $WALL_TIME)."
echo "Watch progress:   bash monitor_followup.sh        (or: squeue -u \$USER)"
