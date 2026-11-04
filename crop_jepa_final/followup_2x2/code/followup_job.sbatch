#!/bin/bash
#SBATCH --job-name=cjF
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=outputs_followup/logs/%x_%j.out
#SBATCH --error=outputs_followup/logs/%x_%j.err
# =====================================================================================
#  Crop-JEPA post-hoc FOLLOW-UP: the two missing cells of the 2x2 (token cap x training length), ijepa + veg_pos,
#  seeds 100-109, split A:  cap9k (cap 12, 9,030 steps) and uncap26k (no cap, 26,058 steps).
#  Each phase = prepare pass, NUM_GPUS x 3 workers, report pass; then followup_2x2_report.py combines them
#  with Stage C core (cap/26k) and bridge (uncap/9k). Estimated ~2.5 h on 2 GPUs (~4.5 h on 1); 12 h requested.
#  (run_followup.sh submits with the GPU / CPU / memory / time values of followup_common.sh.)
#  No automatic re-submission: if it stops, `bash run_followup.sh` continues (finished runs are skipped,
#  unfinished runs resume from their checkpoint). Nothing in Stage B or Stage C is written.
# =====================================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
source ./followup_common.sh
module load python/3.11.14 2>/dev/null || echo "NOTE: module load python/3.11.14 not available - using python3 from PATH"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/pipeline.log"; }
stop_here() { echo "$1" > "$STATE/STOP"; log "STOP: $1"; exit 0; }
JOB_START=$(date +%s)
remaining_hours() {   # remaining wall time minus a 30 min safety margin
  local now end e
  now=$(date +%s); end=""
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    e=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | head -1 | cut -d= -f2)
    [ -n "$e" ] && [ "$e" != "Unknown" ] && end=$(date -d "$e" +%s 2>/dev/null)
  fi
  [ -z "$end" ] && end=$(( JOB_START + 12 * 3600 ))
  python3 -c "print(round(max(0.05, ($end - $now) / 3600.0 - 0.5), 3))"
}

log "================ FOLLOW-UP JOB ${SLURM_JOB_ID:-local} START on $HOSTNAME ================"
if [ -n "${SLURM_JOB_ID:-}" ] && squeue -u "$USER" -h -n cjF -t RUNNING -o "%i" 2>/dev/null | grep -vxq "$SLURM_JOB_ID"; then
  log "another cjF job is already RUNNING - this job exits without touching anything"; exit 0
fi
log "GPU(s): $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
[ -f "$STAGEC_VENV/bin/activate" ] || stop_here "Stage C environment not found: $STAGEC_VENV"
source "$STAGEC_VENV/bin/activate"
python - <<'PY' || stop_here "PyTorch cannot use the GPU"
import torch, sys
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 1)
PY
# GPU list: Slurm sets CUDA_VISIBLE_DEVICES to the allocated GPU(s); every worker is pinned to one of them
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"; else GPUS=($(seq 0 $((NUM_GPUS - 1)))); fi
NG=${#GPUS[@]}; GPU=${GPUS[0]}
[ "$NG" -eq "$NUM_GPUS" ] || log "NOTE: $NG GPU(s) visible, config asks for $NUM_GPUS - using $NG"
WORKERS=$(( NG * WORKERS_PER_GPU ))
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK} / (WORKERS + 1) )); [ "$THREADS" -lt 2 ] && THREADS=2
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
log "workers: $WORKERS ($WORKERS_PER_GPU per GPU on GPUs ${GPUS[*]}), $THREADS CPU threads each"

for PH in $FU_PHASES; do
  [ -f "$STATE/$PH.done" ] && continue
  REM=$(remaining_hours)
  log "phase $PH: prepare pass (budget $REM h)"
  ( phase_env "$PH"; CUDA_VISIBLE_DEVICES=$GPU CJ_MODE=prepare CJ_MAX_SESSION_HOURS=$REM python -u crop_jepa_followup.py ) \
    >> "$LOGS/${PH}_prepare.log" 2>&1 || stop_here "phase $PH prepare pass FAILED - see $LOGS/${PH}_prepare.log"
  REM=$(remaining_hours)
  log "phase $PH: $WORKERS workers (budget $REM h)"
  PIDS=()
  for ((i=0; i<WORKERS; i++)); do
    ( phase_env "$PH"; CUDA_VISIBLE_DEVICES=${GPUS[$((i % NG))]} CJ_MODE=full CJ_SHARD_ID=$i CJ_NUM_SHARDS=$WORKERS CJ_SKIP_REPORT=1 \
        CJ_MAX_SESSION_HOURS=$REM python -u crop_jepa_followup.py ) >> "$LOGS/${PH}_worker${i}.log" 2>&1 &
    PIDS+=($!)
    sleep 15
  done
  for p in "${PIDS[@]}"; do wait "$p" || log "a $PH worker exited with an error (see its log); the report pass retries unfinished runs"; done
  log "phase $PH workers finished: $(n_done "$PH")/20 runs"
  REM=$(remaining_hours)
  if [ "$(n_done "$PH")" -lt 20 ] && python3 -c "import sys; sys.exit(0 if $REM < 1.0 else 1)"; then
    stop_here "phase $PH incomplete ($(n_done "$PH")/20) and only $REM h left - run: bash run_followup.sh (resumes)"
  fi
  log "phase $PH: report pass (finishes leftovers, statistics; budget $REM h)"
  ( phase_env "$PH"; CUDA_VISIBLE_DEVICES=$GPU CJ_MODE=full CJ_NUM_SHARDS=1 CJ_SHARD_ID=0 CJ_MAX_SESSION_HOURS=$REM \
      python -u crop_jepa_followup.py ) >> "$LOGS/${PH}_report.log" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ] && [ "$(n_done "$PH")" -eq 20 ] && [ -f "$OUT_ROOT/$PH/stats/seed_level_primary.csv" ]; then
    touch "$STATE/$PH.done"; log "phase $PH DONE (20/20 runs)"
  else
    stop_here "phase $PH not finished (report exit $RC, $(n_done "$PH")/20 runs) - see $LOGS/${PH}_report.log; then: bash run_followup.sh"
  fi
done

log "2x2 analysis"
python -u followup_2x2_report.py >> "$LOGS/report_2x2.log" 2>&1 || stop_here "2x2 report FAILED - see $LOGS/report_2x2.log"
date > "$STATE/DONE"
log "ALL DONE. Read $OUT_ROOT/FOLLOWUP_2x2_REPORT.md; package with: bash package_followup.sh"
log "================ FOLLOW-UP JOB ${SLURM_JOB_ID:-local} END ================"
