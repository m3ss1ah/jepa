#!/bin/bash
#SBATCH --job-name=cropjepa
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --output=outputs_stageB/logs/%x_%j.out
#SBATCH --error=outputs_stageB/logs/%x_%j.err
# =====================================================================================
#  ONE job that does everything, in order, and can be killed / re-submitted at any point:
#   phase 1  smoke   : whole pipeline on generated data (catches any cluster/env bug in minutes)
#   phase 2  pilot   : Gate 0 (does training produce usable features?)  -> STOP here if it FAILS
#   phase 3  grid    : 4 arms x 6 seeds + baselines, NUM_SHARDS workers sharing the GPU
#   phase 4  report  : statistics, figures, paper tables, results_bundle.zip
#  Every phase leaves a marker in outputs_stageB/state/ and every training run checkpoints to
#  disk, so a restart continues where it stopped - nothing is ever redone.
# =====================================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
source ./config.env
module load python/3.11.14 2>/dev/null || echo "NOTE: module load python/3.11.14 not available - using python3 from PATH"
OUT="$CJ_OUTPUT_DIR"; LOGS="$OUT/logs"; STATE="$OUT/state"
mkdir -p "$LOGS" "$STATE"
JOB_START=$(date +%s)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/pipeline.log"; }

# ---------- remaining wall-time (hours), minus a 45 min safety margin ----------
remaining_hours() {
  local now end
  now=$(date +%s)
  end=""
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    local e; e=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | head -1 | cut -d= -f2)
    [ -n "$e" ] && [ "$e" != "Unknown" ] && end=$(date -d "$e" +%s 2>/dev/null)
  fi
  [ -z "$end" ] && end=$(( JOB_START + WALL_HOURS * 3600 ))
  python3 -c "print(round(max(0.05, ($end - $now) / 3600.0 - 0.75), 3))"
}

stop_here() {   # permanent stop: no automatic re-submit
  echo "$1" > "$STATE/STOP"; log "STOP: $1"; exit 0
}

log "================ JOB $SLURM_JOB_ID START on $HOSTNAME ================"
log "GPU(s): $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | tr '\n' ';')"

# ---------- environment ----------
bash ./setup_env.sh >> "$LOGS/setup_env.log" 2>&1 || stop_here "environment setup failed - see $LOGS/setup_env.log"
source venv/bin/activate
python - <<'PY' || NEED_FALLBACK=1
import torch, sys
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda available:", ok, "| device:", torch.cuda.get_device_name(0) if ok else "-")
if not ok: sys.exit(1)
x = torch.ones(64, 64, device="cuda"); print("GPU matmul ok:", float((x @ x).sum()))
PY
if [ "${NEED_FALLBACK:-0}" = "1" ]; then
  log "PyPI torch cannot use the GPU - trying $TORCH_FALLBACK_INDEX"
  (uv pip install --reinstall torch torchvision --index-url "$TORCH_FALLBACK_INDEX" 2>&1 || pip install --force-reinstall torch torchvision --index-url "$TORCH_FALLBACK_INDEX" 2>&1) >> "$LOGS/setup_env.log"
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || stop_here "torch cannot see the GPU even with the fallback wheel. Send $LOGS/setup_env.log + nvidia-smi output to the cluster admins."
fi

# workers share the CPUs: limit BLAS/OpenMP threads per process
THREADS=$(( SLURM_CPUS_PER_TASK / (NUM_SHARDS + 1) )); [ "$THREADS" -lt 2 ] && THREADS=2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

# ---------- dataset (normally already downloaded by prep_job; this is only a safety net) ----------
if [ ! -f "$STATE/prep.ok" ]; then
  log "dataset not marked complete - running download_lucas.py now (resumable)"
  python -u download_lucas.py 2>&1 | tee -a "$LOGS/download.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || stop_here "dataset download failed - see $LOGS/download.log (guide: 'If the download fails')"
  touch "$STATE/prep.ok"
fi

# ---------- phase 1: smoke ----------
if [ ! -f "$STATE/smoke.ok" ] && [ "${SKIP_SMOKE:-0}" != "1" ]; then
  log "PHASE 1/4  smoke test (generated data, a few minutes)"
  CJ_MODE=smoke python -u crop_jepa_stageB.py 2>&1 | tee -a "$LOGS/phase1_smoke.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || stop_here "smoke test FAILED - see $LOGS/phase1_smoke.log (nothing expensive was started)"
  touch "$STATE/smoke.ok"; log "smoke test PASSED"
else
  log "PHASE 1/4  smoke: already done / skipped"
fi

# ---------- phase 2: pilot = Gate 0 ----------
G0="$OUT/stats/gate_G0.json"
if [ ! -f "$STATE/pilot.ok" ]; then
  log "PHASE 2/4  pilot (Gate 0): $CJ_PILOT_STEPS steps; downloads/caches data on first run; budget $(remaining_hours) h"
  CJ_MODE=pilot CJ_SKIP_REPORT=1 CJ_MAX_SESSION_HOURS=$(remaining_hours) python -u crop_jepa_stageB.py 2>&1 | tee -a "$LOGS/phase2_pilot.log"
  if [ -f "$G0" ]; then
    touch "$STATE/pilot.ok"
  else
    log "pilot not finished (time budget reached or error) - see $LOGS/phase2_pilot.log"
  fi
fi
if [ -f "$STATE/pilot.ok" ]; then
  DECISION=$(python -c "import json;print(json.load(open('$G0'))['decision'])")
  log "GATE 0 decision: $DECISION"
  if [ "$DECISION" != "PASS" ] && [ "$CJ_IGNORE_GATE0" != "1" ]; then
    stop_here "Gate 0 did not pass ($DECISION). The grid was NOT started. Read $G0 (field 'recommendation') and $OUT/figures/gate0_curves.png. To run the grid anyway set CJ_IGNORE_GATE0=1 in config.env and re-run run_all.sh."
  fi

  # ---------- phase 3: grid ----------
  REM=$(remaining_hours)
  log "PHASE 3/4  core grid with $NUM_SHARDS worker(s) on one GPU; budget $REM h"
  PIDS=()
  for ((i=0; i<NUM_SHARDS; i++)); do
    CJ_MODE=full CJ_SHARD_ID=$i CJ_NUM_SHARDS=$NUM_SHARDS CJ_SKIP_REPORT=1 CJ_MAX_SESSION_HOURS=$REM \
      python -u crop_jepa_stageB.py > "$LOGS/phase3_grid_worker${i}.log" 2>&1 &
    PIDS+=($!)
    sleep 20      # stagger start-up so the workers do not all read the cache at once
  done
  for p in "${PIDS[@]}"; do wait "$p" || log "worker pid $p exited with an error (its log is in $LOGS) - the report pass will retry unfinished runs"; done
  log "grid workers finished: $(python check_complete.py | head -1)"

  # ---------- phase 4: report (also finishes any run a worker could not complete) ----------
  REM=$(remaining_hours)
  log "PHASE 4/4  report: statistics, figures, paper tables, bundle; budget $REM h"
  CJ_MODE=full CJ_NUM_SHARDS=1 CJ_SHARD_ID=0 CJ_MAX_SESSION_HOURS=$REM python -u crop_jepa_stageB.py 2>&1 | tee -a "$LOGS/phase4_report.log"
  REPORT_RC=${PIPESTATUS[0]}
  if python check_complete.py | tee -a "$LOGS/pipeline.log"; then
    if [ "$REPORT_RC" -eq 0 ] && [ -f "$OUT/results_bundle.zip" ]; then
      echo "$(date)" > "$STATE/DONE"; log "ALL DONE. Everything is in $OUT (start with $OUT/results_bundle.zip)"
    else
      stop_here "all runs finished but the report step exited with code $REPORT_RC - see $LOGS/phase4_report.log. Fix/ask, then run: bash run_all.sh (only the report is redone)."
    fi
  fi
fi

# ---------- not finished (ran out of wall time / transient failure): re-submit myself ----------
if [ ! -f "$STATE/DONE" ] && [ ! -f "$STATE/STOP" ]; then
  A=$(( $(cat "$STATE/attempts" 2>/dev/null || echo 0) + 1 )); echo "$A" > "$STATE/attempts"
  python check_complete.py > /dev/null
  N_NOW=$(cat "$STATE/progress" 2>/dev/null || echo 0)
  N_BEFORE=$(cat "$STATE/last_progress" 2>/dev/null || echo -1); echo "$N_NOW" > "$STATE/last_progress"
  if [ "$A" -ge "$MAX_ATTEMPTS" ]; then stop_here "gave up after $A attempts. Look at $LOGS and run: bash run_all.sh"; fi
  if [ "$N_NOW" -le "$N_BEFORE" ] && [ "$A" -gt 1 ]; then stop_here "no progress since the previous attempt. Look at $LOGS/phase3_grid_worker*.log and runs/*/error.txt"; fi
  log "not finished yet - re-submitting automatically (attempt $A/$MAX_ATTEMPTS)"
  sbatch "${SLURM_SUBMIT_DIR}/gpu_job.sbatch" | tee -a "$LOGS/pipeline.log" || log "automatic re-submit failed: run 'bash run_all.sh' yourself"
fi
log "================ JOB $SLURM_JOB_ID END ================"
