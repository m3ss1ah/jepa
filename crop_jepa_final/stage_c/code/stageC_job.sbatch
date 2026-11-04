#!/bin/bash
#SBATCH --job-name=cjC
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=outputs_stageC/logs/%x_%j.out
#SBATCH --error=outputs_stageC/logs/%x_%j.err
# =====================================================================================
#  Crop-JEPA Stage C - ONE GPU job that does everything in order; it can be killed / re-submitted at any point.
#  (run_stageC.sh submits it with the GPU / CPU / memory / time values of config_stageC.env; the #SBATCH values
#  above are only defaults and match that file.) --requeue lets Slurm restart the job after a node failure; the
#  explicit re-submission at the end only happens after a normal exit, and both are guarded against duplicates.
#   1 smoke   : whole pipeline on generated data (minutes)
#   2 phase0  : validation-only decisions + FROZEN pre-registration (SHA-256) -> sanity gate -> main phases released
#   3 phases  : core -> splitB -> vits -> bridge; each = prepare pass, NUM_GPUS x WORKERS_PER_GPU workers, report pass
#   4 final   : cross-phase report (meta-analysis, bridge pooled with Stage B) -> outputs_stageC/FINAL_REPORT.md
#  Every step leaves a marker in outputs_stageC/state/ and every run checkpoints, so a restart continues where it
#  stopped. Nothing in ~/crop_jepa_cluster (Stage B) is ever written.
# =====================================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
source ./stageC_common.sh
module load python/3.11.14 2>/dev/null || echo "NOTE: module load python/3.11.14 not available - using python3 from PATH"
JOB_START=$(date +%s)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/pipeline.log"; }
stop_here() { echo "$1" > "$STATE/STOP"; log "STOP: $1"; exit 0; }

remaining_hours() {   # remaining wall time minus a 45 min safety margin
  local now end e
  now=$(date +%s); end=""
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    e=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | head -1 | cut -d= -f2)
    [ -n "$e" ] && [ "$e" != "Unknown" ] && end=$(date -d "$e" +%s 2>/dev/null)
  fi
  [ -z "$end" ] && end=$(( JOB_START + $(walltime_seconds) ))
  python3 -c "print(round(max(0.05, ($end - $now) / 3600.0 - 0.75), 3))"
}
walltime_seconds() {  # WALL_TIME of config_stageC.env ([D-]HH:MM:SS) in seconds; 36 h if it cannot be parsed
  python3 -c "
import re, sys
m = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', sys.argv[1].strip())
print(int(m[1] or 0) * 86400 + int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]) if m else 36 * 3600)" "${WALL_TIME:-}"
}
other_cjC_jobs() {    # my other cjC jobs in the given states (comma list), excluding this job
  squeue -u "$USER" -h -n cjC -t "$1" -o "%i" 2>/dev/null | grep -vx "${SLURM_JOB_ID:-none}"
}

log "================ Stage C JOB ${SLURM_JOB_ID:-local} START on $HOSTNAME ================"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "$(other_cjC_jobs RUNNING)" ]; then
  log "another cjC job is already RUNNING ($(other_cjC_jobs RUNNING | tr '\n' ' ')) - this job exits without touching anything"
  exit 0
fi
log "GPU(s): $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
[ -f "$STATE/prep.ok" ] || stop_here "prep job has not completed (outputs_stageC/state/prep.ok missing) - run: bash run_stageC.sh"
source venv/bin/activate

# ---------- GPU list (Slurm sets CUDA_VISIBLE_DEVICES to the allocated GPUs) ----------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"; else GPUS=($(seq 0 $((NUM_GPUS - 1)))); fi
NG=${#GPUS[@]}
[ "$NG" -ge 1 ] || stop_here "no GPU visible in the allocation"
[ "$NG" -eq "$NUM_GPUS" ] || log "NOTE: $NG GPU(s) visible, config asks for $NUM_GPUS - using $NG"
NW=$(( NG * WORKERS_PER_GPU ))
python - <<'PY' || stop_here "PyTorch cannot use the GPU(s) - see $LOGS/setup_env.log; ask the cluster admins"
import torch, sys
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda:", ok, "| devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if ok else "-")
sys.exit(0 if ok else 1)
PY
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK} / (NW + 1) )); [ "$THREADS" -lt 2 ] && THREADS=2
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
log "workers: $NW ($WORKERS_PER_GPU per GPU on GPUs ${GPUS[*]}), $THREADS CPU threads each"

# ---------- 1 smoke ----------
if [ ! -f "$STATE/smoke.ok" ]; then
  log "STEP 1  smoke test (generated data)"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} CJ_MODE=smoke CJ_OUTPUT_DIR="$OUT_ROOT/_smoke" CJ_PHASE=core CJ_CTX_CAP=none \
    python -u crop_jepa_stageC.py > "$LOGS/step1_smoke.log" 2>&1 \
    || stop_here "smoke test FAILED - see $LOGS/step1_smoke.log (nothing expensive was started)"
  touch "$STATE/smoke.ok"; log "smoke test PASSED"
fi

# ---------- 2 phase 0 ----------
if [ ! -f "$STATE/phase0.ok" ]; then
  log "STEP 2  Phase 0 (validation only): token audit, capped vs uncapped ijepa, cut points, power, pre-registration"
  ( phase_env phase0; CUDA_VISIBLE_DEVICES=${GPUS[0]} CJ_MAX_SESSION_HOURS=$(remaining_hours) python -u crop_jepa_stageC.py ) \
    >> "$LOGS/step2_phase0.log" 2>&1
  if [ -f "$CJ_PREREG_FILE" ] && [ -f "$CJ_PREREG_FILE.sha256" ] && [ -f "$CJ_LOCKED_FILE" ]; then
    touch "$STATE/phase0.ok"; log "Phase 0 done: $(python -c "import json;d=json.load(open('$CJ_LOCKED_FILE'));print('protocol',d['protocol'],'| cap',d['ctx_tokens_cap'])")"
  elif grep -q "TimeBudgetExceeded\|time budget" "$LOGS/step2_phase0.log"; then
    log "Phase 0 stopped by the time budget - will resume after re-submission"
  else
    stop_here "Phase 0 FAILED - see $LOGS/step2_phase0.log"
  fi
fi
# ---------- Phase-0 sanity gate + internal freeze record (no external registration) ----------
# The main phases are released automatically once Phase 0 passes this gate. PREREG_CONFIRMED records the frozen
# pre-registration hash and the release time; every phase still verifies that hash, the plan and the code hash.
if [ -f "$STATE/phase0.ok" ] && [ ! -f "$CJ_PREREG_CONFIRMED" ]; then
  log "Phase-0 sanity gate"
  python - <<'PY' 2>&1 | tee -a "$LOGS/pipeline.log"
import json, os, sys
d = json.load(open(os.environ["CJ_LOCKED_FILE"]))
r = d["phase0_runs"]
bad = [f"{k}: {r[k]['nonfinite_steps']} non-finite steps" for k in ("uncapped", "capped") if r[k]["nonfinite_steps"] != 0]
if not r["uncapped"]["knn_val_f1_last"] >= 0.30:   # Stage B ijepa at the same 9,030 steps: 0.463-0.468
    bad.append(f"uncapped k-NN val F1 {r['uncapped']['knn_val_f1_last']} < 0.30")
if not r["uncapped"]["rankme_last"] >= 20:         # Stage B ijepa: 47-51
    bad.append(f"uncapped RankMe {r['uncapped']['rankme_last']} < 20")
print(f"  protocol {d['protocol']} | cap {d['ctx_tokens_cap']} | k-NN uncapped {r['uncapped']['knn_val_f1_last']:.4f} "
      f"capped {r['capped']['knn_val_f1_last']:.4f} | RankMe uncapped {r['uncapped']['rankme_last']:.1f}")
print("  gate FAILED: " + "; ".join(bad) if bad else "  gate passed")
sys.exit(1 if bad else 0)
PY
  [ "${PIPESTATUS[0]}" -eq 0 ] || stop_here "Phase-0 sanity gate FAILED (see $LOGS/pipeline.log, $CJ_LOCKED_FILE) - main phases NOT started"
  { echo "INTERNAL FREEZE - not externally registered (no OSF registration)."
    echo "pre-registration $CJ_PREREG_FILE sha256 $(cut -d' ' -f1 "$CJ_PREREG_FILE.sha256"), frozen $(date -r "$CJ_PREREG_FILE" '+%F %T')"
    echo "main phases released $(date '+%F %T') by job ${SLURM_JOB_ID:-local} after the Phase-0 sanity gate"; } > "$CJ_PREREG_CONFIRMED"
  log "pre-registration frozen internally (sha256 $(cut -d' ' -f1 "$CJ_PREREG_FILE.sha256")) - main phases released"
fi

# ---------- 3 phases ----------
if [ -f "$STATE/phase0.ok" ] && [ -f "$CJ_PREREG_CONFIRMED" ]; then
  for PH in $PHASES; do
    [ -f "$STATE/$PH.done" ] && continue
    REM=$(remaining_hours)
    log "STEP 3  phase $PH: prepare pass (budget $REM h)"
    ( phase_env "$PH"; CUDA_VISIBLE_DEVICES=${GPUS[0]} CJ_MODE=prepare CJ_MAX_SESSION_HOURS=$REM python -u crop_jepa_stageC.py ) \
      >> "$LOGS/phase_${PH}_prepare.log" 2>&1 || stop_here "phase $PH prepare pass FAILED - see $LOGS/phase_${PH}_prepare.log"
    REM=$(remaining_hours)
    log "        phase $PH: $NW workers (budget $REM h)"
    PIDS=()
    for ((i=0; i<NW; i++)); do
      ( phase_env "$PH"; CUDA_VISIBLE_DEVICES=${GPUS[$((i % NG))]} CJ_MODE=full CJ_SHARD_ID=$i CJ_NUM_SHARDS=$NW CJ_SKIP_REPORT=1 \
          CJ_MAX_SESSION_HOURS=$REM python -u crop_jepa_stageC.py ) >> "$LOGS/phase_${PH}_worker${i}.log" 2>&1 &
      PIDS+=($!)
      sleep 15
    done
    for p in "${PIDS[@]}"; do wait "$p" || log "a $PH worker exited with an error (see its log); the report pass retries unfinished runs"; done
    log "        phase $PH workers finished: $(python check_complete_stageC.py "$PH" | head -1)"
    REM=$(remaining_hours)
    if ! python check_complete_stageC.py "$PH" > /dev/null && python -c "import sys; sys.exit(0 if $REM < 2.0 else 1)"; then
      log "phase $PH incomplete and only $REM h left - re-submitting instead of starting the report pass"
      break
    fi
    log "        phase $PH: report pass (finishes leftovers, statistics, figures, tables; budget $REM h)"
    ( phase_env "$PH"; CUDA_VISIBLE_DEVICES=${GPUS[0]} CJ_MODE=full CJ_NUM_SHARDS=1 CJ_SHARD_ID=0 CJ_MAX_SESSION_HOURS=$REM \
        python -u crop_jepa_stageC.py ) >> "$LOGS/phase_${PH}_report.log" 2>&1
    RC=$?
    if python check_complete_stageC.py "$PH" >> "$LOGS/pipeline.log" && [ "$RC" -eq 0 ] && \
       [ -f "$OUT_ROOT/$PH/stats/gate_G1_stageC_$PH.json" ]; then
      touch "$STATE/$PH.done"; log "phase $PH DONE"
    else
      log "phase $PH not finished (report exit code $RC) - continuing after re-submission"
      break
    fi
  done
  if [ -f "$STATE/bridge.done" ] && [ -f "$STATE/core.done" ] && [ -f "$STATE/splitB.done" ] && [ -f "$STATE/vits.done" ]; then
    log "STEP 4  final cross-phase report"
    if python -u stageC_final_report.py >> "$LOGS/step4_final.log" 2>&1; then
      date > "$STATE/DONE"; log "ALL DONE. Read $OUT_ROOT/FINAL_REPORT.md; package with: bash package_results.sh"
    else
      stop_here "final report FAILED - see $LOGS/step4_final.log (all runs are finished; fix and run: bash run_stageC.sh)"
    fi
  fi
fi

# ---------- not finished: re-submit (with a no-progress guard) ----------
if [ ! -f "$STATE/DONE" ] && [ ! -f "$STATE/STOP" ]; then
  A=$(( $(cat "$STATE/attempts" 2>/dev/null || echo 0) + 1 )); echo "$A" > "$STATE/attempts"
  python check_complete_stageC.py all > /dev/null 2>&1
  N_NOW=$(cat "$STATE/progress" 2>/dev/null || echo 0)
  N_BEFORE=$(cat "$STATE/last_progress" 2>/dev/null || echo -1); echo "$N_NOW" > "$STATE/last_progress"
  [ "$A" -ge "$MAX_ATTEMPTS" ] && stop_here "gave up after $A attempts - look at $LOGS and run: bash run_stageC.sh"
  if [ "$N_NOW" -le "$N_BEFORE" ] && [ "$A" -gt 1 ]; then stop_here "no progress since the previous attempt - see $LOGS/phase_*_worker*.log and runs/*/error.txt"; fi
  log "not finished yet - re-submitting (attempt $A/$MAX_ATTEMPTS)"
  if [ -n "$(other_cjC_jobs PENDING,RUNNING,CONFIGURING,REQUEUED)" ]; then
    log "a cjC job is already queued ($(other_cjC_jobs PENDING,RUNNING,CONFIGURING,REQUEUED | tr '\n' ' ')) - not submitting another"
  else
    bash ./submit_gpu.sh | tee -a "$LOGS/pipeline.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || log "automatic re-submit failed: run 'bash run_stageC.sh' yourself"
  fi
fi
log "================ Stage C JOB ${SLURM_JOB_ID:-local} END ================"
