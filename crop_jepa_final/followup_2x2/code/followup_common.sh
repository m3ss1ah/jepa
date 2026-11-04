#!/bin/bash
# Sourced by every follow-up script (after `cd` into the follow-up folder).
# Stage B and Stage C are only READ: the image cache, split A, the Stage C Phase-0 decisions, the frozen Stage C
# pre-registration, the Stage C core/bridge results (for the 2x2 analysis) and the Stage C Python environment.
export STAGEB_DIR="$HOME/crop_jepa_cluster"
export STAGEC_DIR="$HOME/crop_jepa_stageC"
export CJ_CACHE_DIR="$STAGEB_DIR/outputs_stageB/cache"
export STAGEB_SPLITS="$STAGEB_DIR/outputs_stageB/splits.csv"
export STAGEC_OUT="$STAGEC_DIR/outputs_stageC"
export STAGEC_VENV="$STAGEC_DIR/venv"

export OUT_ROOT="./outputs_followup"
export CJ_FOLLOWUP_ROOT="$OUT_ROOT"
export CJ_FOLLOWUP_FREEZE="$OUT_ROOT/followup_plan.json"
export CJ_LOCKED_FILE="$STAGEC_OUT/phase0/stats/stageC_locked.json"
export CJ_PREREG_FILE="$STAGEC_OUT/phase0/stats/preregistration_stageC.json"
export CJ_PREREG_CONFIRMED="$STAGEC_OUT/PREREG_CONFIRMED"

# job resources: 3 workers per GPU (the per-GPU setup of Stage B job 2070 and Stage C job 2146); NUM_GPUS=1 also works
export PARTITION="general" NUM_GPUS=2 WORKERS_PER_GPU=3 CPUS_PER_TASK=48 MEM_GB=96 WALL_TIME="12:00:00"

export TORCH_HOME="$(pwd)/torch_home"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export CJ_DELETE_LAST_PT=1 CJ_SAVE_FEATURES=1 CJ_ALLOW_CODE_CHANGE=0
export LOGS="$OUT_ROOT/logs" STATE="$OUT_ROOT/state"
FU_PHASES="cap9k uncap26k"      # cap9k first: it is cheap (~1 h) and alone gives the cap effect at the Stage B length
mkdir -p "$LOGS" "$STATE"

# phase_env <cap9k|uncap26k>: the exact FOLLOWUP_PLAN settings (the Python side re-checks them)
phase_env() {
  export CJ_PROFILE=big_gpu CJ_N_DRAWS=10 CJ_SPLITS_FROM="$STAGEB_SPLITS"
  export CJ_ARMS="ijepa,veg_pos" CJ_SEEDS="$(seq -s, 100 109)" CJ_SUP_SEEDS=none CJ_BASELINES=none
  unset CJ_SPLIT_SEED CJ_DINOV2_REPO
  case "$1" in
    cap9k)    export CJ_PHASE=cap9k    CJ_OUTPUT_DIR="$OUT_ROOT/cap9k"    CJ_TARGET_STEPS=9000  CJ_CTX_CAP=auto ;;
    uncap26k) export CJ_PHASE=uncap26k CJ_OUTPUT_DIR="$OUT_ROOT/uncap26k" CJ_TARGET_STEPS=26000 CJ_CTX_CAP=none ;;
    *) echo "unknown follow-up phase $1"; return 1 ;;
  esac
}

# finished runs of a phase (eval.json present), out of 20
n_done() { ls "$OUT_ROOT/$1"/runs/*/eval.json 2>/dev/null | wc -l; }
