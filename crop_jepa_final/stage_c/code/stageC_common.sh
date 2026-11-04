#!/bin/bash
# Sourced by every Stage C script (after `cd` into the Stage C folder). Exports the shared paths and, through
# phase_env, the exact per-phase settings of PHASE_PLAN in crop_jepa_stageC.py (the Python side re-checks them).
source ./config_stageC.env
export TORCH_HOME="$(pwd)/torch_home"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export LOGS="$OUT_ROOT/logs" STATE="$OUT_ROOT/state"
export CJ_LOCKED_FILE="$OUT_ROOT/phase0/stats/stageC_locked.json"
export CJ_PREREG_FILE="$OUT_ROOT/phase0/stats/preregistration_stageC.json"
export CJ_PREREG_CONFIRMED="$OUT_ROOT/PREREG_CONFIRMED"
export CJ_DELETE_LAST_PT=1 CJ_SAVE_FEATURES=1
PHASES="core splitB vits bridge"
mkdir -p "$LOGS" "$STATE"

seq_csv() { seq -s, "$1" "$2"; }

# phase_env <phase0|core|splitB|vits|bridge>: exports CJ_* for that phase (and clears settings of other phases)
phase_env() {
  unset CJ_SPLITS_FROM CJ_SPLIT_SEED CJ_DINOV2_REPO
  export CJ_PROFILE=big_gpu CJ_TARGET_STEPS=26000 CJ_CTX_CAP=auto CJ_N_DRAWS=10
  case "$1" in
    phase0)
      export CJ_MODE=phase0 CJ_PHASE=core CJ_OUTPUT_DIR="$OUT_ROOT/phase0" CJ_CTX_CAP=none CJ_SPLITS_FROM="$STAGEB_SPLITS"
      export CJ_ARMS="ijepa,veg_pos,veg_neg,pos_matched,neg_matched,attn" CJ_SEEDS=900 CJ_SUP_SEEDS=900 CJ_BASELINES=none
      export CJ_PHASE0_SEED=900 CJ_PHASE0_STEPS=9000 ;;
    core)
      export CJ_PHASE=core CJ_OUTPUT_DIR="$OUT_ROOT/core" CJ_SPLITS_FROM="$STAGEB_SPLITS"
      export CJ_ARMS="ijepa,veg_pos,veg_neg,pos_matched,neg_matched,attn" CJ_SEEDS="$(seq_csv 100 119)"
      export CJ_SUP_SEEDS="$(seq_csv 100 109)" CJ_BASELINES="random_init,supervised"
      if [ "${ENABLE_DINOV2:-0}" = "1" ] && [ -f "$STATE/dinov2_repo" ]; then
        export CJ_BASELINES="random_init,supervised,dinov2" CJ_DINOV2_REPO="$(cat "$STATE/dinov2_repo")"
      fi ;;
    splitB)
      export CJ_PHASE=splitB CJ_OUTPUT_DIR="$OUT_ROOT/splitB" CJ_SPLIT_SEED=20261102
      export CJ_ARMS="ijepa,veg_pos,veg_neg,pos_matched,neg_matched" CJ_SEEDS="$(seq_csv 100 109)" CJ_SUP_SEEDS=none
      export CJ_BASELINES="random_init" ;;
    vits)
      export CJ_PHASE=vits CJ_OUTPUT_DIR="$OUT_ROOT/vits" CJ_SPLITS_FROM="$STAGEB_SPLITS" CJ_PROFILE=big_gpu_vits160
      export CJ_ARMS="ijepa,veg_pos" CJ_SEEDS="$(seq_csv 100 109)" CJ_SUP_SEEDS=none CJ_BASELINES="random_init" ;;
    bridge)
      export CJ_PHASE=bridge CJ_OUTPUT_DIR="$OUT_ROOT/bridge" CJ_SPLITS_FROM="$STAGEB_SPLITS" CJ_TARGET_STEPS=9000
      export CJ_CTX_CAP=none CJ_ARMS="ijepa,veg_pos" CJ_SEEDS="$(seq_csv 100 109)" CJ_SUP_SEEDS=none CJ_BASELINES=none ;;
    *) echo "unknown phase $1"; return 1 ;;
  esac
}
