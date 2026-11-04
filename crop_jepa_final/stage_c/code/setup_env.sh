#!/bin/bash
# Idempotent: creates ./venv once (python 3.11 module + uv) with the same package versions as Stage B.
# The Stage B environment in ~/crop_jepa_cluster/venv is never touched.
set -o pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./config_stageC.env
if type module >/dev/null 2>&1; then module load python/3.11.14 2>/dev/null || echo "NOTE: module load python/3.11.14 failed - using python3 from PATH"; fi

if [ -f venv/.ready ]; then echo "[setup_env] venv already prepared."; exit 0; fi

if command -v uv >/dev/null 2>&1; then
  uv venv venv || exit 1
  source venv/bin/activate
  INSTALL="uv pip install"
else
  python3 -m venv venv || exit 1
  source venv/bin/activate
  python -m ensurepip --upgrade >/dev/null 2>&1
  python -m pip install -U pip >/dev/null
  INSTALL="python -m pip install"
fi

echo "[setup_env] installing: $TORCH_SPEC $OTHER_PKGS"
$INSTALL $TORCH_SPEC $OTHER_PKGS || { echo "[setup_env] ERROR: package install failed (internet / PyPI reachable?)"; exit 1; }
python -c "import torch, torchvision, sklearn, scipy, pandas, matplotlib; print('[setup_env] torch', torch.__version__, '| torchvision', torchvision.__version__)" || exit 1
touch venv/.ready
echo "[setup_env] done."
