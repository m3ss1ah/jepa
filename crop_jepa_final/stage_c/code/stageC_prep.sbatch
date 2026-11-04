#!/bin/bash
#SBATCH --job-name=cjC_prep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs_stageC/logs/%x_%j.out
#SBATCH --error=outputs_stageC/logs/%x_%j.err
# CPU only: Python environment, read-only checks of the Stage B inputs, DINOv2 download. No GPU is used.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
source ./stageC_common.sh
module load python/3.11.14 2>/dev/null || echo "NOTE: module load python/3.11.14 not available - using python3 from PATH"
echo "=== Stage C PREP START $(date) on $HOSTNAME ==="

bash ./setup_env.sh || { echo "PREP FAILED: environment"; exit 1; }
source venv/bin/activate

echo "--- Stage B inputs (read only) ---"
for f in "$CJ_CACHE_DIR/images_192.npy" "$CJ_CACHE_DIR/images_192.csv" "$STAGEB_SPLITS" "$STAGEB_SEED_LEVEL"; do
  if [ -f "$f" ]; then ls -l "$f"; else echo "PREP FAILED: missing $f"; exit 1; fi
done
python - <<'PY' || { echo "PREP FAILED: Stage B cache / splits do not match"; exit 1; }
import os, numpy as np, pandas as pd
c = os.environ["CJ_CACHE_DIR"]
arr = np.load(f"{c}/images_192.npy", mmap_mode="r")
meta = pd.read_csv(f"{c}/images_192.csv", keep_default_na=False)
s = pd.read_csv(os.environ["STAGEB_SPLITS"], keep_default_na=False)
assert arr.shape == (len(meta), 192, 192, 3) and arr.dtype == np.uint8, (arr.shape, arr.dtype)
assert set(s["path"]) <= set(meta["path"]), "splits.csv rows missing from the cache"
n = s["split"].value_counts().to_dict()
assert (n["train"], n["val"], n["test"]) == (11115, 2380, 2381), n
print("cache", arr.shape, "| split A", n, "| OK")
PY

echo "--- DINOv2 reference (optional) ---"
rm -f "$STATE/dinov2_repo"
if [ "${ENABLE_DINOV2:-0}" = "1" ]; then
  python - <<'PY' && echo "DINOv2 ready: $(cat "$STATE/dinov2_repo")" || echo "NOTE: DINOv2 unavailable (no GitHub access?) - the study runs without this optional reference."
import glob, os, torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)   # downloads code + weights into TORCH_HOME
repos = sorted(glob.glob(os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_*")))
assert repos, "checkout not found"
m = torch.hub.load(repos[-1], "dinov2_vits14", source="local")             # offline load, as the GPU job does it
with torch.no_grad():
    out = m(torch.rand(2, 3, 224, 224))
assert tuple(out.shape) == (2, 384), out.shape
open(os.path.join(os.environ["STATE"], "dinov2_repo"), "w").write(repos[-1])
PY
fi
touch "$STATE/prep.ok"
echo "=== Stage C PREP OK $(date) ==="
