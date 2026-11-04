#!/bin/bash
#SBATCH --job-name=cropjepa_prep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs_stageB/logs/%x_%j.out
#SBATCH --error=outputs_stageB/logs/%x_%j.err
# CPU-only job: builds the Python environment and downloads the dataset. No GPU is used or blocked.
# Safe to run again at any time (it only does what is still missing).

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
source ./config.env
module load python/3.11.14 2>/dev/null || echo "NOTE: module load python/3.11.14 not available - using python3 from PATH"
mkdir -p "$CJ_OUTPUT_DIR/logs" "$CJ_OUTPUT_DIR/state"
echo "=== PREP START $(date) on $HOSTNAME ==="

bash ./setup_env.sh || { echo "PREP FAILED: environment"; exit 1; }
source venv/bin/activate

echo "--- internet check (JRC dataset server) ---"
python - <<'PY'
import urllib.request, sys
try:
    r = urllib.request.urlopen("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/B11/", timeout=60)
    print("JRC server reachable, HTTP", r.status)
except Exception as e:
    print("WARNING: cannot reach the JRC server from this node:", e)
    print("         -> see 'If the download fails' in the guide (upload lucas_data/ yourself).")
PY

python -u download_lucas.py 2>&1 | tee -a "$CJ_OUTPUT_DIR/logs/download.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "PREP FAILED: dataset incomplete (exit $rc). Re-run: sbatch prep_job.sbatch"; exit 1; fi
touch "$CJ_OUTPUT_DIR/state/prep.ok"
echo "=== PREP OK $(date) ==="
