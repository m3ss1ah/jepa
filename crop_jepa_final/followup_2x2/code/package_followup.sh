#!/bin/bash
# Packages the follow-up results (no checkpoints, no features) + SHA-256:  bash package_followup.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./followup_common.sh
rm -f followup_results.zip
if command -v zip >/dev/null 2>&1; then
  zip -r -q followup_results.zip "$OUT_ROOT" crop_jepa_followup.py make_followup_script.py followup_2x2_report.py \
      followup_common.sh followup_job.sbatch -x "*.pt" -x "*/features_*.npy" -x "*/cache/*" -x "*.tmp"
  F=followup_results.zip
else
  F=followup_results.tar.gz
  tar --exclude="*.pt" --exclude="*/features_*.npy" --exclude="*/cache/*" --exclude="*.tmp" -czf "$F" "$OUT_ROOT" \
      crop_jepa_followup.py make_followup_script.py followup_2x2_report.py followup_common.sh followup_job.sbatch
fi
sha256sum "$F" | tee "$F.sha256"
ls -lh "$F"
echo "Download on Windows (PowerShell):  scp $USER@cluster.gpu.spit.ac.in:~/crop_jepa_followup/$F \"C:\\Users\\kalhan\\Desktop\\crop_jepa_cluster\\\""
