#!/bin/bash
# Packages Stage C results (no checkpoints, no image cache) + a SHA-256 checksum, as in the cluster guide.
#   bash package_results.sh            -> stageC_results.zip   (reports, stats, tables, figures, logs, per-run json/csv/npz)
#   bash package_results.sh features   -> also include the float16 features_*.npy (larger; lets you recompute everything)
cd "$(dirname "$(readlink -f "$0")")" || exit 1
source ./config_stageC.env
EXCL=(-x "*.pt" -x "*/cache/*" -x "*/_smoke/*" -x "*.tmp")
[ "${1:-}" = "features" ] || EXCL+=(-x "*/features_*.npy")
rm -f stageC_results.zip
if command -v zip >/dev/null 2>&1; then
  zip -r -q stageC_results.zip "$OUT_ROOT" crop_jepa_stageC.py Crop_JEPA_StageC.ipynb config_stageC.env stageC_common.sh "${EXCL[@]}"
  F=stageC_results.zip
else
  F=stageC_results.tar.gz
  tar --exclude="*.pt" --exclude="*/cache/*" --exclude="*/_smoke/*" --exclude="*.tmp" \
      $([ "${1:-}" = "features" ] || echo --exclude="*/features_*.npy") \
      -czf "$F" "$OUT_ROOT" crop_jepa_stageC.py Crop_JEPA_StageC.ipynb config_stageC.env stageC_common.sh
fi
sha256sum "$F" | tee "$F.sha256"
ls -lh "$F"
echo "Download on Windows (PowerShell):  scp $USER@cluster.gpu.spit.ac.in:~/crop_jepa_stageC/$F \"C:\\Users\\kalhan\\Desktop\\\""
echo "Then compare:  Get-FileHash \"C:\\Users\\kalhan\\Desktop\\$F\" -Algorithm SHA256"
