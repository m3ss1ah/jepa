# Stage A: controlled re-validation of vegetation-aware and multi-scale I-JEPA

[`Crop_JEPA_StageA.ipynb`](Crop_JEPA_StageA.ipynb) is the Stage A notebook. It is a 2×2 factorial test of the two original Crop-JEPA modifications against plain I-JEPA:

| arm | vegetation-aware targets | multi-scale targets |
|---|---|---|
| `ijepa` | no | no |
| `veg` | yes | no |
| `ms` | no | yes |
| `veg_ms` | yes | yes |

It also has a random-init baseline, a supervised baseline and an optional DINOv2 reference, with several seeds, several label draws and paired statistics. The statistics are a hierarchical bootstrap, Wilcoxon tests, Holm correction and factorial effects. Gate G1 asks whether vegetation-aware sampling survives.

**Modes:** `RUN_MODE` = `smoke` (generated data, minutes), `benchmark` or `full`. Re-running resumes: finished runs are skipped, and unfinished runs continue from their checkpoint.

**This copy has no saved outputs.** It is the code as uploaded. Stage A results are on the cluster and are not in this repository.

**Where it leads:** Stage B ([`../stage_b`](../stage_b)) narrowed the study to vegetation-guided target placement and added position-matched controls. Stage C ([`../stage_c`](../stage_c)) is the final confirmatory study.
