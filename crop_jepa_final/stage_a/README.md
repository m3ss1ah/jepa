# Stage A: controlled re-validation of vegetation-aware and multi-scale I-JEPA

[`Crop_JEPA_StageA.ipynb`](Crop_JEPA_StageA.ipynb) is the Stage A notebook **with its executed outputs**: the full run, figures, tables and Gate G1. It is a 2×2 factorial test of the two original Crop-JEPA modifications against plain I-JEPA:

| arm | vegetation-aware targets | multi-scale targets |
|---|---|---|
| `ijepa` | no | no |
| `veg` | yes | no |
| `ms` | no | yes |
| `veg_ms` | yes | yes |

There is also a random-init baseline and a supervised-from-scratch baseline, with paired statistics: a hierarchical bootstrap, Wilcoxon tests, Holm correction and factorial effects. Gate G1 asks whether vegetation-aware sampling survives.

## The run recorded in this notebook

- **Hardware:** laptop, NVIDIA RTX 3050 Laptop GPU (4 GB); profile `laptop_4gb`, bf16.
- **Data:** a **1,200-image LUCASvision subset**, 100 images per class for 12 classes, split 840 train / 180 val / 180 test (chance 8.3%). The images were fetched by the notebook's `setup_lucas_data(limit_per_class=100)` cell.
- **Model:** ViT-Tiny (dim 192, depth 12), patch 16 at 160 px; predictor dim 128 × depth 4; batch 128; 200 epochs.
- **Scale:** 3 seeds × 5 label draws; 18 queue items, all finished.

## Results (linear probe, test macro-F1 %, mean ± SD over seeds × draws)

| arm | 10-shot | 25-shot | 10 % labels | 100 % labels |
|---|---|---|---|---|
| ijepa | 22.7 ± 2.3 | 25.2 ± 3.4 | 19.7 ± 1.6 | 29.8 ± 2.2 |
| veg | 24.7 ± 2.9 | 26.9 ± 3.1 | 22.5 ± 2.7 | 29.9 ± 4.2 |
| ms | 22.2 ± 2.6 | 25.0 ± 2.3 | 19.2 ± 1.9 | 28.5 ± 1.4 |
| veg_ms | 24.1 ± 2.3 | 27.3 ± 2.9 | 22.3 ± 2.3 | 30.7 ± 3.7 |
| random_init | 21.3 ± 1.6 | 22.5 ± 2.0 | 19.2 ± 2.9 | 28.3 ± 1.1 |
| supervised from scratch | – | – | 21.8 ± 3.4 | 53.3 ± 2.2 |

- **Paired differences vs ijepa (probe).** veg is +1.96 pp at 10-shot, +1.73 pp at 25-shot and +2.84 pp at 10% labels. All 95% CIs include 0.
- **Factorial effects.** The vegetation main effect at 10% labels is +2.96 pp [+0.24, +5.66]. Multi-scale has no detectable effect in any regime.
- **Gate G1: FAIL.** veg and veg_ms win 0 of the 3 low-label regimes, harm 0, and have a mean difference of about +1.0 pp. The notebook's recorded next step: *"Drop ExG from the headline; report Stage A as a controlled-study note."*

**How to read this.** On 840 training images, SSL is only a little above an untrained network, and the CIs are wide (n = 3 seeds). Stage A therefore showed a *trend* for vegetation-aware placement, no benefit from multi-scale targets, and that a larger, better-controlled study was needed.

## Where it leads

- **Stage B** ([`../stage_b`](../stage_b)) dropped multi-scale targets. It retested vegetation-guided placement on the full 15,876-image dataset, with position-matched controls.
- **Stage C** ([`../stage_c`](../stage_c)) is the final confirmatory study. Under a fair, controlled protocol, vegetation-guided placement is equivalent to uniform I-JEPA masking (see [`../docs/STAGE_C_RESULTS_REPORT.md`](../docs/STAGE_C_RESULTS_REPORT.md)).

The outputs show the notebook's run folder (a local OneDrive path, `...\vjepa\try 1\outputs`); that is where this run was executed.
