# Crop-JEPA: vegetation-guided target placement in I-JEPA

Does it help a self-supervised I-JEPA encoder if its **prediction targets are placed on vegetation**? Vegetation is scored per patch with the Excess Green index, ExG = (2g − r − b)/(r + g + b). The study runs on LUCASVision crop photographs: 15,876 ground-level images, 12 crop classes.

The project ran in stages on the SPIT GPU cluster (SLURM, RTX 6000 Ada). Stage B was the pilot, and Stage C is the final, confirmatory study, with an analysis plan frozen by SHA-256 before the main runs. This repository holds the code, the notebook, all results tables and statistics, the figures and the logs. It does **not** hold model checkpoints or the image cache (see [What is not included](#what-is-not-included)).

## Headline result

**Under a fair comparison, vegetation-guided masking is equivalent to standard uniform I-JEPA masking.** In the fair comparison, every method gets the same 12 context tokens and trains for 26k steps.

Test macro-F1 (%) from a frozen linear probe. The primary endpoint is the mean over the 10-shot, 25-shot and 10%-label regimes. Values are mean ± SD over pretraining seeds.

| Stage C core (ViT-Tiny/16 @160, 20 seeds) | primary endpoint | 10-shot | 100 % labels |
|---|---|---|---|
| I-JEPA (uniform targets) | 52.4 ± 0.6 | 46.5 | 67.0 |
| **veg_pos** (targets on vegetation) | 52.4 ± 0.7 | 46.2 | 66.9 |
| veg_neg (targets away from vegetation) | 51.9 ± 0.4 | 45.8 | 66.6 |
| pos_matched (veg_pos spatial layout, no content) | 52.0 ± 0.7 | 45.9 | 66.7 |
| neg_matched (veg_neg spatial layout, no content) | 52.5 ± 0.7 | 46.6 | 66.9 |
| attn (AttMask-style saliency) | 52.0 ± 0.5 | 45.9 | 66.5 |
| random init (untrained) | 20.4 | 18.5 | 29.2 |
| supervised from scratch (unmatched reference) | – | – | 74.5 |
| DINOv2 ViT-S/14, frozen (external reference) | 83.6 | 80.7 | 88.0 |

**veg_pos vs uniform I-JEPA:**
- **Core, 20 seeds:** −0.09 pp, 95% CI [−0.59, +0.47]. This is equivalent to zero within ±1 pp.
- **New data split:** −0.20 pp, so the result replicates.
- **ViT-S:** −1.54 pp; veg_pos was worse on all 10 seeds.

**The one confirmed effect is content, not position.** Steering targets *away* from vegetation costs about 0.5 pp, but the same spatial layout with random content costs nothing (neg_matched − veg_neg = +0.52 pp, Holm p = 0.024).

**The pilot's advantage depended on the protocol.** Stage B's small veg_pos advantage reproduces under its original protocol: no context-token cap and 9k steps, giving +0.60 pp pooled over 16 seeds, p = 0.017. It disappears under the controlled protocol.

Full interpretation, with all numbers: [`docs/STAGE_C_RESULTS_REPORT.md`](docs/STAGE_C_RESULTS_REPORT.md).

## Repository layout

```
stage_a/
  Crop_JEPA_StageA.ipynb    2x2 factorial notebook: vegetation-aware x multi-scale targets vs I-JEPA (executed, with results)
docs/
  STAGE_C_RESULTS_REPORT.md         detailed results, interpretation, allowed and forbidden claims
  STAGE_C_MASTER_SPEC_proposal.md   the original Stage C proposal (before the final design)
  STAGE_B_COMPREHENSIVE_ANALYSIS.md collaborator's Stage B write-up (see note below)
stage_b/
  code/      crop_jepa_stageB.py + SLURM scripts (run_all.sh, prep_job.sbatch, gpu_job.sbatch, ...)
  results/   stats/, paper_tables/ (CSV + LaTeX), figures/, logs/, runs/<arm>_s<seed>/ (eval.json, preds.npz, train_log.csv)
stage_c/
  code/      crop_jepa_stageC.py, Crop_JEPA_StageC.ipynb (generated, code identical to the .py), SLURM scripts,
             README_STAGEC.md and STAGE_C_RUN_GUIDE.md (how to run on the cluster)
  results/   FINAL_REPORT.md, consistency_table.csv, final_summary.json, PREREG_CONFIRMED,
             phase0/ (protocol decision, frozen pre-registration + .sha256, power analysis),
             core/ splitB/ vits/ bridge/ (each: stats/, paper_tables/, figures/, runs/), logs/
followup_2x2/
  code/, results/   post-hoc 2x2 follow-up (context-token cap x training length): pipeline, results, README with conclusions
```

Mapping to the paths used inside the reports:
- `stageC_results/outputs_stageC/` → `stage_c/results/`
- `outputs_stageB/` → `stage_b/results/`
- `crop_jepa_stageC/` → `stage_c/code/`

## Stages

| stage | what | outcome |
|---|---|---|
| A | 2×2 factorial: vegetation-aware × multi-scale targets vs I-JEPA ([`stage_a/Crop_JEPA_StageA.ipynb`](stage_a/Crop_JEPA_StageA.ipynb)) | Gate G1 **FAIL** on a 1,200-image laptop run (veg ≈ +1 pp, CIs include 0; multi-scale no effect) → Stage B kept only vegetation placement |
| B | pilot: 4 arms × 6 seeds, 9k steps, uncapped context | veg_pos − ijepa +0.89 pp (inconclusive); veg_pos − veg_neg +3.97 pp. Later shown to depend on the protocol. |
| C | final study: 5 phases, 260 runs, frozen plan, 20-seed core, replication split, ViT-S, bridge to Stage B | decision **P-B**: equivalent to uniform placement |
| follow-up | 2×2: cap × training length (post hoc, exploratory; [`followup_2x2/`](followup_2x2/README.md)) | neither change alone removes the small veg_pos gain (only both together); unequal context did not create it; the 12-token cap itself adds +2.87 pp to I-JEPA at 26k steps (10/10 seeds) |

## Reproducing

Everything runs headless through SLURM.

**Stage C:**
1. Put the code in `~/crop_jepa_stageC` and the Stage B cache in `~/crop_jepa_cluster/outputs_stageB/cache`.
2. Run `bash run_stageC.sh`.

The step-by-step guide is [`stage_c/code/STAGE_C_RUN_GUIDE.md`](stage_c/code/STAGE_C_RUN_GUIDE.md).

**Notebook:** [`stage_c/code/Crop_JEPA_StageC.ipynb`](stage_c/code/Crop_JEPA_StageC.ipynb) is a cell-by-cell mirror of `crop_jepa_stageC.py`, generated by `make_notebook.py`. It's for reading, and can run a smoke test.

**Environment:** Python 3.11, torch 2.8.0, torchvision 0.23.0, numpy, pandas, scikit-learn, scipy, matplotlib (see `setup_env.sh`).

**Integrity:** the Stage C code that ran has SHA-256 `859588be2b085008602897f4a5d10210ce25cbaeb850846c63866fbf7870324e`. The frozen analysis plan is `stage_c/results/phase0/stats/preregistration_stageC.json` (sha256 `c6774dac…9343`, frozen 2026-11-02 21:04:23, before any main run). It was frozen locally and **not externally registered**.

**Follow-up:** `followup_2x2/code/make_followup_script.py` regenerates `crop_jepa_followup.py` from the Stage C script, and checks the hash above. It expects the cluster layout (`crop_jepa_followup/` next to `crop_jepa_stageC/`), so run it there.

## What is not included

- **Model checkpoints** (`*.pt`, ~4.5 GB for Stage B alone) and the **image cache** (`cache/images_192.npy`, 1.6 GB). Both are reproducible: run the pipeline, and `download_lucas.py` fetches the images.
- **Saved float16 features**, the zip bundles, and the smoke-test or synthetic data.
- **Stage A run folder** (checkpoints, cache); the executed Stage A notebook with all its outputs is included.

## Notes

- `docs/STAGE_B_COMPREHENSIVE_ANALYSIS.md` was written by a collaborator. It misstates some of the configuration (aspect ratio, ExG formula, probe C range, k-NN tuning). Where they differ, trust the code and the `stats/` files.
- **Data:** LUCAS ground-level survey photographs (Eurostat); LUCASVision is the manually checked subset used by Yordanov et al., 2023, *Sensors*. These are not aerial or satellite images.
- Every number in the reports can be recomputed from the per-run `eval.json` / `preds.npz` and the `stats/` CSVs.
