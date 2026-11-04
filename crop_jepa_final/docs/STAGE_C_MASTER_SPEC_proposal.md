# Crop-JEPA — Stage C Master Specification

### Final confirmatory study · Complete lineage Stage A → Stage B → Stage C

**Version:** 1.0 (draft for pre-registration lock) · **Date:** 2 November 2026 **Purpose of this file:** a single, self-contained specification from which a notebook-generating AI model (and the human researcher) can build, run, analyse and write up Stage C **without needing any other context**. Stage C is the **last** experimental stage: after it, no new experiments are run; only the paper is written.

---

## 0\. How to use this document

### 0.1 For the human (read this first)

1. Attach **this file plus the companion files in 0.2** to the generator model.  
2. Generate the notebook **in parts** (0.3). Do not ask for the whole notebook in one shot.  
3. Before launching Phase 1, **freeze the pre-registration** (Section 8\) and register it on an external service (OSF or AsPredicted) so the timestamp is independent of your own repository. Stage B's pre-registration was only a self-timestamped JSON file; reviewers prefer an external one.  
4. Fill in every item in the **"Missing inputs" checklist (0.6)**. The generator must not invent them.

### 0.2 Companion files that MUST be attached to the generator

| File | Why it is needed |
| :---- | :---- |
| `crop_jepa_stageB.py` (Stage B source; contains `MaskGenerator`, ViT, predictor, train loop, probe/kNN eval) | Source of truth for every value tagged **\[S-B CODE\]**. Re-use its logic verbatim for the four Stage B arms. |
| Stage B notebook(s) (`Crop_JEPA_StageB.ipynb` / `SW_Crop_JEPA_StageB.ipynb`) | Cell structure, helper utilities, logging style. |
| `STAGE_B_COMPREHENSIVE_ANALYSIS.md` | All Stage B numbers (summarised again in Section 2 and Appendix A). |
| Stage B `outputs_stageB/` items: `stats/dataset_audit.json`, split index files (if saved), `stats/preregistration.json`, `locked_profile.json` | Reproduce the exact train/val/test split and measured throughput. |
| Any Stage A notes / notebook | Section 2.1 has gaps (marked **\[USER TO CONFIRM\]**). |

### 0.3 Generation protocol (give the generator one part at a time)

| Part | Spec sections to implement | Output |
| :---- | :---- | :---- |
| 1 | 9.1–9.4 (config, env lock, utilities, data, indices, model) | `crop_jepa_stageC.py` module part 1 \+ notebook cells 0–6 |
| 2 | 4.6–4.8, 9.5 (masks, training loop, resume) \+ unit tests in Section 10 | module part 2 \+ cells 7–9 |
| 3 | 4.9–4.11 (evaluation, fine-tuning, diagnostics) | module part 3 \+ cells 10–12 |
| 4 | Section 5 and 9.7 (run matrix, orchestrator, SLURM template) | cells 13–15 \+ `run_stageC.sh` |
| 5 | Section 7 (statistics), Section 11 (tables/figures/report) | stats module \+ cells 16–20 |

After each part: run it, paste back errors, fix, only then request the next part.

### 0.4 Non-negotiable rules for the generator model

1. **No placeholders, no `TODO`, no pseudo-code** in final code. Every function in Section 9 must be fully implemented with the stated signature.  
2. **Never invent hyperparameters.** Any value tagged **\[S-B CODE\]** must be copied from the attached Stage B source. If it is not available, emit a cell titled `MISSING INPUTS` listing what is missing and stop; do not guess.  
3. Values tagged **\[LOCKED\]** are fixed by this spec. Values tagged **\[PHASE-0\]** are determined by the stated rule on **validation data only** and then frozen in `locked_profile_stageC.json`.  
4. All randomness must follow the seeding scheme in 4.8. Runs must be bit-for-bit repeatable where the hardware allows, and the notebook must record when it is not.  
5. The orchestrator must be **resumable and idempotent**: a run with a valid `train_done.json` \+ `eval.json` is skipped; all writes are atomic (write to temp, then rename).  
6. **Unit tests (Section 10\) run before any training** and abort on failure.  
7. **The test split is never touched** by Phase 0 or by any design decision. Test metrics are computed only inside `evaluate_run` for locked runs and are never printed during Phase 0\.  
8. Long jobs run in **subprocesses** (not inside the notebook kernel), so a notebook disconnect does not kill training.  
9. No silent `except: pass`. Every exception is logged with the run id and re-raised or recorded in `manifest/run_manifest.csv`.  
10. Keep a `DEVIATIONS.md` log; any departure from this spec must be written there with date and reason.

### 0.5 Tag conventions

`[LOCKED]` fixed here · `[S-B CODE]` copy from Stage B source · `[PHASE-0]` decided by rule on validation data · `[OPTIONAL]` run only if compute allows · `[USER TO CONFIRM]` the human must supply.

### 0.6 Missing inputs checklist (human fills before generation)

- [ ] Stage B source file attached (`crop_jepa_stageB.py`).  
- [ ] Learning-rate peak, warm-up length, AdamW betas, grad-clip value, EMA schedule details, target-feature normalisation (layer-norm or not) — all **\[S-B CODE\]**.  
- [ ] Train-time augmentation (crop/flip/colour) and eval-time view (resize vs centre crop 160 from the 192 cache) — **\[S-B CODE\]**.  
- [ ] Feature standardisation before the logistic probe — **\[S-B CODE\]**.  
- [ ] Whether Stage B saved split indices (file path) — otherwise regenerate with the same regex \+ hash rule and assert the counts in 4.1.  
- [ ] Stage A details: definition of the `ms` factor, hyperparameters, results table (paste into 2.1).  
- [ ] GPU model and VRAM of the Stage C machine; SLURM wall-time limit; number of CPU cores.  
- [ ] Second dataset: choice, local path, licence (Section 4.2).  
- [ ] LUCASVision root path on the Stage C machine.

---

## 1\. The project on one page

**Research question.** Does *vegetation-aware target selection* (choosing which image regions the I-JEPA predictor must predict, using an Excess-Green vegetation score) improve self-supervised representations of agricultural field photographs — and if so, *when* and *why*?

**Why Stage C is the last stage.** Stage A was a data-starved pilot (inconclusive). Stage B had a rigorous design but ended **Mixed / Inconclusive** on its primary hypothesis (+0.89 pp, 95% CI \[−0.01, \+1.90\], Holm p \= 0.0625, n \= 6 seeds, undertrained models). Stage B did confirm two things (masking polarity matters; a static spatial prior is equivalent to uniform masking) and produced one new, unconfirmed lead (benefit concentrates in row crops with visible soil). Stage C is designed so that, **whatever the outcome**, the paper has a defensible conclusion:

* a **confirmed benefit**, or  
* a **confirmed equivalence** ("greenness guidance does not help on average"), or  
* a **bounded effect** with precise confidence intervals,

*plus* a pre-registered test of **when** it helps (scene heterogeneity), a pre-registered **adaptive method** that encodes that finding, and **three levels of replication** (fresh pretraining seeds, fresh data split, fresh dataset/model size).

**Final-paper claims this stage must be able to support or refute** (Section 3.3 maps outcomes to claims):

1. Masking polarity matters (replication of Stage B H3).  
2. Static spatial priors do not explain vegetation-guided gains (replication of Stage B equivalence).  
3. Vegetation-guided masking vs uniform masking — definitive answer at convergence-length training with 20 seeds.  
4. The benefit depends on scene heterogeneity (image-level and class-group tests).  
5. A scene-adaptive guidance rule (`veg_gate`) recovers the gain without the dense-canopy penalty (or fails to).  
6. Results generalise (second split, second dataset, larger model, other vegetation indices).

---

## 2\. Lineage: Stage A → Stage B → what remains open

### 2.1 Stage A (controlled pilot) — what is known

| Item | Value |
| :---- | :---- |
| Hardware | Laptop, RTX 3050, 4 GB VRAM |
| Data | 1,200 images (100 per class × 12 classes), 840 train |
| Seeds | 3 (s0, s1, s2) |
| Arms | 2×2 factorial: `ijepa`, `veg` (greenness-guided targets), `ms`, `veg_ms` — **\[USER TO CONFIRM definition of `ms` from the Stage A notebook\]** |
| Controls | `random_init`, `supervised` |
| Endpoints | 10-shot, 25-shot, 10% labels, analysed **separately** |
| Statistics | Wilcoxon on pooled (seed, label-draw) pairs (**pseudo-replication**), Holm across regimes within comparison |
| SSL vs random init | \+1.5 pp (probe 100%: 29.8% vs 28.3%) — essentially a floor effect |
| Best SSL vs supervised (100% labels) | 30.7% vs 53.3% |
| Greenness-guided vs baseline | G1 gate **FAIL** (0 of 3 regimes won); exploratory k-NN drop of −1.4 to −2.2 pp |
| Lessons | Too little data for ViT to learn; no spatial or polarity controls; pseudo-replicated statistics; too few seeds |

> **Human: paste any further Stage A facts here (hyperparameters, `ms` definition, exact tables).**

### 2.2 Stage B (cluster grid study) — design

| Item | Value |
| :---- | :---- |
| Hardware | SPIT cluster, RTX 6000 Ada 48 GB, CUDA 12.8, PyTorch 2.8.0, BF16; 3,020 img/s, 1.34 GB peak VRAM |
| Data | LUCASVision, 15,876 images, 12 crop classes; split 11,115 / 2,380 / 2,381 (train/val/test); group-aware, 0 leaking groups |
| Model | ViT-Tiny encoder (5.49 M params; D=192, depth 12, heads 3), predictor (0.84 M; D=128, depth 4, heads 4); 160×160 input, 16×16 patches → 10×10 grid; sin-cos positions; no CLS; mean pooling |
| Objective | I-JEPA: EMA target encoder (μ cosine 0.996→1.0), Smooth-L1 (β=1) on predicted vs target tokens |
| Optimiser | AdamW, cosine LR, weight decay 0.05, batch 128 (other values **\[S-B CODE\]**) |
| Training length | Core grid 9,030 steps (105 epochs; 86 steps/epoch); pilot 26,058 steps (303 epochs) |
| Arms | `ijepa` (uniform), `veg_pos` (ExG-softmax, τ=0.5, ε=0.2), `veg_neg` (inverse polarity), `pos_matched` (static spatial prior reproducing `veg_pos` target locations, KL=0.0006) |
| Grid | 4 arms × 6 seeds (0–5) \+ 6 random-init \+ 6 supervised \+ 1 pilot \= 37 runs |
| Evaluation | Frozen target encoder → global-average features; logistic-regression probe (C grid tuned on val) and cosine k-NN (k grid); regimes 1/5/10/25/50/100-shot, 10%, 100%; 5 stratified label draws |
| Primary endpoint | Mean test macro-F1 over {10-shot, 25-shot, 10%} (probe) |
| Statistics | Unit \= pretraining seed; exact one-sided Wilcoxon, Holm family of 3, 2,000-sample hierarchical bootstrap, TOST ±1.5 pp |
| Gates | G-1 PASS, G0 PASS, G1 **Mixed / Inconclusive** |

### 2.3 Stage B results to carry forward (numbers Stage C must be compared against)

**Linear probe, test macro-F1 (%), mean over 6 seeds.**

| Arm | 10-shot | 25-shot | 10% | 100% | Primary endpoint |
| :---- | :---: | :---: | :---: | :---: | :---: |
| random\_init | 17.9 | 20.4 | 22.9 | 29.9 | 20.4 |
| veg\_neg | 35.0 | 41.4 | 48.2 | 55.8 | 41.5 |
| pos\_matched | 38.3 | 44.2 | 50.9 | 57.7 | 44.5 |
| ijepa | 38.5 | 44.3 | 51.1 | 58.1 | 44.6 |
| veg\_pos | 39.2 | 45.5 | 51.8 | 58.7 | 45.5 |
| pilot ijepa (303 ep, n=1) | 43.5 | 49.4 | 56.2 | 64.1 | 49.7 |
| supervised (from scratch) | — | — | 55.0 | 74.2 | — |

**k-NN primary endpoint:** random 14.6 · veg\_neg 36.5 · pos\_matched 39.3 · ijepa 39.4 · veg\_pos 39.6.

**Confirmatory tests (probe primary endpoint, n=6).**

| Hypothesis | Δ (pp) | 95% CI | Seeds \+ | Holm p | Status |
| :---- | :---: | :---: | :---: | :---: | :---- |
| H1 veg\_pos − ijepa | \+0.893 | \[−0.01, \+1.90\] | 5/6 | 0.0625 | Inconclusive |
| H2 veg\_pos − pos\_matched | \+1.027 | \[−0.01, \+2.00\] | 4/6 | 0.0781 | Inconclusive |
| H3 veg\_pos − veg\_neg | \+3.966 | \[+2.96, \+4.85\] | 6/6 | 0.0469 | **Confirmed** |
| pos\_matched − ijepa | −0.13 | \[−0.89, \+0.59\] | — | — | **Equivalent** (TOST ±1.5) |
| veg\_pos − ijepa (k-NN) | \+0.20 | \[−0.96, \+1.47\] | — | — | **Equivalent** (TOST ±1.5) |

**Seed-level paired differences, veg\_pos − ijepa (pp):** \+0.986, \+0.995, \+0.339, −0.311, \+0.579, \+2.774 (mean 0.893, SD ≈ 1.04).

**Manipulation check (training logs).** Target greenness z-score: ijepa \+0.028, veg\_pos \+0.424, veg\_neg −0.383, pos\_matched \+0.035. Context tokens: ijepa 28.22, veg\_pos 31.14, veg\_neg 30.57, pos\_matched 26.99 (target tokens 70.93 for all). KL(veg\_pos‖pos\_matched) \= 0.0006; KL(veg\_pos‖ijepa) \= 0.0485.

**Per-class effect, veg\_pos − ijepa at 100% labels (pp; exploratory, unadjusted):** Potatoes \+5.19\*, Sugar beet \+3.56, Soya \+2.45, Maize \+1.50\*, Durum wheat \+0.83, Oats \+0.62, Rye \+0.21, Common wheat −0.03, Sunflower −1.10, Rapeseed −1.52, Barley −1.68, Temporary grassland −1.80 (\* 95% CI excludes 0).

**Training diagnostics.** Loss 0.414 → 0.116–0.131; RankMe 46–52 (collapse floor 9.6); top singular value 8.5–9.1% of variance.

### 2.4 Threats to validity of Stage B that Stage C must close

| ID | Threat | Stage C response |
| :---- | :---- | :---- |
| T1 | n=6 seeds: min attainable Holm p \= 0.0469, any discordant seed fails | n=20 fresh seeds (power analysis, 7.7) |
| T2 | Undertrained (105 vs 303 epochs; pilot \+5.1 pp) — conclusions may not hold at convergence | Primary protocol \= 26,058 steps; length axis (100 / 303 / 606 epochs) |
| T3 | Context-token imbalance (31.1 vs 28.2 vs 27.0) | Fixed-K token clamp for all arms (4.7), bridge replication of Stage B protocol |
| T4 | One dataset, one split; same test images for all seeds (seed replication ≠ data replication) | Split B (new group split) \+ second dataset \+ meta-analysis |
| T5 | One small model (ViT-Tiny) | ViT-Small replication |
| T6 | One vegetation index, one (τ, ε) | Index variants (VARI, ExGR), guidance-strength sweep (validation-based, descriptive) |
| T7 | Why `veg_neg` is worse is ambiguous (uninformative targets? vegetation withheld from context?) — Stage B text over-interpreted | Target-information diagnostics (4.11); wording rules in 3.4 |
| T8 | Row-crop vs dense-canopy pattern found post hoc; per-class CIs unadjusted over 12 classes | Pre-registered moderation tests H5/H6 on fresh seeds; groups fixed by agronomy before Stage C results |
| T9 | "SSL beats supervised at 10%" rests on one seed, one arm | Supervised/fine-tune baselines with ≥ 10 seeds, identical recipe |
| T10 | Only frozen-feature evaluation | Add full fine-tuning at 10% and 100% labels |
| T11 | Dataset described as "aerial/field" although LUCAS photos are survey photographs taken on the ground — **\[USER TO VERIFY\]** | Describe accurately in the paper |
| T12 | Stage B test split has been seen | No Stage C design choice may use test results; Phase 0 uses validation only |
| T13 | Pre-registration was only self-timestamped | External registration (Section 8\) |

---

## 3\. Stage C objectives, hypotheses and decision logic

### 3.1 Design principle: three levels of replication

| Level | What changes | What stays | Phases |
| :---- | :---- | :---- | :---- |
| L1 — Pretraining randomness | 20 fresh seeds (100–119), full training length | data, split, model | Phase 1 |
| L2 — Data sample | new group-aware split (Split B), or new dataset | model recipe | Phases 3-S, 4 |
| L3 — Capacity / representation | ViT-Small; other vegetation indices | data, split | Phase 3-V, 3-I |

Stage B only had L1 with n=6. A "final" paper needs all three.

### 3.2 Hypothesis registry

Endpoints: **PE-probe** \= mean test macro-F1 of the logistic probe over {10-shot, 25-shot, 10%}; **PE-kNN** \= same with k-NN. Unit of analysis \= pretraining seed (label draws are averaged within seed first). SESOI (smallest effect size of interest) \= **\+0.5 pp** \[LOCKED\].

| ID | Family | Statement | Test | Seeds | Setting |
| :---- | :---- | :---- | :---- | :---: | :---- |
| H1 | P (Holm, m=3) | veg\_pos \> ijepa on PE-probe | exact paired Wilcoxon, one-sided | 20 | LUCAS split A, ViT-Tiny, 26,058 steps |
| H2 | P | veg\_pos \> pos\_matched on PE-probe | same | 20 | same |
| H3 | P | veg\_pos \> veg\_neg on PE-probe | same | 20 | same |
| E1 | E (Holm, m=2) | \|pos\_matched − ijepa\| \< 1.0 pp, PE-probe | TOST, 90% CI inside ±1.0 | 20 | same |
| E2 | E | \|veg\_pos − ijepa\| \< 1.5 pp, PE-kNN | TOST, 90% CI inside ±1.5 | 20 | same |
| H5 | S (Holm, m=2) | Benefit of veg\_pos over ijepa is larger on high-ExG-dispersion test images than on low-dispersion ones | paired Wilcoxon on stratum deltas, one-sided | 20 | same |
| H6 | S | Benefit is larger for row crops than dense canopies (groups fixed in 3.5) | paired Wilcoxon on group deltas, one-sided | 20 | same |
| H7a | M (Holm, m=3) | veg\_gate \> ijepa on PE-probe | exact Wilcoxon, one-sided | 12 | same |
| H7b | M | veg\_gate \> veg\_gate\_shuf (gating signal matters) | same | 12 | same |
| H7c | M | veg\_gate \> veg\_pos | same | 12 | same |
| R1 | R (reported jointly) | veg\_pos \> ijepa on Split B | exact Wilcoxon, one-sided | 10 | split B |
| R2 | R | veg\_pos \> ijepa on dataset 2 | same | 10 | dataset 2 |
| R3 | R | veg\_pos \> ijepa with ViT-Small | same | 10 | LUCAS split A |
| R4 | R | pooled effect across L1–L3 settings \> 0 | random-effects meta-analysis (DerSimonian–Laird), one-sided z | settings | all |
| X\* | X (exploratory) | everything else (index variants, sweeps, length axis, fine-tuning, per-class, features) | estimates \+ 95% CI, BH-FDR q-values, **no "significant" wording** | — | — |

Multiplicity: Holm is applied **within** each family only. Families P, E, S, M are tested at α \= 0.05 each (families are pre-specified and answer distinct questions). Family R is not pooled; it is reported as a set plus the meta-analysis R4.

### 3.3 Outcome decision matrix (fixed before data are seen)

**Primary claim (H1):**

| Code | Condition | Paper claim |
| :---- | :---- | :---- |
| P-A | Holm p \< 0.05 **and** 95% CI lower bound \> 0 **and** point estimate ≥ \+0.5 pp | "Vegetation-aware target selection improves I-JEPA representations" (main claim) |
| P-A′ | Holm p \< 0.05, CI \> 0, but point estimate \< \+0.5 pp | "Statistically reliable but small" |
| P-B | 90% CI ⊂ (−1.0, \+1.0) | "Greenness guidance is equivalent to uniform masking on average" (strong, publishable null) |
| P-C | neither of the above | "Inconclusive; effect bounded to \[lower, upper\]" — report the interval; no claim of benefit |
| P-D | Holm p \< 0.05 for ijepa \> veg\_pos | "Vegetation guidance is harmful" |

**Mechanism claims (independent of H1):** H3 and E1 replicate the Stage B polarity and spatial-prior findings. H5/H6 confirmed → "the benefit is conditional on scene heterogeneity", publishable even under P-B/P-C.

**Method claim:** H7a ∧ H7b confirmed (Holm) → "scene-adaptive guidance is a useful method"; H7c additionally confirmed → "better than fixed guidance". Otherwise report as a negative exploratory result.

**Generalisation:** R4 pooled estimate and the consistency table (sign and CI in each setting) are reported regardless of outcome.

### 3.4 Claims that must NOT be made (wording rules)

1. Never claim a benefit unless P-A / P-A′ holds. Never use "significant" for exploratory (X) results.  
2. Never claim "SSL beats supervised" unless the fine-tuning comparison (Phase 5\) shows it with CI \> 0 for the stated regime and the same recipe for both.  
3. Do not explain `veg_neg` underperformance as "models need vegetation targets" unless the diagnostics in 4.11 support it; present both explanations (uninformative targets vs vegetation withheld from context).  
4. Do not claim ExG is the best index unless the index sweep supports it; it is evaluated against VARI and ExGR only.  
5. Do not describe LUCAS photographs as aerial imagery (**\[USER TO VERIFY\]**).  
6. The row-crop/dense-canopy grouping was motivated by Stage B data; say so explicitly. Stage C tests it on fresh pretraining seeds, **not** on fresh images (the test images are the same in split A).

### 3.5 Pre-specified class groups for H6 \[LOCKED\]

Defined by agronomic canopy structure before Stage C is run:

* **Row-crop group R** (visible inter-row soil): B16 Maize, B21 Potatoes, B22 Sugar beet, B33 Soya, B31 Sunflower.  
* **Dense-canopy group D** (continuous cover): B11 Common wheat, B12 Durum wheat, B13 Barley, B14 Rye, B15 Oats, B55 Temporary grassland.  
* **Excluded from H6, reported separately:** B32 Rapeseed (ambiguous canopy).

Sunflower is deliberately in R although its Stage B delta was negative: the grouping is by agronomy, not by outcome. Durum wheat stays in D although its F1 is very low and noisy.

H6 statistic per seed: Δ\_R − Δ\_D, where Δ\_g \= mean over classes in g of \[per-class F1(veg\_pos) − per-class F1(ijepa)\], per-class F1 averaged over the three PE regimes and all label draws.

### 3.6 Image-level moderator for H5 \[LOCKED\]

* Per-patch ExG on the **evaluation view** (centre 160×160 crop) as defined in 4.6.  
* **ExG dispersion** D(x) \= standard deviation of the 100 patch ExG values of image x (raw ExG units, **not** z-scored).  
* Strata: tertiles of D computed on the **train** split (cut-points saved); test images assigned to Low / Mid / High.  
* Per seed: stratum accuracy for veg\_pos and ijepa from saved per-sample predictions at each PE regime (averaged over regimes and draws); Δ\_stratum \= acc(veg\_pos) − acc(ijepa); H5 statistic \= Δ\_High − Δ\_Low.

---

## 4\. Locked experimental design

### 4.1 Primary data: LUCASVision \[LOCKED\]

* Source: JRC LUCASVision; 15,876 readable images; filename regex `^LUCAS\d{4}_(\d+)_[A-Za-z0-9]+\.jpg$` (100% must match); 14,760 survey points; perceptual-hash near-duplicate clusters merged with survey-point IDs into 14,729 groups.  
* **Split A (primary) must reproduce Stage B exactly:** use saved Stage B indices if available; otherwise regenerate with the same hash rule and **assert** the counts below (abort if any differs).

| Code | Crop | Train | Val | Test |
| :---- | :---- | :---: | :---: | :---: |
| B11 | Common wheat | 1,464 | 319 | 313 |
| B12 | Durum wheat | 337 | 75 | 73 |
| B13 | Barley | 1,313 | 273 | 287 |
| B14 | Rye | 807 | 174 | 170 |
| B15 | Oats | 722 | 149 | 161 |
| B16 | Maize | 978 | 209 | 206 |
| B21 | Potatoes | 861 | 188 | 173 |
| B22 | Sugar beet | 748 | 156 | 163 |
| B31 | Sunflower | 419 | 84 | 93 |
| B32 | Rapeseed | 1,511 | 328 | 324 |
| B33 | Soya | 623 | 140 | 134 |
| B55 | Temporary grassland | 1,332 | 285 | 284 |
|  | **Total** | **11,115** | **2,380** | **2,381** |

* **Split B (robustness):** a second group-aware stratified split (≈70/15/15) using a different hash salt `SPLIT_B_SALT` \[LOCKED \= "stageC-splitB-v1"\]. Assert 0 leaking groups; log the fraction of Split-A test images that fall in Split-B test.  
* SSL pretraining uses **train images only** (labels never used). Val is used for probe C and k selection and all Phase-0 decisions. Test is used only for final reporting.  
* Pre-cache at 192×192 uint8 (≈1.76 GB total) and keep on GPU; do augmentation on GPU for throughput.

### 4.2 Second dataset (Phase 4\) \[USER TO CONFIRM\]

Purpose: show the finding is not specific to LUCAS. Requirements: RGB photographs of crops/vegetation with visible soil/background; ≥ 8 classes; ≥ 8,000 images; every class has ≥ 40 train images (needed for the 25-shot regime). Candidate datasets (verify availability, size and licence before committing): DeepWeeds, the Aarhus Plant Seedlings dataset, or any UAV/ground crop RGB set the user already holds.

The generator must implement a **dataset adapter** reading an `ImageFolder`\-style directory (`DATA2_ROOT/<class>/*.jpg`) with an optional CSV of group IDs. If no group IDs exist: cluster near-duplicates by perceptual hash and split by cluster; flag residual leakage risk in Limitations. The choice of dataset 2 must be recorded **before** any result on it is seen. If no suitable dataset is available, skip Phase 4 and state this limitation.

### 4.3 Pre-processing and augmentation \[S-B CODE\]

Copy from the Stage B source: 192×192 cache → training crop to 160×160 (type, scale range), flips, normalisation constants, evaluation view. These are identical for all arms and all datasets. ExG is always computed from the **same view that the model sees**.

### 4.4 Models \[LOCKED\]

|  | ViT-Tiny (primary) | ViT-Small (R3) |
| :---- | :---- | :---- |
| Patch / grid | 16 / 10×10 | same |
| Encoder | D=192, depth 12, heads 3, MLP ratio 4 (5.49 M) | D=384, depth 12, heads 6, MLP ratio 4 (\~22 M) |
| Predictor | D=128, depth 4, heads 4, MLP ratio 4 (0.84 M) | D=256, depth 4, heads 8, MLP ratio 4 |
| Positions | fixed 2D sin-cos; no CLS | same |
| Pooling for evaluation | mean over all 100 tokens of the **target (EMA) encoder** | same |

### 4.5 Configuration lock table

| Parameter | Value | Tag |
| :---- | :---- | :---- |
| Batch size | 128 | \[LOCKED\] |
| Optimiser | AdamW, weight decay 0.05, cosine decay | \[LOCKED\] |
| Peak LR, warm-up, betas, grad-clip | — | \[S-B CODE\] |
| EMA momentum | cosine 0.996 → 1.0 over total steps | \[LOCKED\] |
| Loss | Smooth-L1, β=1.0, mean over target tokens | \[LOCKED\] |
| Target feature normalisation | — | \[S-B CODE\] |
| Precision | bfloat16 autocast; fp32 master weights | \[LOCKED\] |
| Steps, primary | **26,058** (= 303 epochs × 86 steps, matches the Stage B pilot) | \[LOCKED\] |
| Steps, short / long (length axis) | 9,030 (Stage B) / 52,116 | \[LOCKED\] |
| Target blocks | 4 blocks; scale ∈ \[0.15, 0.20\]; aspect ∈ \[0.75, 1.33\] | \[LOCKED\] |
| Context block | scale ∈ \[0.85, 1.00\], target patches removed | \[LOCKED\] |
| veg\_pos temperature τ / exploration ε | 0.5 / 0.2 | \[LOCKED\] |
| Final checkpoint | last step only; no checkpoint selection, no early stopping | \[LOCKED\] |

### 4.6 Arm specifications

**Per-patch vegetation score** (for every arm that needs it): reuse the Stage B function (ExG \= 2G − R − B per patch on the model's view), giving a 10×10 map `s`. For a candidate block with top-left (r, c) and shape (h, w), block score \= mean of `s` over the block; scores are z-scored **within each image** across all valid candidate positions → `z(r,c)`.

| Arm | Target-position distribution over valid top-left positions | Tier |
| :---- | :---- | :---- |
| `ijepa` | uniform | core |
| `veg_pos` | P \= (1−ε)·softmax(z/τ) \+ ε/N\_valid | core |
| `veg_neg` | P \= (1−ε)·softmax(−z/τ) \+ ε/N\_valid | core |
| `pos_matched` | P\_prior(r,c \| h,w) \= mean over training crops of P\_veg\_pos(r,c \| x,h,w); content-independent | core |
| `veg_gate` | P \= g(x)·P\_veg\_pos \+ (1−g(x))·Uniform, with g(x) defined below | method |
| `veg_gate_shuf` | same as veg\_gate but g is taken from a **different random image of the same batch** (fresh permutation each step) | control |
| `veg_pos_vari` | veg\_pos with VARI \= (G−R)/(G+R−B+1e-6) as the patch score | robustness |
| `veg_pos_exgr` | veg\_pos with ExGR \= ExG − (1.4R − G) as the patch score | robustness |
| `veg_pos_t{τ}_e{ε}` | veg\_pos with the stated (τ, ε) | sweep |

**veg\_gate definition \[LOCKED\].** D(x) \= std over the 100 patch ExG values of the training view of x. D\_lo and D\_hi \= 25th and 75th percentiles of D over the **train** split (computed in Phase 0 on centre-cropped images, saved in `locked_profile_stageC.json`). g(x) \= clip((D(x) − D\_lo)/(D\_hi − D\_lo), 0, 1). Homogeneous canopies (low D) fall back to uniform masking; heterogeneous scenes (high D) get full vegetation guidance. No test data are used to set D\_lo/D\_hi.

`pos_matched` must be rebuilt with the same recipe as Stage B for each (dataset, split, protocol) combination; log KL(veg\_pos‖pos\_matched).

### 4.7 Token-budget protocol

Stage B equalised token counts only within each batch (minimum over the batch), leaving arm-level differences (context tokens 31.1 / 28.2 / 27.0).

* **Protocol C (primary, \[PHASE-0\] parameters).** Draw masks for 20,000 train views per arm (all arms in the run matrix). Let `n_ctx` and `n_tgt` be the context and target token counts. Set `K_CTX = floor(min over arms of the 1st percentile of n_ctx)` and `K_TGT` likewise. During training, for every image, **randomly subsample exactly K\_CTX context tokens and exactly K\_TGT target tokens** (seeded). The rare image with fewer tokens is re-drawn up to 10 times; a persistent shortfall must be \< 0.1% of draws (assert). All arms therefore have identical token counts.  
* **Protocol B (bridge, Stage B protocol).** Exactly the Stage B per-batch-minimum equalisation, reused verbatim from the Stage B source. Used only for the bridge replication (3-Br) and as the fallback below.  
* **Phase-0 adoption rule (validation data only).** Train `ijepa` with seed 900 for 9,030 steps under Protocol C and Protocol B. Adopt Protocol C as primary if its validation primary endpoint (k-NN or probe on val) is ≥ (Protocol B value − 3 pp) **and** RankMe ≥ 20 **and** no non-finite steps. Otherwise Protocol B is primary and Protocol C becomes the sensitivity analysis. The decision is written to `locked_profile_stageC.json` before Phase 1\.

### 4.8 Training protocol

* **Seeds \[LOCKED\].** Confirmatory seeds 100–119; Phase-0 seeds 900–909 (validation-only, never analysed as results). Seeds in sweeps use 100–105 (or the first N of 100–119).  
* `seed_everything(seed)`: python, numpy, torch, CUDA; separate generators for (a) weight init, (b) data order, (c) augmentation, (d) mask sampling, (e) token subsampling. The same seed gives the same init and data order across arms, so seeds are **paired** across arms.  
* Label-draw seed for evaluation \= deterministic hash of (pretraining seed, regime, draw index): identical across arms for the same pretraining seed, different across seeds.  
* Cosine LR over all steps, cosine EMA over all steps, BF16 autocast, gradient clipping per \[S-B CODE\]. `torch.compile` optional; if used, apply identically to all arms.  
* Checkpoint every 2,000 steps (`last.pt` with optimizer, scaler, RNG states) to allow exact resume; on completion save `encoder_final.pt` (context \+ target encoders) and delete `last.pt`.  
* Log every 50 steps: loss, LR, EMA μ, grad-norm, mean target z-score, context/target token counts, step time. Log RankMe and online val k-NN every 5 epochs.  
* **Failure rule:** a run with non-finite loss or crash is retried once from its last checkpoint. If it fails again, that **seed is excluded for all arms** in that phase (list-wise by seed) and documented; exclusions are never decided by results.

### 4.9 Evaluation protocol \[S-B CODE unless stated\]

* Frozen **target encoder**; global average pooling over all 100 patch tokens; evaluation view per 4.3.  
* **Linear probe:** L2 logistic regression, L-BFGS, `max_iter=2000`, C ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100, 1000} chosen on validation macro-F1; assert 0 convergence warnings. Feature standardisation per \[S-B CODE\].  
* **k-NN:** cosine, k ∈ {1, 3, 5, 10, 15, 20} (capped by class size), chosen on validation macro-F1.  
* **Regimes:** 1, 5, 10, 25, 50, 100-shot per class; 10% of train; 100% of train. **Label draws: 10** per (seed, regime) for regimes \< 100% **\[LOCKED; Stage B used 5\]**, stratified, drawn from the train split only.  
* Report test macro-F1 for every (arm, seed, regime, draw); save per-sample test predictions (int8) for every (regime, draw) in `preds.npz`, plus features for train/val/test as float16 (`features_{split}.npy`) so all analysis can be redone without the model.  
* Random-init baseline: an untrained encoder with the same seed, evaluated identically (20 seeds).  
* Datasets with different sizes (dataset 2): same regimes where class sizes allow; report which regimes were feasible.

### 4.10 Fine-tuning and supervised baselines (Phase 5\)

* Inits compared with **identical recipe**: `random_init` (= supervised from scratch), `ijepa`, `veg_pos` (`veg_gate` too if G2-C passes). Seeds 100–109. Label regimes: 10% and 100% (train split).  
* Recipe \[LOCKED unless the Stage B supervised baseline already defines one — then reuse it for all inits\]: AdamW, weight decay 0.05, cosine schedule, 5-epoch warm-up, label smoothing 0.1, random-resized-crop (scale 0.4–1) \+ horizontal flip, BF16, layer-wise LR decay 0.75, epochs \= 100 (100% labels) / 300 (10% labels), batch 128 (64 at 10%).  
* Learning rate chosen from {1e-4, 3e-4, 1e-3} on **validation** macro-F1 using seed 100 **separately per init type and regime**, then frozen for the other seeds.  
* Report test macro-F1 and accuracy; paired by seed.

### 4.11 Diagnostics (every pretraining run)

1. **Manipulation check:** mean target ExG z-score, mean context ExG z-score, mean all-patch ExG, context/target token counts, KL of target-centre distribution vs `veg_pos` and vs `pos_matched`.  
2. **Target information content (new, supports T7):** mean pixel standard deviation, mean Sobel gradient magnitude, and mean EMA-target-feature norm inside target regions.  
3. **Collapse metrics:** RankMe (collapse floor \= 0.05 × D), feature spectrum (`spectrum.npy`), top singular value share.  
4. **Mask audit** (`mask_audit.json`): per-arm histograms of target centres (10×10), block sizes, overlap fraction.  
5. **ExG dispersion D(x)** of every train/val/test image (saved once per dataset/split).

---

## 5\. Run matrix, priorities and compute budget

**Unit of compute:** 1 TRE (tiny-run-equivalent) \= one 26,058-step ViT-Tiny pretraining run plus its full evaluation. At Stage B's measured 3,020 img/s, pretraining ≈ 18 min and evaluation ≈ 5 min, so **1 TRE ≈ 0.4 GPU-hours**. This is an estimate from the Stage B throughput; **Phase 0 must re-measure it on the Stage C GPU** and the notebook must print an updated ETA.

Priority: **P0** must run · **P1** strongly recommended · **P2** run if time allows.

| Phase | ID | Content | Arms | Seeds | Steps | Runs | ≈TRE | Prio |
| :---- | :---- | :---- | :---- | :---: | :---: | :---: | :---: | :---: |
| 0 | — | Smoke, unit tests, throughput, power analysis, K\_CTX/K\_TGT, D\_lo/D\_hi, protocol adoption, **validation-only** | ijepa, veg\_pos | 900–902 | 9,030 | \~8 | \~3 | P0 |
| 1 | C1 | **Core grid** (family P, E, S) | ijepa, veg\_pos, pos\_matched, veg\_neg | 100–119 | 26,058 | 80 | 80 | P0 |
| 1 | B1 | Random-init baseline (eval only) | random\_init | 100–119 | 0 | 20 | 2 | P0 |
| 2 | M1 | **Method** (family M) | veg\_gate, veg\_gate\_shuf | 100–111 | 26,058 | 24 | 24 | P1 |
| 3 | S1 | **Split B** replication (R1) | ijepa, veg\_pos, pos\_matched | 100–109 | 26,058 | 30 | 30 | P1 |
| 4 | D2 | **Second dataset** (R2) | ijepa, veg\_pos, pos\_matched, veg\_gate | 100–109 | 26,058 | 40 | 40 | P1 |
| 5 | F1 | Fine-tuning \+ supervised | random\_init, ijepa, veg\_pos × {10%, 100%} | 100–109 | recipe 4.10 | 60 | 15 | P1 |
| 3 | V1 | **ViT-Small** (R3) | ijepa, veg\_pos (+veg\_gate if G2-C passes) | 100–109 | 26,058 | 20–30 | \~60–90 | P2 |
| 3 | I1 | Index variants | veg\_pos\_vari, veg\_pos\_exgr | 100–109 | 26,058 | 20 | 20 | P2 |
| 3 | T1 | Guidance-strength sweep: (τ,ε) ∈ {(0.25,0.2), (1.0,0.2), (2.0,0.2), (0.5,0.0), (0.5,0.5)} | veg\_pos variants | 100–105 | 26,058 | 30 | 30 | P2 |
| 3 | L1 | Length axis (303 epochs shared with C1) | ijepa, veg\_pos × {9,030, 52,116} | 100–105 | 9,030 / 52,116 | 24 | \~28 | P2 |
| 3 | Br | **Bridge**: Stage B protocol (Protocol B), 9,030 steps | ijepa, veg\_pos | 100–109 | 9,030 | 20 | 7 | P2 |

**Totals:** P0 ≈ 85 TRE (\~34 GPU-h) · P0+P1 ≈ 195 TRE (\~78 GPU-h) · everything ≈ 340–370 TRE (\~140 GPU-h). Running 2–3 workers concurrently on one 32 GB GPU (each ViT-Tiny run needs ≈ 1.3 GB) may cut wall-clock time, but speed-up is unverified: measure it in Phase 0 and pick the worker count that maximises total img/s.

**Ordering rule.** Seeds are scheduled in **blocks**: for seed s, all arms of a phase run before seed s+1. Any prefix of seeds is then a complete paired dataset.

**Compute-shortfall rule (pre-registered).** If compute runs out, analyse Phase 1 at the largest number of **complete** seeds, never fewer than 12, decided by compute availability alone and never by looking at results. Record the final n in the pre-registration addendum before unblinding.

**Sizing note.** Pretraining uses ≈ 11 k images × 303 epochs; ViT-Small cost is an estimate (\~3× per run); do not run V1 until P0 and P1 are finished.

---

## 6\. Gates and decision rules

| Gate | When | Checks | Fail action |
| :---- | :---- | :---- | :---- |
| **G-1C Data hygiene** | Phase 0 | Counts match 4.1 exactly; 0 leaking groups for split A and B; regex 100%; cache hash saved; spatial-variance share of ExG \< 10% | Stop; fix data |
| **G0-C Engineering** | end of Phase 0 | All Section 10 unit tests pass; manipulation check thresholds (veg\_pos z ≥ \+0.30, veg\_neg z ≤ −0.25, \|ijepa z\| ≤ 0.10, KL(veg\_pos‖pos\_matched) ≤ 0.005); RankMe ≥ 20; no non-finite steps; measured throughput recorded; resume test passes; protocol adopted (4.7); pre-registration frozen and hashed | Stop; fix code |
| **G1-C Primary** | after Phase 1 completes | Evaluate H1–H3, E1–E2, H5–H6 exactly as in Section 7; classify with the decision matrix 3.3 | none (outcome is the result) |
| **G2-C Method** | after Phase 2 | H7a–H7c (Holm). If H7a ∧ H7b confirmed → add `veg_gate` to V1 and Phase 5 | Skip `veg_gate` extensions |
| **G3-C Generalisation** | after S1, D2 (and V1) | R1–R3 estimates, R4 meta-analysis, consistency table | none |
| **G4-C Completeness** | before writing | Run flow chart complete (planned/finished/excluded); all tables/figures regenerate from saved artifacts; archive checksums verified; `DEVIATIONS.md` complete | Fix before writing |

Phase 0 decisions use **validation data only**. After G0-C passes and the pre-registration is frozen, no design choice may change.

---

## 7\. Statistical analysis plan (SAP)

### 7.1 Data preparation

* For each (run, regime) average test macro-F1 over label draws → seed-level value. Primary endpoint per seed \= mean of the three PE regimes.  
* Analysis tables are built from saved `eval.json`/`preds.npz`; all computations are reproducible from those files.  
* Arms are paired by seed. Seeds excluded for failure are removed for all arms (list-wise).

### 7.2 Confirmatory tests (families P, M, S, R)

* Test: **exact one-sided paired Wilcoxon signed-rank** on seed-level differences (`scipy.stats.wilcoxon(..., alternative="greater", method="exact")`; fall back to full sign-flip enumeration of signed ranks if zeros/ties block the exact routine). Directions are fixed in the registry.  
* Holm–Bonferroni within family; report raw and adjusted p.  
* Effect size: mean paired difference (pp) with a **seed-level percentile bootstrap 95% CI** (10,000 resamples, fixed RNG seed 20261102; resample seeds, then resample label draws within seed where draws are available).  
* **Confirmed Greater** requires Holm p \< 0.05 and CI lower bound \> 0; classification (P-A / P-A′ …) per 3.3.  
* **Sensitivity analyses (reported, not decisive):** paired t-test; exact sign-flip permutation test on means; linear mixed model `F1 ~ arm + (1 | seed)` (statsmodels `MixedLM`); leave-one-seed-out range of the estimate; analysis using only the first 12 seeds (shows stability).

### 7.3 Equivalence (family E)

TOST via 90% bootstrap CI within ±margin (E1 ±1.0 pp, E2 ±1.5 pp) **and** paired-t TOST p \< 0.05; Holm over m \= 2\. Also report the same comparisons at the Stage B margin ±1.5 pp for E1.

### 7.4 Moderation (family S)

* **H5:** per seed Δ\_High − Δ\_Low (3.6); one-sided exact Wilcoxon on 20 values.  
* **H6:** per seed Δ\_R − Δ\_D (3.5); one-sided exact Wilcoxon on 20 values.  
* Holm over m \= 2\.  
* Exploratory companion: Spearman correlation between D(x) and the per-image correctness difference, averaged over seeds, with a seed-level bootstrap CI.

### 7.5 Method family (M) and replications (R)

* H7a–c on seeds 100–111 (12 paired seeds); Holm over m \= 3\.  
* R1–R3: same test as H1 on their own seeds (n=10). **R4:** DerSimonian–Laird random-effects meta-analysis of per-setting mean paired differences (using per-setting SE from seeds), settings \= {LUCAS-A Tiny, LUCAS-B Tiny, dataset 2, LUCAS-A Small}; report pooled effect, 95% CI, I², one-sided z-test.  
* A **consistency table** lists, for each setting: n, Δ, 95% CI, sign agreement.

### 7.6 Exploratory family X

Estimates with 95% CIs and Benjamini–Hochberg q-values within each exploratory sub-table (index variants, τ/ε sweep, length axis, per-class, per-regime, fine-tuning). The language must be descriptive ("estimated difference"), never "significant". Guidance-strength sweeps may be used to describe sensitivity only; **no hyperparameter is selected from them for any confirmatory test**.

### 7.7 Power analysis (Phase 0 cell, saved as `stats/power_analysis.json`)

Simulate paired differences from a normal distribution with SD \= 1.04 pp (Stage B) and true effects {0.3, 0.5, 0.7, 0.9} pp; n ∈ {10, 12, 15, 20}; estimate power of the exact one-sided Wilcoxon at α \= 0.05/3 (conservative first Holm step) and α \= 0.05. Print the table. For orientation, a paired design with SD 1.04 and n=20 has high power for a true \+0.9 pp, moderate power for \+0.5 pp; this is why the registered n is 20\. The Stage B effect is probably over-estimated (small-sample winner's curse), so the \+0.5 pp SESOI is used for interpretation.

### 7.8 Reporting standards

* A run-flow table: planned / completed / excluded (with reasons) per phase.  
* Every confirmatory result shows: n, mean, SD, Δ, 95% CI, raw and Holm p, and the seed-by-seed paired differences (figure).  
* Report all pre-registered tests even if non-significant. Any analysis not in this registry is labelled **exploratory**.

---

## 8\. Pre-registration procedure

1. In Phase 0, generate `stats/preregistration_stageC.json` containing: version, creation timestamp (UTC), git commit hash (if any), SHA-256 of **this specification file**, the hypothesis registry (3.2), endpoint definitions, class groups (3.5), moderator definition (3.6), SESOI, margins, α, test names, seeds, steps, run matrix, decision matrix (3.3), exclusion rules (4.8), compute-shortfall rule (5), and the \[PHASE-0\] values (K\_CTX, K\_TGT, D\_lo, D\_hi, adopted protocol).  
2. Write `preregistration_stageC.sha256`. The notebook refuses to start Phase 1 if the hash file is missing, and refuses to run the statistics if the registry file's hash differs from the stored one.  
3. **Human step:** upload this specification and the JSON to OSF (or AsPredicted) **before** launching Phase 1 and keep the registration link; cite it in the paper.  
4. Any later change is an **amendment** recorded in `DEVIATIONS.md` with date and reason; amendments after unblinding must be labelled as such in the paper.  
5. Phase 0 outputs are saved in `locked_profile_stageC.json`: GPU model, CUDA/PyTorch versions, measured img/s and peak VRAM per model size, K\_CTX, K\_TGT, D\_lo, D\_hi, adopted protocol, worker count, ETA table.

---

## 9\. Software architecture and notebook blueprint

### 9.1 Design

* One Python module `crop_jepa_stageC.py` (written by the notebook with `%%writefile`) holds all logic; the notebook is a thin controller that **launches worker subprocesses**: `python crop_jepa_stageC.py --phase C1 --worker k --n-workers N`.  
* Each worker reads `manifest/run_manifest.csv`, claims the next `pending` run (file-lock or atomic rename of a `.claim` file), trains, evaluates, and marks it `done` or `failed`. Several workers can share one GPU; several jobs can share one manifest.  
* Environment variables (prefix `CJC_`): `CJC_DATA_ROOT`, `CJC_DATA2_ROOT`, `CJC_OUT`, `CJC_WORKERS`, `CJC_DEVICE`, `CJC_COMPILE` (0/1).

### 9.2 Directory layout

outputs\_stageC/

  env\_lock.json                     \# pip freeze, CUDA, torch, GPU, git hash, spec SHA-256

  locked\_profile\_stageC.json        \# Phase-0 decisions (K\_CTX, K\_TGT, D\_lo, D\_hi, protocol, throughput)

  DEVIATIONS.md

  manifest/run\_manifest.csv         \# run\_id, phase, arm, seed, steps, protocol, dataset, split, model, status, started, finished

  data/                             \# split\_A.json, split\_B.json, dataset2\_split.json, classes.json, cache\_hashes.json, exg\_dispersion\_{dataset}.npy

  stats/                            \# preregistration\_stageC.json (+ .sha256), power\_analysis.json, gate\_\*.json, results\_\*.csv

  runs/\<phase\>/\<run\_id\>/            \# one directory per run

     config.json  train\_log.csv  train\_done.json  eval.json  preds.npz

     features\_train.npy features\_val.npy features\_test.npy   \# float16

     encoder\_final.pt  spectrum.npy  target\_centres.npy  mask\_audit.json  diagnostics.json

  paper\_tables/  figures/  logs/  report/FINAL\_REPORT.md

`run_id = {phase}_{arm}_{model}_{dataset}{split}_{steps}_{protocol}_s{seed}` (e.g. `C1_veg_pos_tiny_lucasA_26058_C_s103`).

### 9.3 Required functions (exact names and signatures)

| Module area | Functions / classes |
| :---- | :---- |
| Utilities | `seed_everything(seed) -> dict[str, torch.Generator]`, `atomic_write_json(path, obj)`, `sha256_file(path)`, `get_logger(name, logfile)`, `load_config(run_id) -> RunConfig` |
| Data | `discover_lucas(root) -> DataFrame`, `audit_dataset(df) -> dict`, `make_group_split(df, salt, ratios=(0.7,0.15,0.15)) -> dict`, `load_or_make_split(name) -> dict`, `build_cache(df, size=192) -> np.ndarray(uint8)`, `Dataset2Adapter(root, group_csv=None)` |
| Indices | `patch_scores(x_float, index="exg", patch=16) -> Tensor[B,10,10]` (index ∈ {exg, vari, exgr}), `exg_dispersion(x_float) -> Tensor[B]`, `gate_value(D, d_lo, d_hi) -> Tensor[B]` |
| Model | `ViTEncoder(embed_dim, depth, heads)`, `Predictor(enc_dim, pred_dim, depth, heads)`, `sincos_pos_embed(dim, grid)`, `ema_update(target, online, mu)` |
| Masks | `MaskGenerator(arm, cfg, prior=None)` with `__call__(scores, gen) -> (ctx_idx, tgt_idx, meta)`; `build_position_prior(train_views, cfg) -> dict`; `clamp_tokens(ctx_idx, tgt_idx, k_ctx, k_tgt, gen)`; `mask_audit(arm, n=20000) -> dict` |
| Train | `train_run(run_cfg) -> None` (resumable), `compute_rankme(feats) -> float`, `online_knn_val(model, split) -> float` |
| Eval | `extract_features(encoder, images) -> Tensor`, `probe_eval(feats, labels, regimes, draws, seed) -> dict`, `knn_eval(...) -> dict`, `evaluate_run(run_dir) -> dict`, `random_init_eval(seed) -> dict` |
| Fine-tune | `finetune_run(init, regime, seed, lr) -> dict` |
| Orchestration | `build_manifest(phase_list) -> DataFrame`, `claim_next_run(phase, worker) -> Optional[RunConfig]`, `run_worker(phase, worker, n_workers)`, `phase0_report() -> dict` |
| Stats | `seed_table(phase, endpoint) -> DataFrame`, `wilcoxon_exact_greater(d)`, `holm(pvals)`, `bootstrap_ci(d, B=10000, seed)`, `tost(d, margin)`, `moderation_deltas(...)`, `meta_random_effects(deltas, ses)`, `power_sim(...)`, `build_all_tables()`, `build_all_figures()` |
| Report | `write_final_report() -> Path` (fills the template in Section 11 from saved artifacts only) |

### 9.4 `eval.json` schema (per run)

{

  "run\_id": "...", "arm": "veg\_pos", "seed": 103, "steps": 26058,

  "probe":  {"1shot":{"mean":..,"draws":\[..\]}, "5shot":{...}, "10shot":{...}, "25shot":{...},

             "50shot":{...}, "100shot":{...}, "10pct":{...}, "100pct":{"mean":..}},

  "knn":    {"...same keys..."},

  "primary\_probe": 45.1, "primary\_knn": 39.7,

  "selected\_C": {"10shot":\[..draws..\]}, "selected\_k": {"...": \[..\]},

  "convergence\_warnings": 0, "rankme": 50.3, "top\_sv\_share": 0.088,

  "diagnostics": {"tgt\_exg\_z": 0.43, "ctx\_exg\_z": \-0.12, "ctx\_tokens": 24, "tgt\_tokens": 60,

                  "tgt\_pixel\_std": .., "tgt\_gradmag": .., "tgt\_feat\_norm": ..,

                  "kl\_vs\_veg\_pos": .., "kl\_vs\_pos\_matched": ..},

  "per\_class\_f1": {"B11": {"10shot": \[..draws..\], "25shot": \[..\], "10pct": \[..\]}, "...": {}}

}

### 9.5 Resume and robustness logic

* `train_run`: if `train_done.json` exists → return; if `last.pt` exists → restore model, EMA, optimizer, scaler, step and all RNG states; otherwise start fresh.  
* A worker crash leaves `status=running`; on restart, runs `running` for more than a configurable timeout are reset to `pending`.  
* After each phase, `verify_phase(phase)` asserts every expected run has all artifact files and that `eval.json` passes schema validation.

### 9.6 Notebook cell map

| Cell block | Content |
| :---- | :---- |
| 0 | Title, version, link to this spec, spec SHA-256 |
| 1 | Config (`CJC_*` vars), imports, env lock, GPU/throughput probe |
| 2 | Write `crop_jepa_stageC.py` (`%%writefile`), import it |
| 3 | Data discovery, audit, splits A and B, cache; assert Section 4.1 counts; Gate G-1C |
| 4 | Dataset-2 adapter and audit (skip with message if absent) |
| 5 | Unit tests (Section 10); Gate G0-C part 1 |
| 6 | Phase 0: throughput, mask audits, K\_CTX/K\_TGT, D\_lo/D\_hi, protocol adoption (validation only), power analysis |
| 7 | Freeze pre-registration; print hashes; **stop and wait for the human to register externally** |
| 8 | Build manifest for all phases |
| 9 | Launch workers for Phase 1 (subprocess, nohup / sbatch); progress dashboard (done/pending/failed, ETA) |
| 10 | Phase 1 verification, Gate G1-C statistics, tables and figures |
| 11–14 | Phases 2, 3, 4, 5 launch \+ verification (each guarded by a boolean) |
| 15 | Meta-analysis and consistency table (G3-C) |
| 16 | Final report generation (G4-C), archive \+ checksums |

### 9.7 SLURM template (`run_stageC.sh`)

\#\!/bin/bash

\#SBATCH \--job-name=cjC

\#SBATCH \--gres=gpu:1

\#SBATCH \--cpus-per-task=8

\#SBATCH \--mem=48G

\#SBATCH \--time=23:00:00            \# set to the cluster limit; jobs are resumable

\#SBATCH \--output=logs/slurm\_%j.out

source \~/.bashrc && conda activate cropjepa

export CJC\_OUT=outputs\_stageC CJC\_DATA\_ROOT=/path/to/LUCASvision CJC\_WORKERS=3

for k in \$(seq 0 \$((CJC\_WORKERS-1))); do

  python crop\_jepa\_stageC.py \--phase \${PHASE:-C1} \--worker \$k \--n-workers \$CJC\_WORKERS &

done

wait

Resubmit the same script until the manifest has no `pending` runs.

---

## 10\. Unit tests and integrity checks (run before any training)

| \# | Test | Pass criterion |
| :---- | :---- | :---- |
| U1 | Data counts per class/split | Equal to table 4.1 |
| U2 | Group leakage | 0 groups shared between splits (A and B) |
| U3 | Cache integrity | SHA-256 stored; shape (N,192,192,3), dtype uint8 |
| U4 | Patch scores | `patch_scores` output shape (B,10,10); ExG of a synthetic all-green image \> ExG of a grey image; VARI/ExGR finite |
| U5 | Mask disjointness | Context ∩ target \= ∅ for 20,000 samples per arm |
| U6 | Block geometry | Target block areas and aspect ratios within the specified ranges |
| U7 | Polarity | Mean target z: veg\_pos ≥ \+0.30; veg\_neg ≤ −0.25; \|ijepa\| ≤ 0.10 |
| U8 | Spatial control | KL(veg\_pos ‖ pos\_matched) ≤ 0.005; KL(veg\_pos ‖ ijepa) ≥ 0.02 |
| U9 | Token clamp | After clamping, context and target counts identical across all arms (zero variance) |
| U10 | Gate | g(x) ∈ \[0,1\]; g \= 0 for D ≤ D\_lo; g \= 1 for D ≥ D\_hi; `veg_gate` target-greenness z lies between `ijepa` and `veg_pos`; `veg_gate_shuf` g-distribution equals `veg_gate` g-distribution but correlation(g, D) ≈ 0 |
| U11 | Determinism | Same seed and arm → identical first 5 batches of masks; different seeds → different |
| U12 | Pairing | Same seed gives identical initial weights across arms |
| U13 | EMA | Target params change after steps; target has no gradient; μ schedule starts at 0.996 and ends at 1.0 |
| U14 | Loss | Finite on a smoke batch; decreases over 200 steps on a fixed batch |
| U15 | Resume | Train 200 steps; kill; resume; final state matches an uninterrupted run (within float tolerance) |
| U16 | Eval | Label draws stratified; draw sets identical across arms for the same pretraining seed; probe 0 convergence warnings on a smoke run |
| U17 | Splits in eval | Test indices never used in probe fitting or C/k selection (assert by index sets) |
| U18 | Stats | Exact Wilcoxon matches `scipy` on a toy example; Holm matches a hand-computed example; bootstrap CI covers a known mean in a simulation |
| U19 | Throughput | Measured img/s recorded; peak VRAM recorded |

Any failure aborts the notebook with a clear message.

---

## 11\. Deliverables: tables, figures and the auto-generated report

### 11.1 Paper tables (CSV \+ LaTeX)

| ID | Content |
| :---- | :---- |
| T1 | Setup summary (data, model, arms, steps, seeds) |
| T2 | Main results: probe and k-NN, all regimes, mean ± SD over seeds, all arms |
| T3 | Confirmatory results (families P, E, S, M): n, Δ, CI, raw p, Holm p, classification |
| T4 | Moderation: stratum deltas (Low/Mid/High) and group deltas (R/D/Rapeseed) with CIs |
| T5 | Method arms (`veg_gate`, `veg_gate_shuf`) vs baselines |
| T6 | Replications (R1–R3) \+ meta-analysis R4 \+ consistency table |
| T7 | Fine-tuning vs supervised (10%, 100%) |
| T8 | Diagnostics: manipulation check, token budget, target-information content, RankMe |
| T9 | Exploratory sweeps: index, (τ, ε), length axis, bridge |
| T10 | Run-flow table (planned / done / excluded) and compute cost |

### 11.2 Figures (PNG 300 dpi \+ vector PDF)

F1 design schematic · F2 manipulation check (ExG distributions, centre heat-maps, example masks per arm) · F3 forest plot of primary and equivalence comparisons with margins shaded · F4 seed-paired lines (each seed's arm values connected) · F5 label-efficiency curves · F6 moderation: Δ vs ExG-dispersion tertile · F7 per-class Δ grouped by R/D · F8 method arms · F9 replication forest \+ pooled effect · F10 (τ, ε) sweep · F11 length axis · F12 fine-tuning curves · F13 training curves \+ RankMe.

### 11.3 `FINAL_REPORT.md` template (filled automatically from saved artifacts)

1. Run-flow and integrity audit · 2\. Gates G-1C … G4-C outcomes · 3\. Primary result and decision-matrix classification · 4\. Equivalence results · 5\. Moderation (H5, H6) · 6\. Method family (H7) · 7\. Replications and meta-analysis · 8\. Fine-tuning · 9\. Exploratory results (clearly labelled) · 10\. Deviations · 11\. Auto-generated "Allowed claims / forbidden claims" list based on which hypotheses were confirmed.

---

## 12\. Paper plan

### 12.1 Title options (choose by outcome, after G1-C)

* P-A: *"Does Greenness Guide Latent Prediction? Vegetation-Aware Target Selection Improves Self-Supervised Learning on Crop Photographs"*  
* P-B / P-C: *"Does Greenness Guide Latent Prediction? A Pre-Registered Study of Vegetation-Aware Masking in I-JEPA"*  
* If H5/H6/H7 confirmed: *"When Does Vegetation-Aware Masking Help? Scene Heterogeneity and Adaptive Target Selection in Agricultural JEPA"*

### 12.2 Structure

1. Introduction (annotation cost in agriculture; JEPA; research question; contributions list tied to the hypothesis registry).  
2. Related work (masked image modelling / JEPA; guided and semantic masking; vegetation indices in remote sensing and agri-vision; pre-registration and equivalence testing in ML). **The novelty check against recent literature on guided masking must be completed before submission.**  
3. Method (arms, controls, `veg_gate`, token-budget protocol).  
4. Experimental protocol (data, splits, three replication levels, pre-registration link, SAP).  
5. Results in the order: manipulation check → polarity (H3) → spatial prior (E1) → primary (H1, H2) → moderation (H5, H6) → method (H7) → generalisation (R1–R4) → fine-tuning → exploratory.  
6. Discussion (interpretation per outcome; the two explanations for `veg_neg`; limitations).  
7. Conclusion (only claims permitted by 3.4).

### 12.3 Venue strategy

* Strong positive results across settings: *Computers and Electronics in Agriculture*, *IEEE TGRS / JSTARS*\-type journals.  
* Null or mixed with strong controls: CVPR/ICCV/NeurIPS agriculture or earth-vision workshops, TMLR-style venues that value rigorous negative results.  
* Release: code, split files, pre-registration, per-seed result tables, and `features_*.npy` so reviewers can recompute every statistic.

---

## 13\. Risk register

| Risk | Likelihood | Mitigation |
| :---- | :---: | :---- |
| Protocol C (fixed token clamp) degrades training | Medium | Phase-0 adoption rule (4.7); fallback to Protocol B with C as sensitivity |
| Effect vanishes at 303 epochs | Medium | Pre-registered decision matrix covers P-B/P-C; H5/H6/H7 still informative |
| Generated code contains bugs that affect arms asymmetrically | Medium | Unit tests U5–U13; manipulation checks as gate; `mask_audit.json` per run |
| Cluster pre-emption / wall-time limits | High | Resumable runs, block-ordered seeds, SLURM resubmission |
| Dataset 2 unavailable or leaky | Medium | Adapter \+ perceptual-hash grouping; skip Phase 4 and declare limitation |
| GPU non-determinism makes reruns differ slightly | Certain (small) | Record in `env_lock.json`; analysis relies on saved artifacts, not reruns |
| Test-set contamination through design choices | Low if rules followed | Rules 0.4.7, Section 6; Phase 0 val-only |
| Seed failures | Low | Failure rule 4.8 (list-wise exclusion by seed) |
| Reviewer: "tiny model, tiny data" | High | ViT-S replication, second dataset, meta-analysis |
| Reviewer: "guided masking is known" | Medium | Position the contribution as controls (polarity, spatial prior), pre-registered conditional benefit, and an adaptive rule; complete the literature check |

---

## 14\. Definition of done for Stage C

- [ ] All unit tests passed and logged; G-1C and G0-C passed.  
- [ ] Pre-registration frozen, hashed, **externally registered**; link stored.  
- [ ] Phase 1 complete for n \= 20 seeds (or the pre-registered shortfall n ≥ 12\) with all artifacts verified.  
- [ ] G1-C analysis executed exactly per Section 7; outcome classified per 3.3.  
- [ ] Phases 2, 4, 5 and S1 complete (P1), or their absence declared in Limitations.  
- [ ] Tables T1–T10 and figures F1–F13 generated from saved artifacts only.  
- [ ] `DEVIATIONS.md` complete; run-flow table complete.  
- [ ] Code, splits, registry, per-seed CSVs and archive checksums packaged.  
- [ ] Literature novelty check completed; claims list checked against 3.4.  
- [ ] **No further experiments are run after unblinding.**

### Deviation log template (`DEVIATIONS.md`)

\#\# YYYY-MM-DD — \<short title\>

\- Section affected:

\- What changed:

\- Why:

\- Made before / after seeing results (circle one):

\- Effect on confirmatory analyses:

---

## Appendix A. Reference values from Stage B (for sanity checks in Phase 0 and for the paper)

**k-NN test macro-F1 (%), mean over 6 seeds.**

| Arm | 10-shot | 25-shot | 10% | 100% | Primary |
| :---- | :---: | :---: | :---: | :---: | :---: |
| random\_init | 13.4 | 14.3 | 16.0 | 19.8 | 14.6 |
| veg\_neg | 31.3 | 35.5 | 42.8 | 50.4 | 36.5 |
| pos\_matched | 34.3 | 38.1 | 45.5 | 52.4 | 39.3 |
| ijepa | 34.6 | 38.1 | 45.5 | 52.6 | 39.4 |
| veg\_pos | 34.4 | 38.6 | 45.8 | 53.7 | 39.6 |
| pilot ijepa (303 ep) | 38.6 | 42.4 | 48.1 | 56.6 | 43.0 |

**Regime-level probe deltas, veg\_pos − ijepa (pp, 95% CI):** 1-shot \+0.02 \[−1.03, \+1.03\]; 5-shot \+0.47 \[−0.60, \+1.56\]; 10-shot \+0.76 \[−0.29, \+1.98\]; 25-shot \+1.20 \[+0.13, \+2.35\]; 50-shot \+0.64 \[−0.57, \+1.80\]; 100-shot \+1.35 \[+0.15, \+2.60\]; 10% \+0.72 \[−0.35, \+1.89\]; 100% \+0.68 \[−1.16, \+2.65\].

**veg\_neg − ijepa (pp):** 1-shot −1.93, 5-shot −1.41, 10-shot −3.46, 25-shot −2.88, 50-shot −3.21, 100-shot −2.98, 10% −2.88, 100% −2.23 (all CIs exclude 0).

**Seed-level primary endpoint values (probe, %):**

| Seed | ijepa | pos\_matched | veg\_neg | veg\_pos |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 44.587 | 44.558 | 41.916 | 45.573 |
| 1 | 44.595 | 43.842 | 41.038 | 45.589 |
| 2 | 45.601 | 44.256 | 41.802 | 45.940 |
| 3 | 44.688 | 44.736 | 42.013 | 44.377 |
| 4 | 43.904 | 44.923 | 40.479 | 44.483 |
| 5 | 44.249 | 44.506 | 41.942 | 47.023 |

**Per-class probe F1 at 100% labels (%):**

| Class | random\_init | ijepa | pos\_matched | veg\_pos |
| :---- | :---: | :---: | :---: | :---: |
| B21 Potatoes | 25.34 | 66.17 | 64.54 | 69.96 |
| B22 Sugar beet | 34.36 | 65.93 | 66.22 | 68.45 |
| B33 Soya | 29.68 | 64.94 | 64.01 | 66.20 |
| B16 Maize | 45.95 | 82.46 | 82.23 | 83.05 |
| B12 Durum wheat | 0.37 | 7.64 | 6.18 | 7.55 |
| B15 Oats | 6.76 | 38.03 | 35.12 | 37.14 |
| B14 Rye | 21.36 | 38.50 | 35.09 | 38.46 |
| B11 Common wheat | 37.74 | 59.39 | 58.00 | 58.93 |
| B31 Sunflower | 58.98 | 86.11 | 85.97 | 84.58 |
| B32 Rapeseed | 32.38 | 79.69 | 79.00 | 77.18 |
| B13 Barley | 29.23 | 49.48 | 49.11 | 47.40 |
| B55 Temporary grassland | 36.78 | 68.81 | 67.11 | 66.03 |

**Dataset audit (Stage B):** 15,876 images (0 corrupt); 14,760 survey points; 1,053 points seen in more than one survey year, 851 with label rotation; 16 perceptual near-duplicate clusters (47 images; 8 with conflicting labels); 14,729 final groups; spatial position explains 1.1% of raw ExG variance (3.9% of z-scored).

**Stage B gate values:** G0 pilot: \+36.75 pp k-NN gain over random init, RankMe 83.7 vs floor 9.6. Pilot (303 epochs) primary endpoint 49.69%; 10%-label probe 56.19% vs supervised-from-scratch 55.03% (one seed).

## Appendix B. Glossary

**I-JEPA** — Image Joint-Embedding Predictive Architecture: predicts target-block representations (from an EMA encoder) from a context block. **ExG** — Excess Green index, 2G − R − B. **VARI** — (G − R)/(G \+ R − B). **ExGR** — ExG − (1.4R − G). **Probe** — linear classifier on frozen features. **TOST** — two one-sided tests for equivalence. **SESOI** — smallest effect size of interest. **Holm** — step-down multiplicity correction. **RankMe** — effective rank of representations (collapse indicator). **Pre-registration** — time-stamped, hashed statement of hypotheses and analysis before data are seen.

## Appendix C. One-paragraph brief for the generator (paste first)

"Build the Stage C notebook for Crop-JEPA exactly as specified in STAGE\_C\_MASTER\_SPEC.md, one part at a time (Section 0.3). Reuse the Stage B source for every \[S-B CODE\] value and for the four Stage B mask arms. Do not invent hyperparameters; list any missing input under MISSING INPUTS. Implement all functions in 9.3, all unit tests in Section 10, the resumable subprocess orchestrator in 9.1/9.5, the statistics in Section 7, and the outputs in Sections 9.2 and 11\. Never use test data for any decision. Return complete, runnable code with no placeholders."