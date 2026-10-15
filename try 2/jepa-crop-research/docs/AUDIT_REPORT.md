# Research audit and decision report

**Scope:** critical audit of `JEPA_Crop_Classification_Research_Handover.md` (v1.0, 9 Oct 2026) and the frozen design implemented in `jepa_crop_research.ipynb` v1.0.0.

**Evidence labels used throughout**

* **[R]** — my recollection of the literature or datasets from training data (reliable to roughly mid-2026). **No web or paper access was available in this session**, so nothing here was verified against primary sources during this task. Citations may be inaccurate; verify each one.
* **[H]** — stated in the handover.
* **[N]** — checked programmatically by the notebook at runtime on your data.
* **[Rec]** — my recommendation or hypothesis.

---

## 1. What should remain unchanged

* **Research philosophy [H]:** one modification at a time, decision gates, falsifiability, and "a negative result is acceptable".
* **Separate budgets [H]:** self-supervised pretraining data and labelled downstream data are tracked separately; label fractions are tested explicitly.
* **Strict inductive protocol [H]:** no validation or test imagery in SSL pretraining. Normalisation uses training data only.
* **Reporting [H]:** metrics must match claims; macro-F1 and per-class metrics are required alongside overall accuracy; multiple seeds; uncertainty reporting; results are saved for every run.
* **Novelty warnings [H]:** "JEPA on crops" alone is not novel, and "phenology" needs an operational definition. Both are correct and are kept.
* **PASTIS facts [H/R]:** 2,433 patches of 128×128 px, 10 S2 bands, 38–61 dates, 18 crop classes plus background and void, official 5-fold split. These are consistent with my recollection and are checked at runtime [N].

## 2. What should be corrected

| # | Handover statement | Problem | Correction |
|---|---|---|---|
| C1 | Diagram (§3.3): "TARGET REGION → target encoder" | **Implementation-critical.** In canonical I-JEPA [R] the target encoder sees the **whole input**. Target representations are obtained by selecting target positions from its *output*, then applying LayerNorm. Encoding only the target region gives targets without context and a different method | Implemented as in I-JEPA; verified by a test showing that target representations depend on target inputs, while predictions depend only on context [N] |
| C2 | §3.3 conceptual objective | Missing details that change behaviour [R] | Added all of them; see the list below the table |
| C3 | §3.4 "two encoders avoid collapse" | Simplified. Collapse avoidance in I-JEPA-style training relies on the stop-gradient/EMA asymmetry *and* the predictor, and it is empirical rather than guaranteed. A low SSL loss can itself be a symptom of collapse | Per-epoch effective rank, feature std and pairwise cosine on pooled target embeddings, with an automatic stop after 3 collapsed epochs. Also a random-initialisation linear-probe control (G4) |
| C4 | §5.1 "2,433 multitemporal Sentinel-2 *tiles*" | They are *patches* cut from 4 S2 *tiles*. The distinction matters for geographic hold-out | Terminology fixed; a `tile_holdout` split mode is available |
| C5 | §5.3 pilot with "4–8" or "8–12 selected dates" | **Conflicts with the hypothesis.** Pre-selecting dates is itself a temporal target or context selection, and confounds the comparison. It also discards the phenological transitions under study | All valid acquisitions are kept (≤ 61). This is affordable for parcel series |
| C6 | §9.3 / §9.5: batch 1–2, 64×64 crops, 10–30 h per JEPA run, 80–200 h total | Valid for dense segmentation, not for the parcel task chosen here | Parcel-task estimates are in README §6; the notebook measures them on your machine |
| C7 | §9.5 "cache to disk rather than load the dataset into RAM" | For the parcel cache (≈ 0.1–0.5 GB) the opposite is better | The cache is held in RAM, and on the GPU if it is under 800 MB |
| C8 | SSL checkpoint selection not specified | Choosing a pretraining epoch by downstream validation accuracy invites optimistic bias | Final epoch of a fixed schedule (pre-declared). kNN-on-validation is logged for monitoring only |
| C9 | §8.1 / Part 6: AnySat frozen features as an early baseline | My recollection [R] is that AnySat's pretraining collection includes PASTIS data, possibly including test folds. That would make the comparison transductive or leaky. It also needs AnySat's code and checkpoint | Deferred. Verify AnySat's pretraining data before using it; if it overlaps, report the comparison separately as transductive |
| C10 | T-JEPA treated as one known method | The acronym is ambiguous [R]: there is a tabular-data T-JEPA (ICLR 2025) and a trajectory T-JEPA. The cited arXiv 2610.05731 (5 Oct 2026) postdates my knowledge entirely | Verify the exact paper, method and code before positioning against it |

**C2 detail — canonical I-JEPA elements the handover's objective omits [R]:**

* a narrow ViT predictor that takes context tokens plus *learned mask tokens carrying target positional embeddings*, run once per target block;
* EMA momentum scheduled from 0.996 to 1.0;
* multi-block masking: 4 target blocks of scale 0.15–0.2, and one context block of scale 0.85–1.0 with the targets removed;
* the loss: the paper describes an L2 distance, but my recollection is that the official code uses smooth-L1 (verify);
* evaluation with the average-pooled **target** encoder;
* a cosine schedule on weight decay.

## 3. What should be removed (from phase 1)

* **Dense segmentation, PASTIS-R, SICKLE, EuroCropsML, AnySat features.** The handover's own rule is to change one thing at a time; running several large datasets in parallel would dilute a 4 GB-GPU budget without answering the question.
* **The "date-stratified or regular target schedule" control (§6.4).** It is not needed, because the weighted-start framework already provides the uniform baseline. It can be added later as another weight function.
* **"Hyperspectral, diffusion and cross-modal JEPA" reading directions (§4.3)** as near-term work. They are tangential to temporal target selection.

## 4. What should be added

1. **Overlooked prior work: informed and guided masking in masked image modelling [R].**
   * This is the *main* novelty threat. "Choose prediction targets by a saliency signal" is an established idea:
     * AttMask — Kakogeorgiou et al., ECCV 2022;
     * ADIOS — adversarial masking, Shi et al., ICML 2022;
     * SemMAE — Li et al., NeurIPS 2022;
     * hard-patch mining — Wang et al., CVPR 2023.
   * A phenology-aware policy is a *domain-specific instance* of this idea. The defensible question is therefore whether a **phenological** saliency beats a **generic** one, not whether informed masking helps at all.
2. **Overlooked prior work: remote-sensing time-series SSL [R].**
   * Presto (Tseng et al., 2023): pixel time series with structured, including temporal, masking — the closest SSL analogue.
   * SatMAE (Cong et al., NeurIPS 2022): temporal and spectral masking.
   * U-BARN (Dumeur et al., 2024): SSL on S2 image time series, evaluated on PASTIS.
   * SITS-BERT / SITS-Former (Yuan et al.).
   * Galileo (Tseng et al., 2025): latent and pixel targets with space–time masking.
   * V-JEPA: spatio-temporal tube masking.
   * SeCo: seasonal contrast.
   * Supervised SITS baselines: L-TAE, PSE-TAE, U-TAE (Sainte Fare Garnot & Landrieu), TSViT (Tarasiou et al., CVPR 2023).
3. **Controls that make the experiment interpretable [Rec]:**
   * **Matched masking budget.** Blocks are non-overlapping with identical length distributions, so every policy masks the same number of tokens; this is tested [N]. With overlapping blocks, concentrated placements would produce *fewer* masked tokens — a confound.
   * **`spectral_change`:** a generic-change saliency control.
   * **`inverse_phenology`:** a directionality control.
   * **Random-initialisation encoder probe:** a frozen-feature floor.
   * **Random Forest on interpolated series:** a strong classical reference for crop time series.
4. **Label hygiene for the phenology signal [Rec].**
   * Computed per sample, from that sample's own input only, for pretraining rows only.
   * **No crop calendars derived from labels**, since that would leak labels into SSL.
   * Clouds are screened with a blue-reflectance heuristic plus a running median, because uncleaned PASTIS series [R] would otherwise put "phenology" targets on cloud spikes.
5. **Pre-registration [Rec].** Primary endpoint, decision rule, checkpoint-selection rules, label-budget definition and missing-observation protocol are all fixed before any real-data run (notebook §0.2).

## 5. What requires external verification

| Item | Why it matters | How it is handled |
|---|---|---|
| PASTIS archive URL, size (~29 GB [R]) and licence | Storage planning; terms of use | No hard-coded URL; README tells you to check the Zenodo record |
| PASTIS file names (`ParcelIDs_*.npy` vs `INSTANCE_ANNOTATIONS/INSTANCES_*.npy`), array layout `(T,10,H,W)`, `TARGET[0]` = semantic labels, `dates-S2` format | The loader is written against my recollection of the official loader | Checked and fails loudly [N]; synthetic data in the *assumed* format exercises the parser |
| Band order B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 | NDVI and the phenology signal | NDVI plausibility gate [N] |
| Whether parcel IDs are global across patches | Detecting cross-fold duplicate parcels | Detected heuristically; duplicates are dropped from *train* only [N] |
| PASTIS parcel count (~124 k [R]) | Runtime projections | Measured after preprocessing [N] |
| I-JEPA loss (L2 vs smooth-L1) and other code details | Faithfulness claims | Configurable (`ssl_loss`); verify against `facebookresearch/ijepa` |
| T-JEPA (arXiv 2610.05731), UA-JEPA (2026) | Postdate my knowledge | Read before writing any novelty statement |
| CropSTS, DMT-JEPA, DSeq-JEPA, CrossJEPA, LeJEPA details | Related-work matrix | Verify method, objective and splits from full papers |
| AnySat pretraining data overlap with PASTIS | Fairness and leakage of any AnySat comparison | Deferred (C9) |
| The informed-masking papers above | Main novelty threat | Add to the literature matrix with exact citations |

## 6. Final recommended research question

> In a compact temporal JEPA (an I-JEPA-style objective adapted to parcel-level Sentinel-2 time series), does placing prediction targets on temporal windows of high, label-free vegetation-index change improve frozen-representation crop classification relative to uniform placement — under a matched masking budget, model, data and optimisation — and is the benefit larger with few labels or missing observations? Is any benefit **phenology-specific** (vs generic spectral change) and **directional** (vs low-change placement)?

The pilot answers the first part. `RESEARCH_RUN` (controls, 5 folds, 5 seeds) is required for the second.

## 7. Selected dataset and task

**PASTIS, optical only, parcel-level crop-type classification.**

* **Sample definition:** the per-date median over each parcel's pixels.
* **Exclusions:** parcels under 10 px, under 80% label purity, background/void, or with fewer than 3 valid dates.
* **Label:** the majority class.
* **Splits:** official folds — train {1,2,3}, val {4}, test {5} for the pilot; all 5 rotations for the research run.
* **Metrics:** macro-F1 (primary), balanced accuracy, overall accuracy, per-class F1, confusion matrix. No segmentation metrics.

Why this choice, against the alternatives:

* **vs dense PASTIS segmentation.** The hypothesis is temporal; spatial decoding multiplies compute by orders of magnitude on 4 GB without testing it.
* **vs EuroCropsML.**
  * One PASTIS download also supports a later dense phase.
  * Exact acquisition dates and *uncleaned* (cloudy) series make a realistic test of a phenology heuristic.
  * Official folds and tiles enable leakage control and geographic hold-out.
  * I know its file format well enough to write a defensive loader; I am less sure of EuroCropsML's.
  * EuroCropsML remains the best **second** dataset: a few-shot, cross-country design and 176 classes. Its parcels map onto the same cache schema `(X[N,T,C], valid, days, y, group)`.
* **Limitation.** Results are not comparable with published PASTIS segmentation numbers, and median aggregation discards within-parcel heterogeneity (PSE-style pixel sets are a later option).

## 8. Exact experimental pipeline

1. Parcel extraction (sharded, resumable, SHA-256 recorded).
2. Official-fold split. Leakage assertions: patches, rows, pretraining ⊆ train, cross-split parcel IDs.
3. Training-only normalisation.
4. **Encoder** (shared by every neural method):
   * token embedding LN(W_x·bands) + W_d·sin-encoding(day), with W_d initialised small;
   * pre-LayerNorm transformer, d=128, 4 layers, 4 heads;
   * about 0.8 M parameters.
5. **JEPA:**
   * context encoder plus EMA target encoder (momentum 0.996 → 1);
   * predictor 64-wide with 3 layers, one pass per block;
   * smooth-L1 loss on LayerNorm-ed target outputs;
   * AdamW, lr 1e-3 with cosine decay, weight decay 0.05;
   * 40 epochs, batch 256, bf16.
6. **Masking:**
   * 4 non-overlapping target blocks, each 8–15% of the valid dates;
   * a context window covering 85–100%, with the targets removed;
   * a minimum of 3 context tokens.
7. **Placement policy.** Block-start probability = 0.8 × (normalised sum of per-date weights over the window) + 0.2 × uniform.
   * `random_block`: weight 1 (the baseline).
   * `phenology`: |dNDVI/dt| after cloud screening and a running median, normalised per sample (the modification).
   * `spectral_change` and `inverse_phenology`: controls (research run only).
8. **Downstream:**
   * **Supervised from scratch.** Same encoder; class-weighted cross-entropy (inverse-sqrt frequency); 0–30% random observation dropout as augmentation; best checkpoint by validation macro-F1.
   * **Fine-tuning.** Identical loop *and* hyperparameters (lr 1e-3), initialised from the EMA target encoder, so only the initialisation differs. An earlier draft used a lower encoder LR, which confounded the comparison.
   * **Linear probe.** Frozen, mean-pooled target encoder; L-BFGS multinomial logistic regression; weight decay chosen on validation.
   * **Random Forest** on 36-date interpolated features.
   * **Random-initialisation probe** as a control.
9. **Label budgets.** Nested and stratified per class, with at least 1 labelled parcel per class.
   * Pilot: linear probe at 1/5/10/25/100%; fine-tuning, supervised and Random Forest at 1/10/100%.
   * Every method uses the same subsets for a given seed.
10. **Missing observations.** Fixed, nested random removal of 30/50/70% of dates (at least 3 kept), plus a 1 May–30 June 2019 gap. The masks are identical for every model.
11. **Statistics.**
    * Mean ± SD over seeds and folds.
    * Paired differences against `random_block` on identical test parcels and label subsets, with per-pair bootstrap 95% CIs.
    * The pre-registered verdict rule (notebook §0.2).

**Stop / redesign conditions (decision gates)**

* **G1** — any sanity test fails → fix it first.
* **G2** — the supervised baseline does not beat the majority/chance macro-F1 by more than 0.05 → audit labels, band order and metrics.
* **G3** — collapse indicators for 3 epochs: effective rank falls below 50% of its pre-training value (floor 1.2), or feature std < 1e-3 **or** a positional shortcut (share of target variance explained by date > 0.5, or final SSL loss < 5% of the first-epoch loss; both thresholds calibrated on synthetic data only) → lower the LR, raise the EMA start, check the token embedding.
* **G4** — JEPA linear probe ≤ random-initialisation probe → the baseline JEPA is not learning; do not interpret policy differences.
* **G5** — hypothesis verdict: supported / contradicted / inconclusive / insufficient.

## 9. Main technical and computational risks

* **PASTIS format assumptions may be wrong.** The notebook stops with diagnostics rather than guessing.
* **Phenology signal contaminated by clouds or nodata dates.** The blue-band heuristic is simple. Inspect the season-wise placement plot (13.7) and the enrichment statistic.
* **Effect size smaller than seed variance.** With 3 seeds, only large consistent effects can be "supported". Inconclusive is a likely and acceptable pilot outcome.
* **Positional shortcut (observed during validation).** In a temporal JEPA, target representations can become predictable from the acquisition date alone. With a plain sum of content projection and sinusoidal date encoding, this happened on synthetic data. The SSL loss fell to ~0.005 while fine-tuning from the pretrained encoder was far worse than training from scratch, and the effective-rank collapse check did **not** catch it. Mitigations now in place: LayerNorm-ed content plus a small-init learned date projection, a per-epoch `target_date_var_ratio` diagnostic plus a loss-collapse ratio, and gate G3. A first threshold (0.8) failed to flag the known shortcut (0.65) in validation; it is now 0.5. Re-calibrate on PASTIS by running the `raw_sum` ablation once. Report this diagnostic in any paper; seasonality makes it a structural risk for temporal JEPAs on crops.
* **The compact JEPA may underperform random features or supervised training on ~70 k parcels.** G4 detects this. The fix is to diagnose the baseline (longer pretraining, block scale), not to tune the modification.
* **Extra tuning budget for the modification.** α and ε are fixed in the pilot. Any later tuning must use the validation fold only and be disclosed.
* **Validation-set optimism at low label budgets.** The full validation fold is used for model selection by every method, which is fair *relatively* but optimistic *absolutely*.
* **Compute.** QUICK_PILOT is local-feasible. RESEARCH_RUN (≈ 100 pretraining runs) needs days locally or a cloud GPU.
* **GPU non-determinism.** Runs are seeded, but bit-exact reproducibility on CUDA is not guaranteed.

## 10. Minimum experiment capable of a credible initial result

**QUICK_PILOT:**

* fold config 0;
* 3 seeds × {`random_block`, `phenology`} × 40 pretraining epochs;
* the supervised and Random Forest baselines;
* the random-initialisation probe;
* linear probes at 5 budgets;
* fine-tuning at 1/10/100%;
* missing-observation tests.

**Credible** here means: G1–G4 pass and the paired primary-endpoint differences are reported with CIs. Even a "supported" pilot verdict only justifies running the controls and folds; it is not a publishable claim.
