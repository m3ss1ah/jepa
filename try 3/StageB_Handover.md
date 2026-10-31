# Crop-JEPA Stage B — Notebook Generation Handover

**Audience:** an LLM that will generate ONE complete, runnable Jupyter notebook (`Crop_JEPA_StageB.ipynb`).
**Author's situation:** a student/researcher with very limited time, a laptop GPU (RTX 3050 Laptop, 4 GB, Windows, Python 3.13, PyTorch 2.6.0+cu124) and free Kaggle/Colab access. The notebook must produce results that can go into a research paper.

---

## 0. Instructions to the generating LLM (read first)

1. Produce a single `.ipynb` (valid nbformat v4; validate with `nbformat.validate`). Mix markdown and code cells exactly as in section 14. Every code cell must be syntactically valid Python (run `ast.parse` on each cell before delivering).
2. **If `Crop_JEPA_StageA_Final.ipynb` is attached, treat it as the source of truth for code style and reusable components** (section 13 lists what to keep verbatim and what to change). If it is NOT attached, rebuild those components from the specifications in sections 5–9.
3. Do NOT fabricate results, numbers, p-values or figures. The notebook computes everything at run time. Where this document gives Stage A numbers, they are context only.
4. Where this document says "verify", the notebook must verify at run time and print a clear warning or raise a clear error, rather than assume.
5. Anything in this document marked **[UNVERIFIED]** is from memory or secondary sources. Implement it defensively, expose it as a config option, and add a markdown note telling the user to check it against the primary source.
6. Constraints that must hold:
   - Runs unchanged on a Windows laptop (no DataLoader worker processes; GPU-side augmentation) and on Kaggle.
   - Resumable: finished runs are skipped, unfinished runs resume from checkpoints. Atomic file writes.
   - All arms of one comparison share the same profile (locked-profile mechanism from Stage A).
   - Fixed seeds, deterministic data order per (seed, epoch), same initial weights for every arm with the same seed (paired design).
   - Run modes: `smoke` (generated data, minutes), `benchmark`, `pilot` (Gate 0), `full`.
   - Sharding across machines via `SHARD_ID, NUM_SHARDS`.
   - Every run writes its config, training log, eval results, and predictions to disk.
7. Keep the user-facing config in ONE cell ("Quick configuration").
8. Add short markdown explanations before each section (what it does, what to look at in the output).

---

## 1. Project summary

**Topic:** self-supervised learning (JEPA family) for label-efficient crop-type recognition from field photos.

**Stage A (done):** a 2×2 factorial pilot of two I-JEPA modifications on 1,200 LUCASVision images (12 crops × 100):
- vegetation-aware target placement ("veg"): target blocks placed preferentially on high-Excess-Green (ExG) regions;
- multi-scale target blocks ("ms").

Setup: ViT-Tiny/16 @160 px, 200 epochs, batch 128, bf16, 3 seeds, 5 label draws per regime, linear probe + k-NN, macro-F1, paired bootstrap + Wilcoxon + Holm, pre-defined gate G1.

**Stage A outcome:** G1 FAILED. Veg-aware gave about +2 pp at 10-shot/25-shot/10% on the probe (CIs include zero except one 10% contrast), about −1.4 to −2.2 pp on k-NN, multi-scale had no detectable effect. Supervised ViT-Tiny: 53.3 ± 2.2 % at 100 % labels.

**Stage A diagnosis (why the pilot could not answer the question):**
- Only **1,200 optimizer steps** in total (840 train images ÷ 128 batch = 6 steps/epoch × 200 epochs).
- Features barely trained: in the preserved training log (veg_ms seed 2) effective rank fell ≈30 → 13 of 192 dimensions, and online validation k-NN F1 was flat (18.2 % at epoch 1, 18.7 % at epoch 200). `random_init` beat all SSL arms at 1-shot (13.3 vs 12.2).
- Splits were **not** group-aware (bug, section 3), contrary to what the Stage A report claimed. The practical leakage in the 1,200-image sample was small (10 survey points repeat across years among 1,190 unique points), but it is expected to grow in the full dataset.
- Only 3 seeds and a 180-image test set.
- Unverified confound: greenness correlates with image position in field photos, so "veg-aware" may just be a spatial prior.

**Stage B goal:** properly powered, properly controlled re-test, designed so every outcome is publishable (section 2).

---

## 2. Research questions, hypotheses, decision rules

**RQ1.** With adequate training (full LUCASVision, ≥ ~8–10k optimizer steps), does greenness-biased target placement improve label-efficient probing over uniform placement?

**RQ2.** If so, is the cause greenness itself, or image position / difficulty? (controls)

**RQ3.** Does the answer depend on the collapse regime (EMA I-JEPA vs a SIGReg-regularised JEPA)? (Tier 2)

**Arms (Tier 1, core grid):**

| Arm id | Target placement | Purpose |
|---|---|---|
| `ijepa` | uniform random positions | baseline |
| `veg_pos` | softmax(+z·greenness/τ) mixed with ε uniform (Stage A "veg") | the hypothesis |
| `veg_neg` | softmax(−z·greenness/τ) mixed with ε uniform | polarity control: keep vegetation in context, mask soil/sky |
| `pos_matched` | sampled from the *average position distribution* of `veg_pos`, ignoring pixel content | separates greenness from spatial prior (key control) |

**Optional arm (Tier 2):** `attn` (content-adaptive saliency baseline, section 6.5).

**Dropped:** `ms`, `veg_ms` (no effect in Stage A; keep only as a pilot appendix, do not rerun).

**Baselines:** `random_init`, `supervised` (ViT-Tiny from scratch), optional frozen `dinov2` (ViT-S/14, needs internet).

**Hypotheses (pre-registered, written to `stats/preregistration.json` by the notebook before any result is computed):**
- H1: `veg_pos` > `ijepa` on the primary endpoint.
- H2: `veg_pos` > `pos_matched` (greenness beyond position).
- H3: `veg_pos` > `veg_neg` (polarity matters).
- H4 (Tier 2): the veg effect is smaller under SIGReg than under EMA I-JEPA.

**Primary endpoint:** linear-probe macro-F1 averaged over the regimes {10-shot, 25-shot, 10 %}. Unit of analysis = pretraining seed. Secondary: each regime separately, k-NN, per-class F1.

**Interpretation table (the notebook prints which row applies):**

| Result | Conclusion |
|---|---|
| `veg_pos` beats `ijepa`, `pos_matched`, `veg_neg` | greenness-guided placement helps (positive paper) |
| `veg_pos` ≈ `pos_matched` > `ijepa` | effect is a spatial prior, not greenness |
| `veg_pos` ≈ `ijepa` ≈ others (equivalence within ±1.5 pp) | placement does not matter at this scale; training regime dominates (negative result) |
| `veg_pos` < `ijepa` | greenness-biased placement hurts |

---

## 3. Known Stage A problems the notebook must fix

| # | Problem | Location in Stage A | Required fix |
|---|---|---|---|
| 1 | `METADATA_CSV` is set to the metadata file and then immediately overwritten with `None`; the group column is never used. Log printed "no 'group' column"; audit: 1,200 groups of size 1. | Quick-configuration cell | Remove the overwrite. Build `metadata.csv` with a real `group` column and pass it to `discover_dataset`. |
| 2 | Group id was parsed with `filename.split('_')[1]`, which is correct (survey-point id), but the resulting `group` column was never used because of bug 1. 10 points repeat across years in the 1,200-image sample. | `setup_lucas_data` | Use the confirmed regex (section 4.3), keep the year, add the leakage audit. |
| 3 | Dataset capped at 100 images/class (1,200 total). | `setup_lucas_data(limit_per_class=100)` | `limit_per_class=None` (all images). |
| 4 | Fixed 200 epochs = 1,200 steps. | Config | Define the budget in steps; derive epochs; log both (section 5.3). |
| 5 | Context tokens differ by arm (veg 31.0 vs ijepa 28.1) — confound. | MaskGenerator | Log per arm; add optional `ctx_tokens_cap`; include in diagnostics table. |
| 6 | Only 3 seeds. With n=5 paired seeds the minimum exact two-sided Wilcoxon p is 0.0625. | Config | Use **6 seeds** (min p = 0.031). |
| 7 | Many uncorrected intervals; one "significant" result among ~56. | Statistics | One pre-registered primary endpoint; Holm over a small named family; secondary results labelled exploratory. |
| 8 | Supervised baseline used validation-selected epochs and augmentation while probes were frozen — not like-for-like. | `train_supervised_regime` | Keep it, but label it "upper reference, not matched"; also report val-selection explicitly. |
| 9 | sklearn/scipy `OptimizeWarning: Unknown solver options: iprint` flooded the output (version mismatch under Python 3.13). | `linear_probe` | Suppress this specific warning category/message; do not hide other warnings. |
| 10 | Stage A report claimed split by survey location. | Report | Notebook must print an explicit "split is grouped by `<field>`; N groups; max group size" line. |

---

## 4. Data specification

### 4.1 Dataset
- **LUCASVision** (EU Joint Research Centre), crop photos from the LUCAS survey, 12 crop classes. Paper: Yordanov et al., "Crop identification using deep learning on LUCAS crop cover photos", *Sensors* 2023.
- Stage A source: `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/`, one sub-folder per class code, files listed in the HTML directory index (regex `href="([^"]+\.jpg)"`).
- Size: ≈ 15,876 images reported in a 2026 secondary source (AgriField-40K table). **[UNVERIFIED]** — the notebook must count what it downloads and print per-class counts.
- Stage A used 100 random images per class (seed 42).

### 4.2 Classes

| Code | Crop |
|---|---|
| B11 | Common wheat |
| B12 | Durum wheat |
| B13 | Barley |
| B14 | Rye |
| B15 | Oats |
| B16 | Maize |
| B21 | Potatoes |
| B22 | Sugar beet |
| B31 | Sunflower |
| B32 | Rapeseed |
| B33 | Soya |
| B55 | Temporary grassland |

Chance macro-F1 = 8.3 %. **Expect class imbalance** in the full set; macro-F1 handles it, but the k-shot logic must skip regimes where fewer than 80 % of classes have k training images (Stage A rule, keep).

### 4.3 Grouping / leakage control (critical)
1. Download all images (resumable, 20 threads, skip existing files, retry failures, log failures).
2. **Print the first 30 filenames per class** and store a sample in `stats/filename_sample.txt`.
3. **Group id = LUCAS survey-point id (CONFIRMED by the user on the Stage A sample).** Filenames look like `LUCAS<year>_<point_id>_<suffix>.jpg`; the point id is the second underscore-separated token (an 8-digit number). Default `GROUP_REGEX = r'^LUCAS\d{4}_(\d+)_[A-Za-z0-9]+\.jpg$'`, capture group 1 = group id. On the 1,200-image Stage A sample the user found **1,190 unique points across 1,200 images**: 10 survey points repeat across survey years, so the same location can appear in different files. Splitting on this token is therefore required. Implementation rules: apply the regex to every file; if any file does not match, print the count and 10 examples and fall back to the second `_` token for those files; if the match rate is below 95 %, stop with a clear error instead of silently falling back to per-image groups. Also extract the year (`LUCAS(\d{4})`) and store it as a `year` column for the audit. **At full scale expect many more repeated points than in the sample** (repeats grow roughly with the square of the sampling fraction, so a 7.6 % sample hides most of them); the audit must report how many points appear more than once, how many span several years, and the max group size.
4. Leakage audit cell: groups spanning splits must be 0; print number of groups, max/median group size, and the share of images in groups of size > 1.
5. Keep the Stage A near-duplicate check (dHash 9×8, Hamming ≤ 4, union-find merge of groups).
6. Split: group-stratified by class, `split_seed=42`, val 15 %, test 15 %, train 70 % (Stage A `group_stratified_split`). Splits are written once and never silently regenerated.

### 4.4 Caching
- Read once, EXIF-transpose, RGB, `ImageOps.fit` to `cache_res` (default **192**; 256 optional), store as one uint8 `.npy` (memmap-built). Approx sizes for 15.9k images: 192 px ≈ 1.76 GB, 256 px ≈ 3.12 GB.
- Load into RAM only if the cache is < 35 % of system RAM, else memmap (Stage A `CachedImages`).
- All augmentation on GPU (`roi_align` random-resized-crop + flip), no DataLoader workers.
- Kaggle path: allow `DATA_ROOT` / `CACHE_DATASET_DIR` pointing to a pre-built cache uploaded as a private Kaggle dataset, to avoid re-downloading each session.

### 4.5 Label draws (evaluation subsets)
- Regimes: shots `(1, 5, 10, 25, 50, 100)` and percents `(10, 100)`; 5 draws per regime (1 for 100 %). Draw seed 1234; saved to `label_draws.json`; identical for every arm and seed.
- Print the training count per class and the list of skipped regimes.

---

## 5. Model and training specification (reuse from Stage A)

### 5.1 Architecture (ViT-Tiny/16, ≈5.49 M parameters)
- Patch 16, image 160 (10×10 = 100 tokens), dim 192, depth 12, heads 3, MLP ratio 4, LayerNorm eps 1e-6, fixed 2-D sin-cos positional embeddings (not learned), no CLS token, final LayerNorm, features = mean over tokens.
- Weight init: trunc-normal std 0.02, zeros for bias, I-JEPA depth-dependent rescaling of residual projections (`div sqrt(2·(layer+1))`).
- Attention via `F.scaled_dot_product_attention`.
- Predictor: dim 128, depth 4, heads 4; linear embed from encoder dim; learned mask token + sin-cos positions for targets; output projected back to encoder dim.

### 5.2 I-JEPA training step
- Target encoder = EMA copy of the context encoder, eval mode, no gradients; processes the FULL image; its outputs are LayerNorm'd (no affine) over the feature dim.
- Context encoder processes only context tokens (`keep_idx`). Predictor predicts each target block from context tokens + positional mask tokens.
- Loss: smooth-L1, token-weighted over all predicted target tokens (sum over blocks ÷ total elements).
- AdamW; peak LR = `blr × batch/256` with `blr=5e-4`; linear warmup 10 % of steps then cosine to `min_lr=1e-6`; weight decay 0.04 → 0.4 (cosine ramp, no decay on biases/norms/mask token); EMA momentum 0.996 → 1.0 (linear in step); grad clip 3.0.
- Precision: `bf16` on Ampere+ (RTX 3050), `fp16` + GradScaler on T4/P100-class (no bf16 on T4), `fp32` fallback.
- Augmentation: random-resized-crop scale (0.3, 1.0) + horizontal flip, on GPU, deterministic per (seed, epoch).
- Same seed ⇒ same initial weights, same image order, same crops for every arm (paired design).

### 5.3 Training budget (CHANGED from Stage A)
- Define `TARGET_STEPS` (default **9,000** for the core grid; **26,000** for Gate 0 pilot) and derive `epochs = ceil(TARGET_STEPS / steps_per_epoch)`. `steps_per_epoch = n_train // batch_size` (drop_last).
- Log `total_steps` and `epochs` at the start of every run and in `config.json`. Warmup/cosine/EMA/WD schedules are defined over total steps.
- With ≈11.1k training images and batch 128: 86 steps/epoch → ≈105 epochs for 9,000 steps.

### 5.4 Monitoring per epoch (keep, extend)
- Loss, LR, WD, EMA, epoch time, img/s, peak memory, mean context/target tokens, non-finite steps.
- Collapse monitors on the target encoder over a fixed validation batch (`monitor_n=512`): centred RankMe effective rank, per-dim std of L2-normalised features, feature norm, top singular-value share.
- Online k-NN (k=20, T=0.07) macro-F1 on validation using up to 2,000 training images, every 10 epochs. **Labels used only for monitoring, never for checkpoint selection or hyper-parameters.**
- NEW: log the **mean ExG of the union of target blocks** per epoch (manipulation check: should be higher for `veg_pos`, lower for `veg_neg`, ≈baseline for `pos_matched`).
- Checkpoint every 5 epochs and on time-budget exhaustion; atomic writes; config-hash guard on resume.

### 5.5 Hardware profiles (keep names)

| Profile | Model | Image | Batch | Precision | Notes |
|---|---|---|---|---|---|
| `laptop_4gb` | ViT-T/16 | 160 | 128 | bf16 | main profile; measured 576 img/s (benchmark), ≈880–980 img/s steady, 1.22 GB peak |
| `kaggle_t4_fast` | ViT-T/16 | 160 | 128 | fp16 | `max_session_hours=11` |
| `kaggle_t4` | ViT-S/16 | 224 | 128 | fp16 | confirmation runs only |
| `laptop_4gb_vits` | ViT-S/16 | 224 | 64 | bf16 | slow but fits |
| `cpu_debug` / smoke | tiny | 64 | 32 | fp32 | pipeline test |

Auto-select the profile from the detected GPU; refuse to mix profiles inside one output folder (`locked_profile.json`).

---

## 6. Target-placement (mask generator) specification

### 6.1 Common (from Stage A, keep)
- Grid 10×10 (at 160/16). 4 target blocks per image; block scale (0.15, 0.20) of the image area, aspect (0.75, 1.5); block *sizes* are shared across the batch, *positions* are per image. One context block with scale (0.85, 1.0), aspect 1, **minus the union of target blocks**; re-sample the context up to 10 times if it keeps fewer than `min_context=10` tokens; fall back to "all non-target tokens"; finally "all tokens".
- Context tokens per image are equalised to the batch minimum (random subset).
- Own `torch.Generator` objects (CPU and device) with checkpointable state.

### 6.2 Greenness score (keep)
Chromatic ExG on un-normalised [0,1] pixels: `exg = (2g − r − b)/(r+g+b+1e-6)`, average-pooled to the patch grid. For a block of size (h,w) the block greenness is the average-pool of the patch map with kernel (h,w), stride 1; z-score across all valid positions per image.

### 6.3 Position sampling per arm
For each target block, with `nr × nc` valid top-left positions, `ε = 0.2`, `τ = 0.5` (config):
- `ijepa`: uniform over positions.
- `veg_pos`: `p = (1−ε)·softmax(z/τ) + ε/(nr·nc)`.
- `veg_neg`: `p = (1−ε)·softmax(−z/τ) + ε/(nr·nc)`.
- `pos_matched`: `p = P[h,w]`, a fixed per-block-size map (below). No pixel content is used at training time.

### 6.4 Position prior for `pos_matched` (must be implemented exactly like this)
1. Enumerate all reachable block sizes (h,w) from the scale and aspect ranges (grid sweep over scale and aspect, same rounding as `_size`, clipped to the grid).
2. Using the **training split only**, a fixed generator, and the **same GPU augmentation as training**, run `PRIOR_IMAGES` (default 4,000) augmented crops in batches. For each block size (h,w), compute the full `veg_pos` probability map `p` (including the ε mixture) for each image and average over images → `P[h,w]` of shape `(nr, nc)`, sums to 1.
3. Save to `outputs/priors/pos_prior_{h}x{w}.npy`; the estimate depends only on data and config, not on the pretraining seed, so it is shared by all seeds. Hash it into the run config hash.
4. **Acceptance check:** after training mask generation for a few hundred batches, print the KL divergence between the empirical target-centre histogram of `veg_pos` and of `pos_matched` (should be ≈ 0), and the mean target ExG of each arm (should differ: `pos_matched` ≈ baseline-ish unless position and greenness are collinear — that collinearity is itself a finding).

### 6.5 Optional `attn` arm (Tier 2, content-adaptive baseline)
- Score per patch from the EMA target encoder: distance of each patch token from the mean token of the image, `‖h_i − mean_j h_j‖`, computed from the target-encoder forward already needed for the loss (requires computing `h` before the mask generator; pooled over blocks like greenness). Same softmax/ε/τ machinery as `veg_pos`.
- Label this clearly as an AttMask-style stand-in (no CLS attention exists in this ViT). Config flag `ENABLE_ATTN_ARM=False` by default.

### 6.6 Manipulation-check figure
For each arm, plot (a) mean ExG of target regions, (b) heat map of target-centre frequency, (c) 8 example images with context/target overlays.

---

## 7. Tier 2 — SIGReg / LeJEPA arm (build as a separate, optional section)

Purpose: test whether placement matters once collapse is handled by a principled regulariser (RQ3).

- Source: LeJEPA (Balestriero & LeCun, 2025, arXiv 2511.08544). Reported to need no stop-gradient or teacher-student, and to work for in-domain pretraining on small datasets.
- Official code: the `rbalestr-lab/lejepa` repository. **[UNVERIFIED]** details from memory: loss = (1−λ)·invariance + λ·SIGReg, λ≈0.02; SIGReg = sliced Epps–Pulley test against an isotropic Gaussian over many random 1-D projections (≈256 slices, ≈17 quadrature knots); views = a few global crops + several local crops. **Implement from the paper/official repo, expose all of these as config, and print a markdown warning that the implementation must be checked against the official code.**
- Placement analogue: the "arms" act on the **local-crop positions** (uniform / greenness-biased / anti-greenness / position-matched), using the same score and prior machinery as sections 6.2–6.4.
- Same ViT-Tiny backbone, same evaluation, same seeds, same primary endpoint. Run only after Gate 0 (section 9) says so, or when the user sets `ENABLE_LEJEPA=True`.
- Benchmark per-step cost before queueing (expected to be a multiple of I-JEPA's cost, **[UNMEASURED]**).

---

## 8. Evaluation specification

### 8.1 Features
Target (EMA) encoder, mean of final-layer tokens, centre image at training resolution; features standardised using a scaler fitted on all training features (no labels).

### 8.2 Classifiers (keep)
- **Linear probe:** `LogisticRegression(max_iter=2000)` over `C ∈ {1e-3, 1e-2, 1e-1, 1, 10}`, C chosen by validation macro-F1; test predictions stored.
- **k-NN:** cosine similarity, temperature-weighted votes (T=0.07), `k = min(20, smallest per-class count among labelled images)`.
- Per regime × draw × arm × seed; store test predictions (`preds.npz`) and the confusion-matrix-based macro-F1 / accuracy / balanced accuracy; per-class F1 at 100 %.

### 8.3 Baselines
- `random_init`: same seed's untrained encoder, same evaluation.
- `supervised`: ViT-Tiny from scratch, label smoothing 0.1, AdamW wd 0.05, 100 epochs (config), crop scale (0.35, 1.0); regimes `10pct` and `100pct` (add `1pct` if the count is sensible); best-validation-epoch checkpoint used for test; run for all seeds. Label as an unmatched upper reference.
- `dinov2` (optional, needs internet): frozen `dinov2_vits14` at 224, same evaluation, seed 0 only.

### 8.4 Test-set note
With ≈15 % of ≈15.9k images the test set is ≈2,400 images, much less noisy than Stage A's 180. Print its size and per-class counts.

---

## 9. Gates (decision logic coded in the notebook)

- **Gate −1 (data):** group split verified (0 leaking groups, or loud warning), class counts printed, ExG-position analysis done (below). Fail = stop and print why.
- **ExG-position analysis (cheap, runs before training):** mean ExG map over the training set at the patch grid; Pearson/Spearman correlation between mean ExG and row index and column index; fraction of variance in per-image ExG explained by patch position. Save the heat map. Print a warning if position explains a large share (the veg effect is then likely a spatial prior).
- **Gate 0 (training signal) — `RUN_MODE="pilot"`:** train `ijepa` seed 0 for the pilot budget (default 26,000 steps) and `random_init`; evaluate. Pass if online k-NN val F1 rises clearly above the `random_init` k-NN and the RankMe curve does not collapse below `0.05 × enc_dim`. Print PASS/FAIL with the curves. On FAIL, print the recommendation: enable the Tier-2 SIGReg arm and/or lower LR / raise steps. Do not auto-run the grid.
- **Gate 1 (primary result):** apply the interpretation table of section 2, with the statistics of section 10, and write `stats/gate_G1_stageB.json`.

---

## 10. Statistics specification

- Unit = pretraining seed (6 seeds: 0–5). Label draws within a seed are averaged (or nested), never treated as independent seeds.
- Primary comparisons (the Holm family, size 3): `veg_pos − ijepa`, `veg_pos − pos_matched`, `veg_pos − veg_neg`, on the primary endpoint.
- For each: point estimate (pp), 95 % hierarchical bootstrap CI (resample seeds → draws within seed → test images shared across arms, B ≥ 2,000; reuse Stage A `bootstrap_stat` logic), exact Wilcoxon signed-rank on the 6 seed-level differences, Holm-adjusted p. Also report a paired sign-flip permutation p.
- **Equivalence (for the negative-result branch):** TOST-style check against a margin of ±1.5 pp (config) using the bootstrap CI (equivalent if the CI lies inside the margin).
- Secondary / exploratory (labelled as such, no claims): each regime separately, k-NN, `veg_neg − ijepa`, `pos_matched − ijepa`, per-class F1 differences.
- Always report effect sizes with CIs next to p-values. Print the number of intervals computed and the expected number of chance exclusions at 95 %.
- Dose-response (Tier 2): τ ∈ {0.25, 0.5, 1.0, ∞(=uniform)} for `veg_pos`, 3 seeds, plot Δ vs τ.
- Data-size axis (Tier 2): train on 10 %, 30 %, 100 % of the training pool at **fixed step budget**, `ijepa` vs `veg_pos`, 3 seeds.

---

## 11. Compute and time plan (for the notebook's printed projection and for the user)

Measured on the RTX 3050 laptop (Stage A): 576 img/s in the benchmark; ≈880–980 img/s steady-state training; 1.22 GB peak at batch 128 with ViT-Tiny @160.

For ≈11.1k training images (70 % of ≈15.9k):

| Item | Runs | Laptop time (estimate) |
|---|---|---|
| Gate 0 pilot (~26k steps, 300 epochs) | 1 | ≈ 1–1.6 h |
| Core grid: 4 arms × 6 seeds, ~9k steps | 24 | ≈ 8–13 h |
| Tier 2 extras (attn, τ sweep, size axis, LeJEPA) | varies | ≈ +8–15 h **[estimate]** |
| ViT-S/224 confirmation (2 arms × 1 seed) | 2 | ≈ 3 h per 100 epochs **[FLOP-scaled estimate]** |

Platforms (check current limits; they change): Kaggle free tier ≈ 30 GPU-h/week (floating), 12-h sessions, 20 GB disk, T4×2 or P100; Colab free: no published quota, ≤ 12 h, idle disconnects — overflow only. Kaggle T4 speed for this workload is **unmeasured**; the benchmark mode must measure it.

Notebook must: print projected hours per run and for the queue; stop cleanly before the session limit (`max_session_hours`, `eval_reserve_min`); support `RESUME_FROM_DIR` for Kaggle; support `SHARD_ID/NUM_SHARDS`.

---

## 12. Outputs the notebook must write

```
OUTPUT_DIR/
  stats/
    preregistration.json         # hypotheses, endpoint, family, margin, written BEFORE results
    filename_sample.txt
    dataset_audit.json           # counts, groups, leakage audit
    exg_position_analysis.json
    benchmark.json
    summary_macro_f1.csv
    all_results_long.csv
    seed_level_primary.csv       # one row per (arm, seed): primary endpoint
    paired_comparisons.csv       # primary + secondary, flagged
    equivalence.csv
    gate_G0.json, gate_G1_stageB.json
    token_budget.csv, manipulation_check.csv
  priors/pos_prior_{h}x{w}.npy
  runs/{arm}_s{seed}/ config.json train_log.csv last.pt encoder_final.pt eval.json preds.npz
  figures/
    exg_position_heatmap.png
    manipulation_check.png        # mean target ExG, centre histograms, example masks
    label_efficiency_{probe,knn}.png
    primary_endpoint_forest.png   # deltas with CIs, all comparisons
    seed_level_dots.png           # raw seed-level points per arm
    training_curves.png           # loss, online kNN, RankMe, feat std
    feature_spectrum.png          # singular values per arm
    per_class_delta.png
    confusion_best.png
  results_bundle.zip              # no cache, no .pt unless ZIP_CHECKPOINTS
  paper_tables/                   # CSV + LaTeX (.tex) of: main results table, primary comparisons, ablations
```

---

## 13. Reuse map from `Crop_JEPA_StageA_Final.ipynb`

**Keep (copy, adjust only what is listed):**
`describe_gpu`, `Config`/profiles (add new fields), `seed_everything`, atomic save helpers, `config_hash` (add `arm` ids, `tau`, prior hash), `hours_left`, RNG state helpers, `discover_dataset`, `_process_image`, `build_cache`, `UnionFind`, `hygiene_and_groups`, `group_stratified_split`, `make_or_load_splits`, `make_or_load_draws`, `CachedImages`, `random_resized_crop`, `GPULoader`, `sincos_pos_embed`, `ViT`, `Predictor`, `SupervisedViT`, `param_groups`, `lr_at`, `cos_ramp`, `representation_stats`, `online_knn_f1`, `JEPARun` (extend), `extract_features`, `confusion`, `metrics_from_cm`, `linear_probe`, `knn_predict`, `evaluate_features`, `run_random_init`, `run_dinov2`, `train_supervised_regime`, `benchmark`, `flop_split`, `lock_profile`, queue logic, `f1_fast`, `bootstrap_stat`, `holm`.

**Change:**
- `setup_lucas_data`: all images, real group id, resumable, per-class counts.
- Quick-configuration cell: remove the `METADATA_CSV` overwrite; add the new config keys (section 15).
- `MaskGenerator`: add `veg_neg`, `pos_matched`, optional `attn`; keep `ijepa`; drop `ms` options from the default arm list; add token and ExG logging; add optional `ctx_tokens_cap`.
- Training budget → steps (section 5.3).
- `gate_g1` → replaced by the Stage B gates and statistics (sections 9–10).
- Statistics: seed-level unit, exact Wilcoxon, equivalence, pre-registration file.
- Warning handling for the sklearn `iprint` message.

**Add:** ExG-position analysis, position-prior estimation, manipulation-check outputs, Gate 0 pilot mode, paper-table export, Tier-2 sections behind flags.

---

## 14. Required notebook structure (cells)

0. Title + how-to-use (modes: smoke / benchmark / pilot / full) + warning list of what changed from Stage A.
1. **Quick configuration** (only cell the user edits).
2. Environment check.
3. Profiles and `Config`.
4. Utilities.
5. Data: download (all images), filename inspection, groups, hygiene, split, leakage audit, cache, label draws, dataset audit.
6. **Gate −1:** ExG-position analysis.
7. GPU data pipeline.
8. Model (ViT, predictor, supervised).
9. Mask generator (all arms) + position prior estimation + unit tests (shapes, no target/context overlap, probabilities sum to 1, `pos_matched` histogram ≈ `veg_pos` histogram).
10. I-JEPA training class with monitors.
11. Evaluation (probe, k-NN, baselines).
12. Benchmark and projected time.
13. **Gate 0 pilot** (`RUN_MODE="pilot"`).
14. Run queue (core grid).
15. Tier 2 sections (each behind its own flag): `attn` arm, τ sweep, size axis, SIGReg/LeJEPA, DINOv2 reference.
16. Results tables.
17. Statistics + pre-registered primary analysis + equivalence + interpretation printout.
18. Figures.
19. Paper-table export and bundle.
20. Final markdown: "How to read the results" and a "Limitations to write in the paper" checklist.

---

## 15. Quick-configuration keys (defaults)

```python
RUN_MODE = "full"            # "smoke" | "benchmark" | "pilot" | "full"
PROFILE = "auto"
DATA_ROOT = "./lucas_data"
GROUP_REGEX = r'^LUCAS\d{4}_(\d+)_[A-Za-z0-9]+\.jpg$'   # capture group 1 = survey-point id (confirmed)
LIMIT_PER_CLASS = None       # None = all images
CACHE_RES = 192
OUTPUT_DIR = None
RESUME_FROM_DIR = None
ARMS = ["ijepa", "veg_pos", "veg_neg", "pos_matched"]
ENABLE_ATTN_ARM = False
ENABLE_LEJEPA = False
ENABLE_TAU_SWEEP = False
ENABLE_SIZE_AXIS = False
BASELINES = ["random_init", "supervised"]   # + "dinov2" if internet
SEEDS = [0, 1, 2, 3, 4, 5]
TARGET_STEPS = 9000
PILOT_STEPS = 26000
VEG_TAU = 0.5
VEG_EPS = 0.2
PRIOR_IMAGES = 4000
CTX_TOKENS_CAP = None
EQUIV_MARGIN_PP = 1.5
BOOTSTRAP_B = 2000
SHARD_ID, NUM_SHARDS = 0, 1
OVERRIDES = {}
ALLOW_PROFILE_CHANGE = False
ZIP_CHECKPOINTS = False
```

---

## 16. Acceptance checklist (the LLM should self-verify before delivering)

- [ ] Every code cell parses (`ast.parse`); notebook validates with `nbformat`.
- [ ] `smoke` mode runs end-to-end on generated data without errors in all code paths: all four arms, `pos_matched` prior estimation, evaluation, statistics, figures, bundle.
- [ ] Unit tests in section 14 cell 9 exist and pass in smoke mode.
- [ ] No hard-coded results or invented numbers anywhere.
- [ ] Preregistration file is written before any evaluation result is loaded.
- [ ] Group-split verification prints explicit numbers; fallback to per-image groups prints a loud warning.
- [ ] Steps, epochs, and steps-per-epoch are printed for every run.
- [ ] Same seed ⇒ identical initial weights and data order across arms (assert on the first-batch hash).
- [ ] Resume works (interrupt-and-restart logic exercised at least by a small test).
- [ ] No DataLoader workers; no `localStorage`-style or Windows-incompatible calls; `torch.load(..., weights_only=...)` used safely.
- [ ] T4 path uses fp16 + GradScaler; Ampere path uses bf16; CPU path fp32.
- [ ] All `[UNVERIFIED]` items are exposed as config and flagged in markdown.

---

## 17. Out of scope (do not add)

Multi-scale target arms, ViT-B or larger models, ImageNet-scale data, fine-tuning-based evaluation, hyper-parameter search on test data, any claim about statistical significance beyond the pre-registered family.

---

## 18. Appendix A — Stage A results (context only; do not hard-code)

Linear probe macro-F1 (%), mean ± std over seeds × label draws (n=15; n=3 at 100 %):

| Labels | ijepa | veg | ms | veg_ms | random_init |
|---|---|---|---|---|---|
| 1-shot | 12.2 | 11.6 | 12.0 | 11.8 | 13.3 |
| 5-shot | 19.5 | 18.9 | 19.8 | 19.2 | 19.5 |
| 10-shot | 22.7 | 24.7 | 22.2 | 24.1 | 21.3 |
| 25-shot | 25.2 | 26.9 | 25.0 | 27.3 | 22.5 |
| 50-shot | 29.3 | 30.1 | 28.5 | 29.2 | 26.2 |
| 10 % | 19.7 | 22.5 | 19.2 | 22.3 | 19.2 |
| 100 % | 29.8 | 29.9 | 28.5 | 30.7 | 28.3 |

k-NN (k ≤ 20): ijepa ≥ veg at 5–25 shot by 1.4–2.2 pp. Supervised ViT-Tiny: 100 % labels 53.3 ± 2.2; 10 % labels 21.8 ± 3.4.
G1: 0/3 low-label regimes won by `veg` or `veg_ms` (mean +1.02 / +1.05 pp) → FAIL.
Only veg_ms − ms at 10 % had a CI excluding zero (+3.1 pp, [+0.21, +6.13]; Holm p = 0.004); p-values were optimistic because label draws within a seed share an encoder.
Stage A training log (veg_ms, seed 2): loss 0.458 → 0.045; RankMe 29.9 → 13.1; online val k-NN F1 18.2 % → 18.7 %; ≈880–980 img/s; ctx ≈ 28–31 tokens, tgt ≈ 70.

## 19. Appendix B — Literature to cite in the paper (for the user; the notebook need not use it)

- Assran et al., I-JEPA, CVPR 2023 (arXiv 2301.08243).
- Balestriero & LeCun, LeJEPA, 2025 (arXiv 2511.08544); Mine-JEPA (arXiv 2604.00383) for SIGReg in small-data domains.
- Data-driven masking: AttMask; Hard Patches Mining (Wang et al., CVPR 2023); SemMAE (Li et al., NeurIPS 2022); AutoMAE; DSeq-JEPA (saliency-based curriculum, He et al., Nov 2025).
- Small-data ViT/JEPA: "How to train ViT on small-scale datasets" (arXiv 2210.07240); "Training ViTs with only 2040 images" (arXiv 2201.10728); SCOTT / MIM-JEPA (arXiv 2502.18056); TU Delft BSc thesis on I-JEPA data efficiency (2026).
- Agriculture SSL: Güldenring & Nalpantidis 2021; Sornapudi & Singh 2024 (arXiv 2403.15248); Agri-FM+ (CVPRW 2025); AgriField-40K / AgriMAE (arXiv 2608.07984, includes LUCASVision).
- LUCAS: Yordanov et al. 2023 (LUCASVision); d'Andrimont et al. 2022 (LUCAS cover photos, ESSD).
- The user should confirm novelty with a fresh literature search before claiming it ("vegetation-aware masking" found no direct prior in the searches done so far).
