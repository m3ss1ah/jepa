# Crop-JEPA Stage C: results, interpretation and conclusions

**Data source:** `stageC_results/outputs_stageC/`, from SLURM job 2146, 2026-11-02 20:48 → 2026-11-04 02:48. Every number below comes from the saved artifacts: `FINAL_REPORT.md`, `<phase>/stats/*.csv`, `<phase>/paper_tables/*.csv` and per-run `train_log.csv`. Stage B numbers come from `outputs_stageB/`.

**Units:** "pp" means percentage points of test macro-F1. The **primary endpoint** is the mean test macro-F1 over the 10-shot, 25-shot and 10% label regimes. It uses a frozen-feature logistic probe, averaged over 10 label draws per seed. Seeds are the unit of analysis.

---

## 0. The short version

1. **Vegetation-guided masking (`veg_pos`) gives no improvement over standard I-JEPA once the experiment is fair.**
   - In the main study (20 seeds), veg_pos − ijepa = **−0.09 pp** (95% CI [−0.59, +0.47]).
   - This is statistically **equivalent** to zero within ±1 pp.
   - It replicates on a new data split: **−0.20 pp**.
   - With a larger encoder (ViT-S) it is clearly **worse**: **−1.54 pp**, with veg_pos losing on 0 of 10 seeds' comparisons, i.e. every seed favoured ijepa.
2. **Stage B's promising results were mostly a product of how Stage B was run, not of the vegetation guidance itself.**
   - Re-running Stage B's exact protocol on 10 fresh seeds reproduces Stage B: pooled with Stage B's 6 seeds, veg_pos − ijepa = **+0.60 pp**, p = 0.017.
   - Under the stricter Stage C protocol the effect disappears. That protocol gives every method exactly the same number of context tokens and trains about 3× longer.
   - Stage B's headline gap (veg_pos beats veg_neg by **+3.97 pp**) shrinks to **+0.42 pp** (not significant).
3. **One genuine, confirmed finding is an asymmetry.**
   - Pushing prediction targets **away** from vegetation hurts slightly: veg_neg is −0.50 pp vs ijepa.
   - The harm comes from **what** is in the targets, not **where** they are. A control with veg_neg's exact spatial layout but random content shows no harm (neg_matched − ijepa = +0.02 pp). The content-vs-position test was confirmed: neg_matched − veg_neg = **+0.52 pp**, Holm p = 0.024.
   - Pulling targets **toward** vegetation does not help.
4. **The pipeline itself improved a lot over Stage B.** All methods are **+7.8 pp** better than in Stage B: ijepa goes from 44.6 to 52.4 on the primary endpoint. The I-JEPA features now beat a supervised-from-scratch model when only 10% of labels are available (58.8 vs 54.1). A frozen off-the-shelf DINOv2 model (83.6) is far ahead of everything trained only on LUCAS.
5. **What this means for the paper.** The publishable story is **a carefully controlled negative result with a mechanism**, not "our masking is better". The pitch:
   > Domain-specific masking heuristics that look good in short pilots can vanish once the context-token budget and training length are controlled. We show exactly which part of the heuristic matters (content, not position) and why the ceiling is low.

   That is novel: no prior work tests vegetation-index-guided target placement in JEPA. It is also rigorous: 20 seeds, a plan locked before the main runs, equivalence tests, an independent replication split, a second model size and an exact bridge to the pilot.

---

## 1. What was run

| phase | data split | encoder | methods (arms) | seeds | training steps | context tokens | role |
|---|---|---|---|---|---|---|---|
| Phase 0 | A | ViT-Tiny/16 @160 | ijepa capped vs uncapped | 900 | 9,030 | 28 vs 12 | decides the protocol (validation data only) |
| **core** | A (Stage B split) | ViT-Tiny | ijepa, veg_pos, veg_neg, pos_matched, neg_matched, attn | 100–119 (**20**) | 26,058 | **fixed at 12 for all** | the confirmatory test |
| splitB | B (new group-aware split) | ViT-Tiny | same minus attn | 100–109 (10) | 26,058 | 12 | replication on different data |
| vits | A | **ViT-S/16** @160 | ijepa, veg_pos | 10 | 26,058 | 12 | replication at larger capacity |
| bridge | A | ViT-Tiny | ijepa, veg_pos | 10 | 9,030 | **not capped** (Stage B protocol) | links Stage C to Stage B |

Baselines (core): random_init (20 seeds), supervised from scratch (10 seeds) and frozen DINOv2 ViT-S/14 (1 run; deterministic, no training).

**The methods in plain words.** In I-JEPA the model sees some image patches (the **context**) and must predict the internal representation of other patches (the **targets**).
- **ijepa:** targets placed uniformly at random (the standard method).
- **veg_pos:** targets preferentially on green, vegetated patches, scored by the Excess Green index ExG.
- **veg_neg:** targets preferentially on *non*-vegetated patches (soil, sky, background).
- **pos_matched:** targets placed with the *average spatial layout* of veg_pos, but ignoring each image's content. This is the "is it just position?" control.
- **neg_matched** (new in Stage C): the same idea for veg_neg.
- **attn:** an AttMask-style baseline that places targets on patches the model itself finds salient.

**Execution quality**
- 260 of 260 runs finished: core 150, splitB 60, vits 30, bridge 20.
- No failed runs (`error.txt` count 0), and 0 non-finite training steps in any run.
- No deviations from the locked plan (no `DEVIATIONS.md`). The code hash was unchanged.
- Total time was **~30 h on one GPU**, below the 45–52 h estimate: Phase 0 15 min, core 15.9 h, splitB 7.2 h, ViT-S 5.2 h, bridge 1.0 h.

**Integrity of the plan.** Phase 0 froze the analysis plan with SHA-256 `c6774dac…9343` on 2026-11-02 at 21:04:23. That is before any main-study run, and the main phases were released 1 s later by the automatic sanity gate. In the paper, describe this as **"locally frozen with a timestamped SHA-256 before the main experiments; not externally preregistered."**

---

## 2. Phase 0: the protocol decision

| | uncapped (Stage B style, ~28 context tokens) | capped at K = 12 context tokens |
|---|---|---|
| online k-NN val F1 at 9,030 steps | 0.468 | 0.456 (−1.2 pp; the rule allows up to −3 pp) |
| RankMe (effective rank) | 50.8 | 41.0 (rule requires ≥ 20) |
| non-finite steps | 0 | 0 |

- **Decision:** protocol C, a fixed budget of 12 context tokens for every method.
- Below 12 tokens, only 0–0.3% of batches per arm fall short. The training logs confirm all arms really trained with 11.99–12.00 context tokens on average (`core/paper_tables/token_budget.csv`).
- **Why it matters:** in Stage B, veg_pos silently received more context than ijepa (**31.1 vs 28.2 tokens, +10%**) and pos_matched less (27.0). Any difference between arms could therefore have been "more context", not "better targets". Stage C removes that confound completely.
- **Phase 0 also reproduced Stage B exactly** (uncapped k-NN 0.468, against Stage B's ijepa at 0.463–0.468), so the cluster setup and code behave as before.

---

## 3. Did the manipulation actually work? (Yes, so the null result is real)

A null result is only meaningful if the methods really did something different. Measured during training, averaged over all epochs and seeds (`core/paper_tables/manipulation_check.csv`):

| arm | mean ExG under targets | z-score vs the image's own patches | spatial layout (KL vs veg_pos / vs veg_neg) |
|---|---|---|---|
| ijepa | 0.244 | +0.03 | 0.049 / 0.083 |
| **veg_pos** | **0.274** | **+0.42** | 0 / 0.059 |
| **veg_neg** | **0.210** | **−0.38** | 0.053 / 0 |
| pos_matched | 0.244 | +0.03 | **0.0006** / 0.061 |
| neg_matched | 0.242 | +0.01 | 0.050 / **0.0004** |
| attn | 0.229 | −0.10 | – |

Plain-language reading:
- **veg_pos really put targets on greener patches,** and **veg_neg on less green ones.** The two differ by about 0.8 standard deviations of within-image greenness.
- **The position controls work almost perfectly.** pos_matched copies veg_pos's spatial layout with a KL divergence of 0.0006, about 80× closer than ijepa's 0.049, while its greenness stays at the uniform level (+0.03). neg_matched does the same for veg_neg (0.0004). So H2 and H4 cleanly separate **where** the targets go from **what** is under them.
- **An important caveat for the interpretation: the manipulation is strong in position but modest in content.** I-JEPA's targets are large: 4 blocks covering ~71 of the 100 patches, ~49 distinct patches once overlaps are removed. Large blocks cannot be steered onto green pixels very selectively. veg_pos targets are only 12% greener than uniform ones (0.274 vs 0.244). This is part of *why* the effect is small (see §8).

---

## 4. Main result (core, 20 seeds): the pre-registered tests

### 4.1 Absolute performance (test macro-F1 %, frozen linear probe, mean ± SD over seeds)

| method | 1-shot | 10-shot | 25-shot | 10% labels | 100% labels | **primary endpoint** |
|---|---|---|---|---|---|---|
| ijepa | 25.3 | 46.5 | 52.0 | 58.8 | 67.0 | **52.4 ± 0.6** |
| veg_pos | 24.0 | 46.2 | 52.1 | 58.9 | 66.9 | **52.4 ± 0.7** |
| veg_neg | 24.7 | 45.8 | 51.6 | 58.5 | 66.6 | **51.9 ± 0.4** |
| pos_matched | 24.7 | 45.9 | 51.7 | 58.4 | 66.7 | **52.0 ± 0.7** |
| neg_matched | 25.6 | 46.6 | 52.1 | 58.7 | 66.9 | **52.5 ± 0.7** |
| attn | 25.2 | 45.9 | 51.5 | 58.5 | 66.5 | **52.0 ± 0.5** |
| random_init (untrained) | 10.7 | 18.5 | 20.4 | 22.4 | 29.2 | 20.4 |
| supervised from scratch (upper ref., not matched) | – | – | – | 54.1 | 74.5 | – |
| DINOv2 ViT-S/14, frozen (external reference, 1 run) | 60.5 | 80.7 | 84.0 | 86.1 | 88.0 | 83.6 |

All six I-JEPA variants lie inside a **0.6 pp band** (51.9–52.5), and all are ~32 pp above an untrained network. The k-NN evaluation tells the same story: 45.7–46.7, against 14.5 for random_init (`main_results_knn.csv`).

### 4.2 The hypotheses

| | question | Δ (pp) | 95% CI | seeds favouring the first arm | Holm p | verdict |
|---|---|---|---|---|---|---|
| **H1** | Does vegetation guidance beat standard I-JEPA? (veg_pos − ijepa) | −0.09 | [−0.59, +0.47] | 8/20 | 0.83 | **No: equivalent** (TOST within ±1 pp) |
| **H2** | Is it more than just the spatial layout? (veg_pos − pos_matched) | +0.37 | [−0.14, +0.92] | 14/20 | 0.14 | not confirmed (equivalent) |
| **H3** | Does polarity matter? (veg_pos − veg_neg) | +0.42 | [−0.10, +0.95] | 15/20 | 0.10 | not confirmed (equivalent) |
| **H4** | Is veg_neg's deficit about content, not position? (neg_matched − veg_neg) | **+0.52** | **[+0.09, +0.97]** | 12/20 | **0.024** | **confirmed** |
| **E1** | Is a static spatial prior harmless? (pos_matched − ijepa) | −0.46 | [−0.87, −0.07] | 6/20 | – | equivalent (inside ±1 pp), but CI below 0 |
| **E2** | Is veg_neg's spatial layout harmless? (neg_matched − ijepa) | +0.02 | [−0.39, +0.43] | 11/20 | – | **equivalent** |
| **H5** | Does vegetation guidance help more on heterogeneous (row-crop) scenes? | −0.90 (high − low) | [−1.38, −0.36] | 5/20 | 0.998 | **No, the reverse** |

Pre-registered decision-matrix outcome: **P-B, "equivalent to uniform placement on average."**

**Was the study big enough to see a real effect?** Yes. At n = 20, the Phase-0 power simulation gives **90% power** for a true +0.9 pp effect at the strictest Holm threshold (α = 0.0125), and 70% for +0.7 pp. Stage B's apparent effect was +0.89 pp. The 95% CI for H1 tops out at +0.47 pp, so any benefit larger than about half a point is ruled out.

### 4.3 Where in the label range does veg_pos differ? (veg_pos − ijepa per regime, core)

| regime | 1-shot | 5-shot | 10-shot | 25-shot | 50-shot | 100-shot | 10% | 100% |
|---|---|---|---|---|---|---|---|---|
| Δ (pp) | **−1.28** [−2.03, −0.50] | −0.62 | −0.39 | +0.07 | −0.02 | +0.06 | +0.05 | −0.10 |

- The only regime where the CI excludes zero is **1-shot**, and there veg_pos is **worse**.
- From 25 labels per class upward the two methods are indistinguishable.

### 4.4 Per crop class (veg_pos − ijepa, `core/stats/per_class_delta.csv`)

- Most classes are within ±1 pp with CIs spanning 0.
- **Soya (B33): −1.42 pp** [−2.70, −0.17].
- **Sugar beet (B22): +1.18 pp** [−0.01, +2.30], borderline.
- With 12 classes, about one "significant-looking" class is expected by chance, so these per-class results should not be over-interpreted.

### 4.5 The exploratory k-NN view: a hint of something, but under 1 pp

Exploratory and not part of the confirmatory family; treat as hypothesis-generating only.
- On k-NN, veg_pos beats pos_matched by **+0.99 pp** [+0.35, +1.64] and veg_neg by **+0.74 pp** [+0.13, +1.34].
- pos_matched is **−0.69 pp** below ijepa [−1.16, −0.22].
- **Interpretation:** content-aware placement organises nearest-neighbour structure slightly better than the same spatial layout without the content signal. Even so, it only gets back to ijepa's level.

---

## 5. Replications

### 5.1 New data split (splitB, 10 seeds)

Split B shares only 374 of ~2,380 test images with split A, so **84% of its test set is new**.

| | Δ (pp) | 95% CI | core value | agrees with core? |
|---|---|---|---|---|
| H1 veg_pos − ijepa | −0.20 | [−0.97, +0.57] | −0.09 | yes (equivalent) |
| H2 veg_pos − pos_matched | +0.22 | [−0.41, +0.86] | +0.37 | yes |
| H3 veg_pos − veg_neg | +0.37 | [−0.28, +0.97] | +0.42 | yes |
| H4 neg_matched − veg_neg | +0.65 | [+0.08, +1.20] | +0.52 | same direction and size; Holm p = 0.13 at n = 10 |
| E1 pos_matched − ijepa | −0.43 | [−0.90, +0.04] | −0.46 | yes |
| E2 neg_matched − ijepa | +0.08 | [−0.56, +0.71] | +0.02 | yes |
| H5 high − low | −1.01 | [−1.45, −0.54] | −0.90 | yes (reverse direction again) |

**The whole pattern replicates, including the H4 asymmetry and the reversed H5.** Absolute scores are also nearly identical (ijepa 52.6 vs 52.4).

### 5.2 Larger encoder (ViT-S, 10 seeds)

| | ijepa | veg_pos | Δ | 95% CI | seeds favouring veg_pos |
|---|---|---|---|---|---|
| primary endpoint | 55.8 | 54.3 | **−1.54** | [−2.26, −0.86] | **0/10** |
| 10-shot | 49.2 | 47.4 | −1.82 | [−2.70, −0.97] | 0/10 |
| 10% labels | 62.4 | 61.2 | −1.18 | [−1.86, −0.53] | 0/10 |
| 100% labels | 71.2 | 70.9 | −0.29 | [−1.28, +0.72] | 4/10 |

- **At larger capacity, vegetation guidance clearly hurts.** Every seed was worse on the primary endpoint.
- **It is not representation collapse:** veg_pos ViT-S has a slightly *higher* effective rank than ijepa (RankMe 128 vs 122), and training was stable (0 non-finite steps).
- **Likely reading:** a bigger model can exploit the full diversity of uniformly placed targets. Biasing targets toward vegetation makes the prediction task less diverse, so the representation is slightly less general.
- Moving from ViT-Tiny to ViT-S itself helps ijepa by +3.4 pp (52.4 → 55.8).

### 5.3 Pooled over the three confirmatory phases

- Random-effects meta-analysis of veg_pos − ijepa over core, splitB and ViT-S: **−0.60 pp** [−1.51, +0.32], one-sided p = 0.90.
- Heterogeneity is high (I² = 84%), driven by the ViT-S harm.
- **No setting shows a benefit.**

---

## 6. The bridge: why Stage B looked promising

The bridge phase re-runs **Stage B's exact protocol** (no context cap, 9,030 steps) on 10 new seeds.

| | Stage B (6 seeds) | Stage C bridge (10 new seeds) | pooled (16 seeds) | Stage C core (capped, 26k steps) |
|---|---|---|---|---|
| ijepa primary endpoint | 44.6 | 44.1 | – | 52.4 |
| veg_pos − ijepa | +0.89 pp | +0.42 pp [−0.27, +1.16] | **+0.60 pp [+0.14, +1.07], p = 0.017, 12/16 seeds** | **−0.09 pp** |
| veg_pos context tokens vs ijepa | 31.1 vs 28.2 | 31.1 vs 28.2 | – | 12.0 vs 12.0 |
| H5 tertiles (low / mid / high dispersion) | +1.07 / +0.73 / +2.11 | +0.14 / +0.46 / **+1.39** | – | **+0.67 / −0.21 / −0.23** |
| H5 high − low | +1.04 (exploratory) | **+1.25 [+0.72, +1.74], p = 0.003, 9/10 seeds** | – | −0.90 |

What this shows:
1. **Stage B was not a fluke or a bug.** The bridge reproduces Stage B's absolute numbers (44.1 vs 44.6) and the direction of its effect. Pooled over 16 seeds, the small veg_pos advantage is real under that protocol.
2. **Under Stage B's protocol the advantage concentrates on heterogeneous, row-crop scenes** (high ExG dispersion: maize, sugar beet), exactly as the exploratory Stage B analysis suggested.
3. **Under the controlled protocol** (equal context tokens, 3× longer training) **both the average advantage and the row-crop advantage disappear.** The row-crop pattern even reverses slightly.
4. **Which of the two changes removes the effect cannot be separated.** The bridge differs from core in both context budget and training length. The direct contrast (−0.38 pp [−1.09, +0.32]) is inconclusive, so the paper must say "the effect does not survive matched-budget, longer training" and not blame either factor alone.
   - In the uncapped bridge, veg_pos had **10% more context** tokens than ijepa (31.1 vs 28.2), because vegetation-seeking targets overlap more and leave more patches as context.
   - **The most plausible explanation is that the extra context was the cause.** It fits the evidence, but the experiment was not designed to prove it.

**Stage B's headline result:**
- **veg_pos − veg_neg = +3.97 pp** (6/6 seeds, Stage B's only "confirmed" result) **shrinks to +0.42 pp (n.s.)** in Stage C.
- In Stage B, veg_neg was −3.1 pp below ijepa. In Stage C it is only −0.50 pp.
- So Stage B's big veg_neg penalty was mostly an artefact of veg_neg's conditions there: fewer effective targets, and targets clustered at the top of the image.

---

## 7. Improvement over Stage B

| aspect | Stage B | Stage C | change |
|---|---|---|---|
| ijepa primary endpoint (ViT-Tiny) | 44.6 | 52.4 | **+7.8 pp** |
| ijepa 10-shot / 100% labels | 38.5 / 58.1 | 46.5 / 67.0 | +8.0 / +8.9 pp |
| best in-domain SSL (primary) | 45.5 (veg_pos) | 55.8 (ijepa, ViT-S) | +10.3 pp |
| SSL vs supervised-from-scratch at 10% labels | 51.1 vs 55.0 (SSL **behind**) | 58.8 vs 54.1 (SSL **ahead** by 4.7 pp) | reversed |
| seeds per arm | 6 | 20 (core), plus 10 + 10 + 10 | 3.3× |
| typical 95% CI half-width on a difference | ~±1.0 pp | ~±0.5 pp | 2× more precise |
| context tokens equal across arms? | No (27.0–31.1) | **Yes (12.0 for all)** | confound removed |
| position control for veg_neg | none | neg_matched (KL 0.0004) | new |
| learned-saliency baseline | none | attn (AttMask-style) | new |
| independent replication split | none | split B (84% new test images) | new |
| capacity check | none | ViT-S | new |
| strong external reference | none | DINOv2 (83.6) | new |
| plan frozen before main runs | no | yes (SHA-256, timestamped) | new |
| exact statistics | Wilcoxon exact only for n ≤ 16 | exact for any n; Holm; hierarchical bootstrap; TOST; random-effects meta-analysis | stronger |

**The +7.8 pp gain is from longer training (26k vs 9k steps), not from the cap.** In Phase 0 at equal steps, the cap cost 1.2 pp of k-NN.

---

## 8. Interpretation: why vegetation guidance does not help

Four observations fit together:
1. **The targets are already full of vegetation.** I-JEPA targets cover about half of each image (~49 of 100 patches after overlap removal), and LUCAS photos are close-ups of fields. Uniform targets therefore already include most vegetated patches. veg_pos can only make the targets **12% greener on average** (z +0.42). There is little room to add information.
2. **Removing vegetation from the targets does cost something, and it is the content that matters.**
   - veg_neg is −0.50 pp below ijepa, while neg_matched, with the same layout, is +0.02.
   - H4 (+0.52 pp, Holm p = 0.024) is the study's one confirmed effect, and it replicates in direction on split B (+0.65 pp).
   - So the model **does** learn useful crop features by predicting vegetated patches. Taking those targets away hurts; adding more beyond the uniform share gives nothing. **The benefit has a ceiling, and uniform masking already sits at it.**
3. **Concentrating targets also has a price.**
   - Fixed spatial layouts (pos_matched −0.46 pp) and greener-but-less-diverse targets (veg_pos at ViT-S −1.54 pp) both cost a little.
   - Neither is outside the ±1 pp equivalence margin at ViT-Tiny, but the direction is consistent: **less diverse targets → slightly less general features**, and the larger model shows it more clearly.
4. **Earlier apparent gains came from training conditions.**
   - Under short, uncapped training, veg_pos gets extra context and a small advantage, especially on row-crop scenes.
   - With equal context and longer training, uniform I-JEPA catches up completely.
   - Heuristic masking schemes are often evaluated in short-schedule pilots. This study shows such gains can be properties of the schedule, not of the heuristic.

These findings agree with prior general-domain work:
- **AttMask** found that masking the less salient (background-like) regions works poorly. Here, veg_neg is the worst ViT-Tiny arm on the primary endpoint (tied with pos_matched at 51.9–52.0), which agrees.
- **Semantic-guided masking papers** (SemMAE, ADIOS) report gains mainly in reconstruction-based MAE or at short schedules. In a feature-prediction JEPA with large target blocks, the room for such gains appears to be small.

---

## 9. How the numbers compare with the literature and the references

- **Supervised from scratch, 100% labels: 74.5 macro-F1** (ViT-Tiny, 10 seeds). This is close to the best model reported for the LUCAS crop photo collection by Yordanov et al. (2023, *Sensors*; macro-F1 ≈ 0.75). **But the splits differ**: ours is group-aware, which is usually harder. The numbers are a sanity check, not a head-to-head comparison.
- **Frozen DINOv2 ViT-S/14: 83.6** primary and 88.0 at 100% labels, with no fine-tuning. Large-scale generic pretraining beats every LUCAS-only model by ~28–31 pp on the primary endpoint. **The domain-specific masking question matters far less than pretraining scale.** This belongs in the paper as context, not as a competitor (one run; different resolution and pretraining data).
- **I-JEPA trained only on 11k LUCAS images reaches 52.4 (Tiny) and 55.8 (Small)** on the primary endpoint. It beats supervised-from-scratch at 10% labels (58.8 vs 54.1). The supervised model is an unmatched reference, so state this descriptively, not as "SSL beats supervised".

---

## 10. What is novel and publishable

**Novel contributions** (no prior work found that tests vegetation-index-guided target placement in JEPA):
1. **The first controlled evaluation of vegetation-index (ExG) guided target placement in I-JEPA.** It uses a factorial control design that separates *content* from *spatial position* and *polarity*: veg_pos / veg_neg plus their position-matched twins, with a learned-saliency baseline.
2. **An equivalence result, not just "no significance".** With 20 seeds and 90% power for a 0.9 pp effect, vegetation guidance is within ±1 pp of uniform masking (95% CI −0.59 to +0.47 pp). It replicates on an independent split and is harmful at ViT-S (−1.54 pp, 0/10 seeds better).
3. **A mechanism.**
   - Avoiding vegetation in targets costs ~0.5 pp because of content, not position (H4 confirmed, replicated in direction).
   - Seeking vegetation gives nothing because large uniform target blocks already cover it (target greenness only +12%).
4. **A methodological warning, with evidence.**
   - A pilot effect (Stage B: +0.89 pp vs ijepa, +3.97 pp vs veg_neg) is fully reproduced under its original protocol: +0.60 pp pooled over 16 seeds, p = 0.017, with the row-crop interaction p = 0.003.
   - It vanishes once context tokens are equalised and training is lengthened.
   - Masking papers rarely control the **context-token budget**. This study shows uneven budgets can manufacture effects of exactly the size usually reported.
5. **A reproducibility package.** The analysis plan was frozen by hash before the main runs, with exact tests, a hierarchical bootstrap, and per-run predictions and features saved.

**How to frame it.** "A well-controlled negative result plus a mechanism plus a methodology lesson."
- **Venues that explicitly welcome this:** the NeurIPS "I Can't Believe It's Not Better" workshop series, TMLR (which judges on correctness and evidence rather than positive results), and agriculture-vision or remote-sensing ML workshops.
- **For an applied-agriculture journal,** lead with the benchmark (SSL on LUCAS ground photos, the DINOv2 gap) and the practical advice: don't spend effort on vegetation-guided masking; use uniform masking, longer training, or a large pretrained model.

**Possible titles:**
- *"Does Masking the Crop Help? A Controlled Study of Vegetation-Guided Targets in I-JEPA"*
- *"Green Targets Don't Help: Context Budget, Not Content, Explains Vegetation-Guided Masking Gains in JEPA"*

The second is only acceptable if it's worded carefully, because the cap and training length are confounded.

---

## 11. Claims you may and may not make

**Allowed** (supported by the pre-registered tests):
- Vegetation-guided target placement is **equivalent** to uniform placement on average (±1 pp margin, core and split B).
- It is **worse** at ViT-S (−1.54 pp, CI excludes 0; one replication with 10 seeds).
- veg_neg's deficit comes from target **content**, not position (H4, Holm p = 0.024).
- A static spatial prior does not improve representations (E1, E2 equivalent).
- Under the Stage B protocol (uncapped, 9k steps), a small veg_pos advantage exists. This is exploratory: pooled n = 16, p = 0.017. It does not survive the controlled protocol.

**Not allowed:**
- "veg_pos improves I-JEPA" in any form for the controlled protocol.
- "The benefit depends on scene heterogeneity / row crops." H5 was not confirmed and is reversed in core and split B; the bridge result is exploratory.
- "The context cap alone explains Stage B" (it is confounded with training length).
- "SSL beats supervised learning" (unmatched reference), "ExG is the best vegetation index" (only ExG was tested), or describing LUCAS photos as aerial or satellite imagery (they are ground-level).
- "Significant" for any exploratory row (k-NN comparisons, per-class deltas, bridge).

---

## 12. Limitations

- **Content steering is modest.** The steering is limited by I-JEPA's large target blocks: z +0.42, +12% ExG. A design with smaller or more numerous target blocks might steer content harder and could behave differently.
- **Only ExG.** Only one vegetation index (ExG on RGB) and one dataset (LUCAS ground photos, 12 crops, 15,876 images) were tested.
- **The 12-token cap is low.** The context budget is far below standard I-JEPA. It was chosen to equalise arms and cost only 1.2 pp of k-NN in Phase 0, but results at a higher matched budget are untested.
- **Cap and training length are confounded** in the bridge contrast.
- **Small models.** Only ViT-Tiny and ViT-S were used, at 160 px and ≤ 26k steps. DINOv2 is a single deterministic reference run.

---

## 13. Where to look (files)

| what | file |
|---|---|
| headline report | `stageC_results/outputs_stageC/FINAL_REPORT.md` |
| all pre-registered tests, all phases | `stageC_results/outputs_stageC/consistency_table.csv`, `final_summary.json` |
| core tables (LaTeX-ready) | `stageC_results/outputs_stageC/core/paper_tables/*.tex` |
| forest plot of the primary comparisons | `core/figures/primary_endpoint_forest.png` |
| seed-level dots (each seed visible) | `core/figures/seed_level_dots.png` |
| manipulation check | `core/figures/manipulation_check.png` |
| H5 moderation by ExG dispersion | `core/figures/moderation_dispersion.png`, `bridge/figures/` |
| label-efficiency curves | `core/figures/label_efficiency_probe.png`, `label_efficiency_knn.png` |
| per-class differences | `core/figures/per_class_delta.png`, `core/stats/per_class_delta.csv` |
| Phase-0 decision and power | `phase0/stats/stageC_locked.json`, `power_analysis.json` |
| frozen plan | `phase0/stats/preregistration_stageC.json` (+ `.sha256`), `PREREG_CONFIRMED` |

Recommended main-paper figures:
1. A forest plot of all pre-registered comparisons across core, split B and ViT-S.
2. The bridge-vs-core panel (Stage B protocol vs controlled protocol, with the H5 tertiles).
3. A manipulation-check bar chart (target ExG z and spatial KL).
4. Label-efficiency curves with random_init, supervised and DINOv2 as reference lines.
