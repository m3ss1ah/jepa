# Follow-up: 2×2 of context-token cap × training length (post hoc, exploratory)

Stage C changed two things at once compared with Stage B: it gave every arm a **cap of 12 context tokens**, and it trained **3× longer** (26,058 vs 9,030 steps). Under those rules, veg_pos's small Stage B advantage disappeared. This follow-up runs the two missing combinations to see which change was responsible.

- **Design:** ijepa and veg_pos, seeds 100–109, split A, ViT-Tiny/16 at 160 px. The two new cells were 40 runs, all finished with 0 errors.
- **Frozen plan:** the plan, the analysis rules and the code hash were frozen in `results/followup_plan.json` (sha256 `24d653d3…`) at 2026-11-04 13:05, when 0 follow-up runs existed.
- **Status:** designed after the Stage C results were known. Everything here is **exploratory**: report estimates with CIs, never "significant".

| | 9,030 steps | 26,058 steps |
|---|---|---|
| **no cap** (Stage B rules, ~28 / 31 context tokens) | Stage C bridge | **uncap26k** (new) |
| **cap K = 12** (Stage C rules) | **cap9k** (new) | Stage C core, seeds 100–109 |

Full tables: [`results/FOLLOWUP_2x2_REPORT.md`](results/FOLLOWUP_2x2_REPORT.md), `results/followup_2x2_contrasts.csv` and `results/followup_2x2.json`.

## Results (linear probe, primary endpoint, macro-F1 %, mean over seeds 100–109)

| cell | ijepa | veg_pos | d = veg_pos − ijepa [95% CI] | seeds favouring veg_pos |
|---|---|---|---|---|
| uncap / 9k | 44.12 | 44.54 | +0.42 [−0.14, +1.01] | 7/10 |
| cap / 9k | 44.38 | 44.78 | +0.40 [−0.32, +1.10] | 8/10 |
| uncap / 26k | 49.49 | 49.95 | +0.46 [−0.21, +1.06] | 7/10 |
| cap / 26k | 52.37 | 52.40 | +0.04 [−0.46, +0.57] | 5/10 |

| contrast on d | Δd (pp) [95% CI] |
|---|---|
| cap alone, at 9k | −0.01 [−0.53, +0.59] |
| longer training alone, uncapped | +0.04 [−0.67, +0.71] |
| cap at 26k | −0.42 [−1.29, +0.43] |
| longer training with the cap | −0.36 [−1.00, +0.33] |
| interaction (cap × length) | −0.41 [−1.15, +0.30] |

## Conclusions

1. **Neither change on its own removes the veg_pos advantage.** The advantage is about +0.4 pp with the cap alone, with longer training alone, and under the Stage B rules. It falls to about 0 only when **both** are applied. This is the pre-stated reading "neither factor alone is identified". With 10 seeds, the interaction (−0.41 pp) is not detectable on the probe. It is detectable on k-NN: −1.37 [−2.43, −0.29].
2. **Unequal context did not create the Stage B gain.** With equal context (cap/9k) veg_pos still leads by +0.40. With unequal context and long training (uncap/26k) it leads by +0.46. This **corrects** the earlier suggestion that veg_pos's ~10% extra context explained Stage B.
3. **The advantage is small in every cell.** It is at most about +0.4–0.5 pp, and no single cell's CI excludes 0. The Stage C conclusion stands: no meaningful benefit from vegetation-guided targets.
4. **The row-crop pattern is an early-training effect.** The veg_pos advantage on high-ExG-dispersion (row-crop) scenes at 9k steps is +1.25 pp. Longer training removes it in both protocols: −3.00 [−3.78, −2.16] uncapped (0/10 seeds) and −1.80 [−2.78, −0.79] capped. The cap at 9k changes it only a little: −0.73 [−1.53, −0.02].
5. **New finding: the small context budget itself improves I-JEPA at long training.** ijepa at 26k steps scores 52.37 with the cap vs 49.49 without, so **+2.87 pp [+2.42, +3.33]**, better on 10/10 seeds. veg_pos gains +2.45 pp, also 10/10. At 9k steps the cap makes no difference (+0.26 [−0.37, +0.85]). So the +7.8 pp gain from Stage B to Stage C splits into about +5.4 pp from longer training and about +2.6 to +2.9 pp more from the cap once training is long. Uncapped runs are not collapsed (RankMe 83–86, 0 non-finite steps). Seeing less context makes the prediction task harder, and with enough training that yields better features. This is post hoc, on one split and one model size; it needs its own confirmation before any strong claim.

## Layout

- `code/` is the follow-up pipeline exactly as it ran. `crop_jepa_followup.py` has sha256 `bc5adf39…` and is generated from the Stage C script by `make_followup_script.py`.
- `results/` holds `cap9k/` and `uncap26k/` (stats, paper tables, figures, per-run eval, preds and train logs), `logs/`, the frozen plan and the 2×2 report. Checkpoints and bundles are excluded.
