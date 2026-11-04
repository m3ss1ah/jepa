# Crop-JEPA follow-up: 2x2 (context-token cap x training length)

Generated 2026-11-04T15:18:59.112824 from saved artifacts. **Exploratory, post hoc**: designed after the Stage C results were known; the follow-up plan was frozen by SHA-256 before its first run (`followup_plan.json`, sha256 24d653d3790d…).

Unit: seed (100-109, split A, ViT-Tiny/16 @160). d = veg_pos − ijepa on the linear-probe primary endpoint (mean test macro-F1 over 10-shot, 25-shot, 10 %), in percentage points. 95 % CI = seed bootstrap (B = 10000); p = exact two-sided Wilcoxon.

**Pre-stated reading:** neither factor alone is identified (both contrasts include 0, or the pattern is mixed).

## 1. The four cells (probe)

| cell | source | ctx tokens ijepa / veg_pos | ijepa | veg_pos | d = veg_pos − ijepa | seeds + | p |
|---|---|---|---|---|---|---|---|
| cap/26k | core | 12.0 / 12.0 | 52.37 | 52.40 | +0.04 [-0.46, +0.57] | 5/10 | 1.000 |
| uncap/9k | bridge | 28.2 / 31.1 | 44.12 | 44.54 | +0.42 [-0.14, +1.01] | 7/10 | 0.275 |
| cap/9k | cap9k | 12.0 / 12.0 | 44.38 | 44.78 | +0.40 [-0.32, +1.10] | 8/10 | 0.322 |
| uncap/26k | uncap26k | 28.2 / 31.1 | 49.49 | 49.95 | +0.46 [-0.21, +1.06] | 7/10 | 0.131 |

## 2. Contrasts (probe; per seed, on d)

| contrast | cells | meaning | Δd (pp) [95 % CI] | seeds + | p |
|---|---|---|---|---|---|
| cap_at_9k | cap/9k - uncap/9k | effect of the token cap on the veg_pos advantage, at the Stage B length | -0.01 [-0.53, +0.59] | 3/10 | 0.625 |
| cap_at_26k | cap/26k - uncap/26k | effect of the token cap on the veg_pos advantage, at the Stage C length | -0.42 [-1.29, +0.43] | 4/10 | 0.492 |
| length_uncapped | uncap/26k - uncap/9k | effect of longer training on the veg_pos advantage, without the cap | +0.04 [-0.67, +0.71] | 5/10 | 0.846 |
| length_capped | cap/26k - cap/9k | effect of longer training on the veg_pos advantage, with the cap | -0.36 [-1.00, +0.33] | 4/10 | 0.322 |
| cap_main_effect | 2x2 | average effect of the cap (over both lengths) | -0.22 [-0.80, +0.44] | 4/10 | 0.432 |
| length_main_effect | 2x2 | average effect of 26k vs 9k steps (over both protocols) | -0.16 [-0.71, +0.43] | 3/10 | 0.695 |
| interaction | 2x2 | does the cap act differently at 26k than at 9k | -0.41 [-1.15, +0.30] | 4/10 | 0.432 |

## 3. Same on the k-NN endpoint (secondary)

| cell | d (pp) [95 % CI] | | contrast | Δd (pp) [95 % CI] |
|---|---|---|---|---|
| cap/26k | +0.42 [-0.18, +1.09] | | cap_at_9k | +1.10 [+0.24, +2.09] |
| uncap/9k | -0.39 [-1.14, +0.33] | | cap_at_26k | -0.27 [-1.39, +0.84] |
| cap/9k | +0.71 [-0.32, +1.69] | | length_uncapped | +1.08 [+0.16, +1.98] |
| uncap/26k | +0.69 [-0.13, +1.45] | | length_capped | -0.29 [-1.13, +0.69] |
| | | | cap_main_effect | +0.42 [-0.36, +1.36] |
| | | | length_main_effect | +0.40 [-0.28, +1.18] |
| | | | interaction | -1.37 [-2.43, -0.29] |

## 4. Row-crop moderation (H5: d on high- minus low-ExG-dispersion test images)

| cell | low | mid | high | high − low [95 % CI] | seeds + |
|---|---|---|---|---|---|
| cap/26k | +0.67 | -0.21 | -0.23 | -1.28 [-1.81, -0.77] | 1/10 |
| uncap/9k | +0.14 | +0.46 | +1.39 | +1.25 [+0.71, +1.74] | 9/10 |
| cap/9k | +0.34 | +0.32 | +0.86 | +0.52 [-0.43, +1.44] | 7/10 |
| uncap/26k | +1.68 | +1.36 | -0.07 | -1.75 [-2.49, -0.92] | 2/10 |
| *cap_at_9k* (cap/9k - uncap/9k) | | | | -0.73 [-1.53, -0.02] | 3/10 |
| *cap_at_26k* (cap/26k - uncap/26k) | | | | +0.47 [-0.41, +1.32] | 7/10 |
| *length_uncapped* (uncap/26k - uncap/9k) | | | | -3.00 [-3.78, -2.16] | 0/10 |
| *length_capped* (cap/26k - cap/9k) | | | | -1.80 [-2.78, -0.79] | 2/10 |

## 5. How to read this

- `cap_at_9k` isolates the token cap at the Stage B length: if it is clearly negative, giving every arm the same context budget is what removes veg_pos's Stage B advantage.
- `length_uncapped` isolates training length under the Stage B protocol: if it is clearly negative, the advantage is an early-training effect that fades with longer training.
- `interaction`: whether the cap matters differently at 26k than at 9k steps.
- n = 10 seeds per cell: a 95 % CI that includes 0 means 'not detected', not 'no effect'. All rows are exploratory; do not call them significant in the paper, report estimates and CIs.

