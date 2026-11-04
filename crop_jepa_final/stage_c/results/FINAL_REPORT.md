# Crop-JEPA Stage C — final report

Generated 2026-11-04T02:48:20.669037 from saved artifacts only.

Pre-registration: `outputs_stageC/phase0/stats/preregistration_stageC.json` (registered: INTERNAL FREEZE - not externally registered (no OSF registration).
pre-registration ./outputs_stageC/phase0/stats/preregistration_stageC.json sha256 c6774dac49cc7f3d29036d4f7e4a8ec5f54ff4d9012adcaa6dee5b19ff539343, frozen 2026-11-02 21:04:23
main phases released 2026-11-02 21:04:24 by job 2146 after the Phase-0 sanity gate)

## 1. Core decision (pre-registered decision matrix)

**P-B** — veg_pos - ijepa 90% CI inside (-1.0, +1.0) pp (TOST, alpha 0.05) -> 'equivalent to uniform placement on average' | veg_pos - ijepa = -0.09 pp, 95% CI [-0.59, +0.47], Holm p = 0.8256, 8/20 seeds positive

- Polarity: unresolved (see H3 / H4 rows)
- Spatial prior: E1 equivalent: a static spatial prior does not change representations
- Moderation: H5 not confirmed (high - low = -0.90 pp, 95% CI [-1.38, -0.36], p = 0.9979)

## 2. Every pre-registered comparison, by phase

| phase | family | hyp. | comparison | n | Δ (pp) | 95% CI | p (Holm / H5) | status |
|---|---|---|---|---|---|---|---|---|
| core | PRIMARY (pre-registered, Holm) | H1 | veg_pos - ijepa | 20 | -0.09 | [-0.59, +0.47] | 0.8255949020385742 | equivalent |
| core | PRIMARY (pre-registered, Holm) | H2 | veg_pos - pos_matched | 20 | +0.37 | [-0.14, +0.92] | 0.14290618896484375 | equivalent |
| core | PRIMARY (pre-registered, Holm) | H3 | veg_pos - veg_neg | 20 | +0.42 | [-0.10, +0.95] | 0.09558391571044922 | equivalent |
| core | PRIMARY (pre-registered, Holm) | H4 | neg_matched - veg_neg | 20 | +0.52 | [+0.09, +0.97] | 0.024158477783203125 | greater (confirmed) |
| core | EQUIVALENCE (pre-registered) | E1 | pos_matched - ijepa | 20 | -0.46 | [-0.87, -0.07] | — | equivalent |
| core | EQUIVALENCE (pre-registered) | E2 | neg_matched - ijepa | 20 | +0.02 | [-0.39, +0.43] | — | equivalent |
| core | MODERATION (pre-registered) | H5 | (veg_pos - ijepa) high - low | 20 | -0.90 | [-1.38, -0.36] | 0.997889518737793 | not confirmed |
| splitB | REPLICATION R1 (pre-registered, Holm within phase) | H1 | veg_pos - ijepa | 10 | -0.20 | [-0.97, +0.57] | 0.75390625 | equivalent |
| splitB | REPLICATION R1 (pre-registered, Holm within phase) | H2 | veg_pos - pos_matched | 10 | +0.22 | [-0.41, +0.86] | 0.625 | equivalent |
| splitB | REPLICATION R1 (pre-registered, Holm within phase) | H3 | veg_pos - veg_neg | 10 | +0.37 | [-0.28, +0.97] | 0.3486328125 | equivalent |
| splitB | REPLICATION R1 (pre-registered, Holm within phase) | H4 | neg_matched - veg_neg | 10 | +0.65 | [+0.08, +1.20] | 0.12890625 | inconclusive |
| splitB | EQUIVALENCE (splitB, replication) | E1 | pos_matched - ijepa | 10 | -0.43 | [-0.90, +0.04] | — | equivalent |
| splitB | EQUIVALENCE (splitB, replication) | E2 | neg_matched - ijepa | 10 | +0.08 | [-0.56, +0.71] | — | equivalent |
| splitB | moderation (splitB, replication) | H5 | (veg_pos - ijepa) high - low | 10 | -1.01 | [-1.45, -0.54] | 0.9970703125 | not confirmed |
| vits | REPLICATION R3 (pre-registered, Holm within phase) | H1 | veg_pos - ijepa | 10 | -1.54 | [-2.26, -0.86] | 1.0 | lower |
| vits | moderation (vits, replication) | H5 | (veg_pos - ijepa) high - low | 10 | -1.72 | [-2.16, -1.26] | 1.0 | not confirmed |
| bridge | BRIDGE (exploratory; Holm within phase shown for reference) | H1 | veg_pos - ijepa | 10 | +0.42 | [-0.27, +1.16] | 0.1376953125 | inconclusive |
| bridge | moderation (bridge, replication) | H5 | (veg_pos - ijepa) high - low | 10 | +1.25 | [+0.72, +1.74] | 0.0029296875 | confirmed |

## 3. Replication summary (veg_pos − ijepa, random effects over core / split B / ViT-S)

Pooled -0.60 pp, 95% CI [-1.51, +0.32], one-sided p 0.9002, I² 84%
- core: n = 20, mean -0.09 pp (SE 0.23)
- splitB: n = 10, mean -0.20 pp (SE 0.39)
- vits: n = 10, mean -1.54 pp (SE 0.35)

## 4. Bridge to Stage B (exploratory)

Stage B protocol, pooled n = 16: veg_pos − ijepa +0.60 pp, 95% CI [0.14, 1.07], Wilcoxon p (greater) 0.01677, 12/16 seeds positive
- 26k-capped vs 9k-Stage-B effect contrast: -0.38 pp, 95% CI [-1.09, 0.32] (confounds length and token protocol)

## 5. Data

Split overlap: {'split_A_test': 2381, 'split_B_test': 2380, 'shared_test_images': 374, 'share_of_split_B_test_new': 0.8428571428571429}

## 6. References (core, %)

- attn|probe|100pct: 66.55
- attn|probe|10pct: 58.53
- dinov2|probe|100pct: 88.05
- dinov2|probe|10pct: 86.15
- ijepa|probe|100pct: 66.99
- ijepa|probe|10pct: 58.8
- neg_matched|probe|100pct: 66.95
- neg_matched|probe|10pct: 58.75
- pos_matched|probe|100pct: 66.74
- pos_matched|probe|10pct: 58.39
- random_init|probe|100pct: 29.2
- random_init|probe|10pct: 22.4
- veg_neg|probe|100pct: 66.61
- veg_neg|probe|10pct: 58.48
- veg_pos|probe|100pct: 66.88
- veg_pos|probe|10pct: 58.85

## 7. Allowed claims

- Vegetation-aware placement is equivalent to uniform placement on average (within ±1.0 pp).
- Polarity mechanism: unresolved (see H3 / H4 rows).
- A static spatial prior (pos_matched) is equivalent to uniform placement (replicates Stage B).

## 8. Claims NOT allowed

- Never use 'significant' for exploratory rows.
- Do not describe LUCAS photographs as aerial imagery (they are ground-level survey photos).
- Do not claim SSL beats supervised training: supervised from scratch is an unmatched upper reference.
- Do not claim ExG is the best vegetation index (only ExG was tested).
- Do not claim a benefit of veg_pos over ijepa.
- Do not claim the benefit depends on scene heterogeneity (H5 not confirmed).

## 9. Integrity

- phases without results: none
- deviations: see `outputs_stageC/DEVIATIONS.md` (none logged)
- every number above is recomputable from the per-phase `stats/` folders and per-run `eval.json` / `preds.npz`.
