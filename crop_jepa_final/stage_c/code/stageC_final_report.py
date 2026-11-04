#!/usr/bin/env python
"""Stage C final cross-phase report. Reads ONLY saved artifacts (per-phase gate JSONs and seed-level CSVs, the frozen
pre-registration, Stage B's seed-level CSV) and writes outputs_stageC/FINAL_REPORT.md + final_summary.json +
consistency_table.csv. No model is loaded and nothing is re-trained."""
import datetime, json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(os.environ.get("OUT_ROOT", "./outputs_stageC"))
PREREG = Path(os.environ.get("CJ_PREREG_FILE", ROOT / "phase0/stats/preregistration_stageC.json"))
STAGEB_SEED_LEVEL = os.environ.get("STAGEB_SEED_LEVEL", "")
CONFIRMED = Path(os.environ.get("CJ_PREREG_CONFIRMED", ROOT / "PREREG_CONFIRMED"))


def wilcoxon_exact(d, alternative):
    """Exact signed-rank p (same dynamic programme as crop_jepa_stageC.py)."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    r = rankdata(np.abs(d))
    r2 = np.rint(2 * r).astype(np.int64)
    total = int(r2.sum())
    dist = np.zeros(total + 1)
    dist[0] = 1.0
    for x in r2:
        new = dist.copy()
        new[x:] += dist[:total + 1 - x]
        dist = new
    probs = dist / dist.sum()
    w2 = int(round(2 * r[d > 0].sum()))
    if alternative == "greater":
        return float(probs[w2:].sum())
    c = total / 2
    return float(min(1.0, probs[np.abs(np.arange(total + 1) - c) >= abs(w2 - c) - 1e-9].sum()))


def seed_boot_ci(d, B=10000, seed=7):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def paired_diffs(sl, a, b):
    """Per-seed probe primary-endpoint differences a - b in pp from a seed_level_primary.csv table."""
    p = sl[sl["eval"] == "probe"].pivot_table(index="seed", columns="arm", values="primary_endpoint")
    if a not in p or b not in p:
        return pd.Series(dtype=float)
    return (100 * (p[a] - p[b])).dropna()


def main():
    if not PREREG.exists():
        sys.exit(f"pre-registration {PREREG} not found")
    reg = json.load(open(PREREG, encoding="utf-8"))
    phases = list(reg["phases"])
    gates, seed_tabs = {}, {}
    for ph in phases:
        g, s = ROOT / ph / "stats" / f"gate_G1_stageC_{ph}.json", ROOT / ph / "stats" / "seed_level_primary.csv"
        if g.exists():
            gates[ph] = json.load(open(g, encoding="utf-8"))
        if s.exists():
            seed_tabs[ph] = pd.read_csv(s)
    missing = [ph for ph in phases if ph not in gates]
    if missing:
        print(f"WARNING: phases without results: {missing} (the report covers the rest)")

    # ---- consistency table of every pre-registered comparison per phase ----
    rows = []
    for ph, g in gates.items():
        for fam in ("confirmatory", "equivalence"):
            for r in g.get(fam, []):
                rows.append({"phase": ph, "family": r["family"], "hypothesis": r.get("hypothesis", ""), "comparison": r["comparison"],
                             "n_seeds": r["n_seeds"], "delta_pp": r["diff_pp"], "ci95_lo_pp": r["ci95_lo_pp"],
                             "ci95_hi_pp": r["ci95_hi_pp"], "holm_p": r.get("holm_p"), "status": r["status"]})
        h5 = g.get("moderation_H5") or {}
        if h5:
            rows.append({"phase": ph, "family": h5["family"], "hypothesis": "H5", "comparison": "(veg_pos - ijepa) high - low",
                         "n_seeds": h5["n_seeds"], "delta_pp": h5["high_minus_low_pp"], "ci95_lo_pp": h5["ci95_lo_pp"],
                         "ci95_hi_pp": h5["ci95_hi_pp"], "holm_p": h5["wilcoxon_p_greater"], "status": h5["status"]})
    cons = pd.DataFrame(rows)
    cons.to_csv(ROOT / "consistency_table.csv", index=False)

    # ---- random-effects pooling of veg_pos - ijepa over the 26k-step settings (summary R) ----
    settings = []
    for ph in ("core", "splitB", "vits"):
        if ph in seed_tabs:
            d = paired_diffs(seed_tabs[ph], "veg_pos", "ijepa")
            if len(d) >= 2:
                settings.append({"setting": ph, "n": len(d), "mean_pp": float(d.mean()), "se_pp": float(d.std(ddof=1) / math.sqrt(len(d)))})
    meta = {}
    if len(settings) >= 2:
        y = np.array([s["mean_pp"] for s in settings])
        v = np.array([s["se_pp"] ** 2 for s in settings])
        w = 1 / v
        fixed = float((w * y).sum() / w.sum())
        Q = float((w * (y - fixed) ** 2).sum())
        dfree = len(y) - 1
        C = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
        ws = 1 / (v + tau2)
        pooled = float((ws * y).sum() / ws.sum())
        se = float(math.sqrt(1 / ws.sum()))
        meta = {"method": "DerSimonian-Laird random effects", "settings": settings, "pooled_pp": pooled,
                "ci95_pp": [pooled - 1.96 * se, pooled + 1.96 * se], "z": pooled / se, "p_one_sided": float(norm.sf(pooled / se)),
                "tau2": tau2, "Q": Q, "I2": float(max(0.0, (Q - dfree) / Q)) if Q > 0 else 0.0}

    # ---- bridge: Stage B protocol (no cap, 9,030 steps), Stage C seeds 100-109 pooled with Stage B seeds 0-5 ----
    bridge = {}
    if "bridge" in seed_tabs:
        db = paired_diffs(seed_tabs["bridge"], "veg_pos", "ijepa")
        pooled = list(db.values)
        sources = {"stageC_bridge_seeds": [int(s) for s in db.index]}
        if STAGEB_SEED_LEVEL and Path(STAGEB_SEED_LEVEL).exists():
            sb = pd.read_csv(STAGEB_SEED_LEVEL)
            dsb = paired_diffs(sb[sb["family"] == "core"], "veg_pos", "ijepa")
            pooled += list(dsb.values)
            sources["stageB_seeds"] = [int(s) for s in dsb.index]
        if len(pooled) >= 2:
            bridge = {"protocol": "Stage B (per-batch token equalisation, 9,030 steps)", **sources, "n": len(pooled),
                      "mean_pp": float(np.mean(pooled)), "ci95_pp": seed_boot_ci(pooled),
                      "wilcoxon_p_greater": wilcoxon_exact(pooled, "greater"),
                      "wilcoxon_p_two_sided": wilcoxon_exact(pooled, "two-sided"),
                      "seeds_positive": int((np.array(pooled) > 0).sum()), "note": "exploratory"}
        if "core" in seed_tabs:
            dc = paired_diffs(seed_tabs["core"], "veg_pos", "ijepa")
            common = sorted(set(dc.index) & set(db.index))
            if len(common) >= 2:
                diff = dc.loc[common].values - db.loc[common].values
                bridge["length_and_cap_contrast"] = {
                    "description": "(veg_pos - ijepa at 26k steps, fixed token cap) - (same at 9k steps, Stage B protocol), "
                                   "paired by seed; confounds training length with the token protocol (exploratory)",
                    "n": len(common), "mean_pp": float(diff.mean()), "ci95_pp": seed_boot_ci(diff),
                    "wilcoxon_p_two_sided": wilcoxon_exact(diff, "two-sided")}

    # ---- split overlap ----
    overlap = None
    a_csv, b_csv = ROOT / "core" / "splits.csv", ROOT / "splitB" / "splits.csv"
    if a_csv.exists() and b_csv.exists():
        sa, sb_ = pd.read_csv(a_csv, keep_default_na=False), pd.read_csv(b_csv, keep_default_na=False)
        ta, tb = set(sa.loc[sa["split"] == "test", "path"]), set(sb_.loc[sb_["split"] == "test", "path"])
        overlap = {"split_A_test": len(ta), "split_B_test": len(tb), "shared_test_images": len(ta & tb),
                   "share_of_split_B_test_new": 1 - len(ta & tb) / max(1, len(tb))}

    # ---- references (core) ----
    refs = {}
    sm = ROOT / "core" / "stats" / "summary_macro_f1.csv"
    if sm.exists():
        s = pd.read_csv(sm)
        s = s[(s["eval"].isin(["probe", "supervised"])) & (s["regime"].isin(["10pct", "100pct"]))]
        refs = {f"{r['label']}|{r['eval']}|{r['regime']}": round(100 * r["mean"], 2) for _, r in s.iterrows()}

    core = gates.get("core", {})
    dec = core.get("decision", {})
    allowed, forbidden = [], ["Never use 'significant' for exploratory rows.",
                              "Do not describe LUCAS photographs as aerial imagery (they are ground-level survey photos).",
                              "Do not claim SSL beats supervised training: supervised from scratch is an unmatched upper reference.",
                              "Do not claim ExG is the best vegetation index (only ExG was tested)."]
    code = dec.get("primary")
    if code in ("P-A", "P-A'"):
        allowed.append("Vegetation-aware target placement improves I-JEPA linear-probe performance on LUCASVision"
                       + (" (below the +0.5 pp SESOI)" if code == "P-A'" else "") + ".")
    elif code == "P-B":
        allowed.append("Vegetation-aware placement is equivalent to uniform placement on average (within ±1.0 pp).")
        forbidden.append("Do not claim a benefit of veg_pos over ijepa.")
    elif code == "P-D":
        allowed.append("Vegetation-aware placement is harmful relative to uniform placement.")
    else:
        forbidden.append("Do not claim a benefit or an equivalence for veg_pos vs ijepa; report the bounded interval.")
    if dec.get("polarity"):
        allowed.append(f"Polarity mechanism: {dec['polarity']}.")
    if dec.get("spatial_prior", "").startswith("E1 equivalent"):
        allowed.append("A static spatial prior (pos_matched) is equivalent to uniform placement (replicates Stage B).")
    h5 = core.get("moderation_H5") or {}
    if h5.get("status") == "confirmed":
        allowed.append("The veg_pos benefit is larger on heterogeneous (high ExG-dispersion) scenes (H5). Note: same test "
                       "images as Stage B; fresh pretraining seeds.")
    elif h5:
        forbidden.append("Do not claim the benefit depends on scene heterogeneity (H5 not confirmed).")

    summary = {"created": datetime.datetime.now().isoformat(), "preregistration": str(PREREG),
               "registration_link": CONFIRMED.read_text().strip() if CONFIRMED.exists() else None,
               "core_decision": dec, "meta_analysis_R": meta, "bridge": bridge, "split_overlap": overlap,
               "references_core_pct": refs, "phases_missing": missing, "allowed_claims": allowed, "forbidden_claims": forbidden}
    with open(ROOT / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    def fmt(x, nd=2):
        return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)

    L = [f"# Crop-JEPA Stage C — final report", "", f"Generated {summary['created']} from saved artifacts only.", "",
         f"Pre-registration: `{PREREG}` (registered: {summary['registration_link']})", "",
         "## 1. Core decision (pre-registered decision matrix)", "", f"**{code}** — {dec.get('primary_text', 'n/a')}", "",
         f"- Polarity: {dec.get('polarity', 'n/a')}", f"- Spatial prior: {dec.get('spatial_prior', 'n/a')}",
         f"- Moderation: {dec.get('moderation', 'n/a')}", "", "## 2. Every pre-registered comparison, by phase", ""]
    if len(cons):
        L.append("| phase | family | hyp. | comparison | n | Δ (pp) | 95% CI | p (Holm / H5) | status |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in cons.iterrows():
            L.append(f"| {r['phase']} | {r['family']} | {r['hypothesis']} | {r['comparison']} | {r['n_seeds']} | {r['delta_pp']:+.2f} | "
                     f"[{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}] | {r['holm_p'] if r['holm_p'] == r['holm_p'] else '—'} | {r['status']} |")
    L += ["", "## 3. Replication summary (veg_pos − ijepa, random effects over core / split B / ViT-S)", ""]
    if meta:
        L.append(f"Pooled {meta['pooled_pp']:+.2f} pp, 95% CI [{meta['ci95_pp'][0]:+.2f}, {meta['ci95_pp'][1]:+.2f}], "
                 f"one-sided p {meta['p_one_sided']:.4g}, I² {100 * meta['I2']:.0f}%")
        for s in meta["settings"]:
            L.append(f"- {s['setting']}: n = {s['n']}, mean {s['mean_pp']:+.2f} pp (SE {s['se_pp']:.2f})")
    else:
        L.append("not enough settings finished")
    L += ["", "## 4. Bridge to Stage B (exploratory)", ""]
    if bridge:
        L.append(f"Stage B protocol, pooled n = {bridge.get('n')}: veg_pos − ijepa {fmt(bridge.get('mean_pp'))} pp, 95% CI "
                 f"{[round(x, 2) for x in bridge.get('ci95_pp', [])]}, Wilcoxon p (greater) {bridge.get('wilcoxon_p_greater', float('nan')):.4g}, "
                 f"{bridge.get('seeds_positive')}/{bridge.get('n')} seeds positive")
        if "length_and_cap_contrast" in bridge:
            c = bridge["length_and_cap_contrast"]
            L.append(f"- 26k-capped vs 9k-Stage-B effect contrast: {c['mean_pp']:+.2f} pp, 95% CI {[round(x, 2) for x in c['ci95_pp']]} "
                     "(confounds length and token protocol)")
    L += ["", "## 5. Data", "", f"Split overlap: {overlap}", "", "## 6. References (core, %)", ""]
    L += [f"- {k}: {v}" for k, v in refs.items()]
    L += ["", "## 7. Allowed claims", ""] + [f"- {c}" for c in allowed] + ["", "## 8. Claims NOT allowed", ""] + [f"- {c}" for c in forbidden]
    L += ["", "## 9. Integrity", "", f"- phases without results: {missing or 'none'}",
          f"- deviations: see `{ROOT / 'DEVIATIONS.md'}`" + ("" if (ROOT / "DEVIATIONS.md").exists() else " (none logged)"),
          "- every number above is recomputable from the per-phase `stats/` folders and per-run `eval.json` / `preds.npz`."]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
