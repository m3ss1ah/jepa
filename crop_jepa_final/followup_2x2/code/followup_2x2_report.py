#!/usr/bin/env python
"""Post-hoc follow-up: 2x2 analysis (context-token cap x training length) of veg_pos - ijepa, seeds 100-109, split A.

    cell        token protocol   steps    source
    cap/26k     cap K = 12       26,058   Stage C core (seeds 100-109 of 100-119)
    uncap/9k    none (Stage B)    9,030   Stage C bridge
    cap/9k      cap K = 12        9,030   follow-up cap9k
    uncap/26k   none (Stage B)   26,058   follow-up uncap26k

Reads only saved artifacts (seed_level_primary.csv, moderation.csv, token_budget.csv); works with whichever cells
exist. Writes outputs_followup/FOLLOWUP_2x2_REPORT.md, followup_2x2.json, followup_2x2_cells.csv,
followup_2x2_contrasts.csv. Every result here is EXPLORATORY (planned after the Stage C results were known)."""
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(os.environ.get("OUT_ROOT", "./outputs_followup"))
STAGEC = Path(os.environ.get("STAGEC_OUT", os.path.expanduser("~/crop_jepa_stageC/outputs_stageC")))
SEEDS = list(range(100, 110))
B = 10000
CELLS = {   # cell -> (phase folder root, phase name, capped?, steps)
    "cap/26k": (STAGEC, "core", True, 26058),
    "uncap/9k": (STAGEC, "bridge", False, 9030),
    "cap/9k": (OUT, "cap9k", True, 9030),
    "uncap/26k": (OUT, "uncap26k", False, 26058),
}
CONTRASTS = {   # name -> (plus cell, minus cell, plain-language meaning); per seed, on d = veg_pos - ijepa
    "cap_at_9k": ("cap/9k", "uncap/9k", "effect of the token cap on the veg_pos advantage, at the Stage B length"),
    "cap_at_26k": ("cap/26k", "uncap/26k", "effect of the token cap on the veg_pos advantage, at the Stage C length"),
    "length_uncapped": ("uncap/26k", "uncap/9k", "effect of longer training on the veg_pos advantage, without the cap"),
    "length_capped": ("cap/26k", "cap/9k", "effect of longer training on the veg_pos advantage, with the cap"),
}


def wilcoxon_two_sided(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, 0):
        return float("nan")
    for kw in ({"method": "exact"}, {"mode": "exact"}, {}):
        try:
            return float(stats.wilcoxon(x, alternative="two-sided", **kw).pvalue)
        except TypeError:
            continue
        except ValueError:
            return float("nan")
    return float("nan")


def summarise(x, rng):
    x = np.asarray(x, float)
    boots = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_pp": float(x.mean()), "ci95_lo_pp": float(lo), "ci95_hi_pp": float(hi), "n_seeds": int(len(x)),
            "seeds_positive": int((x > 0).sum()), "wilcoxon_p_two_sided": wilcoxon_two_sided(x)}


def seed_table(root, phase, ev):
    f = root / phase / "stats" / "seed_level_primary.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[(d["eval"] == ev) & d["arm"].isin(["ijepa", "veg_pos"]) & d["seed"].isin(SEEDS)]
    w = d.pivot_table(index="seed", columns="arm", values="primary_endpoint")
    if set(w.columns) != {"ijepa", "veg_pos"} or sorted(w.index) != SEEDS or w.isna().any().any():
        return None
    return 100 * w.sort_index()


def h5_per_seed(root, phase):
    f = root / phase / "stats" / "moderation.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    r = m[m["comparison"] == "veg_pos - ijepa"]
    if not len(r):
        return None
    r = r.iloc[0]
    vals = json.loads(r["seed_high_minus_low_pp"])
    # values are in sorted-seed order; core has seeds 100-119, the 2x2 uses 100-109 (the first ten)
    if len(vals) not in (10, 20) or int(r["n_seeds"]) != len(vals):
        return None
    return {"per_seed": np.array(vals[:10], float), "low": float(r["delta_low_pp"]), "mid": float(r["delta_mid_pp"]),
            "high": float(r["delta_high_pp"])}


def tokens(root, phase):
    f = root / phase / "paper_tables" / "token_budget.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f).set_index("label")
    return {a: float(t.loc[a, "ctx_tokens"]) for a in ("ijepa", "veg_pos") if a in t.index}


def fmt(s):
    return (f"{s['mean_pp']:+.2f} [{s['ci95_lo_pp']:+.2f}, {s['ci95_hi_pp']:+.2f}]" if s else "n/a")


def ci_below(s):
    return bool(s) and s["ci95_hi_pp"] < 0


def ci_has0(s):
    return bool(s) and s["ci95_lo_pp"] <= 0 <= s["ci95_hi_pp"]


def main():
    rng = np.random.default_rng(2027)
    res = {"created": datetime.datetime.now().isoformat(), "status": "EXPLORATORY post-hoc follow-up", "seeds": SEEDS}
    freeze = OUT / "followup_plan.json"
    if freeze.exists():
        res["followup_plan_sha256"] = (OUT / "followup_plan.json.sha256").read_text().split()[0]
    cell_rows, con_rows, per_eval = [], [], {}
    for ev in ("probe", "knn"):
        D, cells = {}, {}
        for c, (root, ph, capped, steps) in CELLS.items():
            w = seed_table(root, ph, ev)
            if w is None:
                continue
            d = (w["veg_pos"] - w["ijepa"]).to_numpy()
            D[c] = d
            cells[c] = {"phase": ph, "capped": capped, "steps": steps, "ijepa_pp": float(w["ijepa"].mean()),
                        "veg_pos_pp": float(w["veg_pos"].mean()), **summarise(d, rng), **{f"ctx_tokens_{a}": v
                        for a, v in tokens(root, ph).items()}}
            cell_rows.append({"eval": ev, "cell": c, **cells[c]})
        cons = {}
        for name, (a, b, meaning) in CONTRASTS.items():
            if a in D and b in D:
                cons[name] = {"cells": f"{a} - {b}", "meaning": meaning, **summarise(D[a] - D[b], rng)}
                con_rows.append({"eval": ev, "contrast": name, **cons[name]})
        if all(c in D for c in CELLS):
            cap_main = ((D["cap/9k"] - D["uncap/9k"]) + (D["cap/26k"] - D["uncap/26k"])) / 2
            len_main = ((D["uncap/26k"] - D["uncap/9k"]) + (D["cap/26k"] - D["cap/9k"])) / 2
            inter = (D["cap/26k"] - D["uncap/26k"]) - (D["cap/9k"] - D["uncap/9k"])
            for name, x, meaning in (("cap_main_effect", cap_main, "average effect of the cap (over both lengths)"),
                                     ("length_main_effect", len_main, "average effect of 26k vs 9k steps (over both protocols)"),
                                     ("interaction", inter, "does the cap act differently at 26k than at 9k")):
                cons[name] = {"cells": "2x2", "meaning": meaning, **summarise(x, rng)}
                con_rows.append({"eval": ev, "contrast": name, **cons[name]})
        per_eval[ev] = {"cells": cells, "contrasts": cons}
    res["probe"], res["knn"] = per_eval["probe"], per_eval["knn"]

    # H5 (row-crop moderation) per cell, and its 2x2 contrasts
    H = {c: h5_per_seed(root, ph) for c, (root, ph, _, _) in CELLS.items()}
    H = {c: h for c, h in H.items() if h is not None}
    h5 = {"cells": {c: {"low_pp": h["low"], "mid_pp": h["mid"], "high_pp": h["high"],
                        **summarise(h["per_seed"], rng)} for c, h in H.items()}, "contrasts": {}}
    for name, (a, b, meaning) in CONTRASTS.items():
        if a in H and b in H:
            h5["contrasts"][name] = {"cells": f"{a} - {b}", **summarise(H[a]["per_seed"] - H[b]["per_seed"], rng)}
    res["h5_high_minus_low"] = h5

    # pre-stated reading (frozen in followup_plan.json -> analysis.reading)
    pc = per_eval["probe"]["contrasts"]
    cap9, len_u = pc.get("cap_at_9k"), pc.get("length_uncapped")
    if cap9 and len_u:
        if ci_below(cap9) and ci_has0(len_u):
            verdict = "the token cap explains the loss of the Stage B veg_pos advantage (training length does not)"
        elif ci_below(len_u) and ci_has0(cap9):
            verdict = "longer training explains the loss of the Stage B veg_pos advantage (the token cap does not)"
        elif ci_below(cap9) and ci_below(len_u):
            verdict = "both the token cap and longer training reduce the veg_pos advantage"
        else:
            verdict = "neither factor alone is identified (both contrasts include 0, or the pattern is mixed)"
    elif cap9:
        verdict = ("partial (uncap26k missing): the cap " + ("reduces" if ci_below(cap9) else "does not detectably reduce")
                   + " the veg_pos advantage at the Stage B length")
    else:
        verdict = "not enough cells for a verdict"
    res["verdict"] = verdict

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "followup_2x2.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(cell_rows).to_csv(OUT / "followup_2x2_cells.csv", index=False)
    pd.DataFrame(con_rows).to_csv(OUT / "followup_2x2_contrasts.csv", index=False)

    L = ["# Crop-JEPA follow-up: 2x2 (context-token cap x training length)", "",
         f"Generated {res['created']} from saved artifacts. **Exploratory, post hoc**: designed after the Stage C results "
         "were known; the follow-up plan was frozen by SHA-256 before its first run"
         + (f" (`followup_plan.json`, sha256 {res.get('followup_plan_sha256', '')[:12]}…)." if freeze.exists() else "."),
         "", "Unit: seed (100-109, split A, ViT-Tiny/16 @160). d = veg_pos − ijepa on the linear-probe primary "
         "endpoint (mean test macro-F1 over 10-shot, 25-shot, 10 %), in percentage points. 95 % CI = seed bootstrap "
         f"(B = {B}); p = exact two-sided Wilcoxon.", "", f"**Pre-stated reading:** {verdict}.", "",
         "## 1. The four cells (probe)", "",
         "| cell | source | ctx tokens ijepa / veg_pos | ijepa | veg_pos | d = veg_pos − ijepa | seeds + | p |", "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        s = per_eval["probe"]["cells"].get(c)
        if not s:
            L.append(f"| {c} | {CELLS[c][1]} | – | – | – | not available | – | – |"); continue
        tok = f"{s.get('ctx_tokens_ijepa', float('nan')):.1f} / {s.get('ctx_tokens_veg_pos', float('nan')):.1f}"
        L.append(f"| {c} | {s['phase']} | {tok} | {s['ijepa_pp']:.2f} | {s['veg_pos_pp']:.2f} | {fmt(s)} | "
                 f"{s['seeds_positive']}/{s['n_seeds']} | {s['wilcoxon_p_two_sided']:.3f} |")
    L += ["", "## 2. Contrasts (probe; per seed, on d)", "", "| contrast | cells | meaning | Δd (pp) [95 % CI] | seeds + | p |",
          "|---|---|---|---|---|---|"]
    for name, s in per_eval["probe"]["contrasts"].items():
        L.append(f"| {name} | {s['cells']} | {s['meaning']} | {fmt(s)} | {s['seeds_positive']}/{s['n_seeds']} | "
                 f"{s['wilcoxon_p_two_sided']:.3f} |")
    L += ["", "## 3. Same on the k-NN endpoint (secondary)", "", "| cell | d (pp) [95 % CI] | | contrast | Δd (pp) [95 % CI] |",
          "|---|---|---|---|---|"]
    kc, kk = per_eval["knn"]["cells"], list(per_eval["knn"]["contrasts"].items())
    for i in range(max(len(CELLS), len(kk))):
        c = list(CELLS)[i] if i < len(CELLS) else ""
        left = f"| {c} | {fmt(kc.get(c))} |" if c else "| | |"
        right = f" | {kk[i][0]} | {fmt(kk[i][1])} |" if i < len(kk) else " | | |"
        L.append(left + right)
    L += ["", "## 4. Row-crop moderation (H5: d on high- minus low-ExG-dispersion test images)", "",
          "| cell | low | mid | high | high − low [95 % CI] | seeds + |", "|---|---|---|---|---|---|"]
    for c in CELLS:
        s = h5["cells"].get(c)
        L.append(f"| {c} | {s['low_pp']:+.2f} | {s['mid_pp']:+.2f} | {s['high_pp']:+.2f} | {fmt(s)} | "
                 f"{s['seeds_positive']}/{s['n_seeds']} |" if s else f"| {c} | – | – | – | not available | – |")
    for name, s in h5["contrasts"].items():
        L.append(f"| *{name}* ({s['cells']}) | | | | {fmt(s)} | {s['seeds_positive']}/{s['n_seeds']} |")
    L += ["", "## 5. How to read this", "",
          "- `cap_at_9k` isolates the token cap at the Stage B length: if it is clearly negative, giving every arm the "
          "same context budget is what removes veg_pos's Stage B advantage.",
          "- `length_uncapped` isolates training length under the Stage B protocol: if it is clearly negative, the "
          "advantage is an early-training effect that fades with longer training.",
          "- `interaction`: whether the cap matters differently at 26k than at 9k steps.",
          "- n = 10 seeds per cell: a 95 % CI that includes 0 means 'not detected', not 'no effect'. All rows are "
          "exploratory; do not call them significant in the paper, report estimates and CIs.", ""]
    (OUT / "FOLLOWUP_2x2_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
