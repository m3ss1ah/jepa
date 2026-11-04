#!/usr/bin/env python
"""Prints how many of the expected runs have finished. Exit 0 = all expected runs have eval.json, 1 = not yet.
Also lists runs that crashed (error.txt) so you can see why."""
import os, sys
from pathlib import Path

out = Path(os.environ.get("CJ_OUTPUT_DIR", "./outputs_stageB"))
seeds = [s.strip() for s in os.environ.get("CJ_SEEDS", "0,1,2,3,4,5").split(",") if s.strip()]
arms = [a.strip() for a in os.environ.get("CJ_ARMS", "ijepa,veg_pos,veg_neg,pos_matched").split(",") if a.strip()]
base = [b.strip() for b in os.environ.get("CJ_BASELINES", "random_init,supervised").split(",") if b.strip()]
t2 = [s.strip() for s in os.environ.get("CJ_TIER2_SEEDS", "0,1,2").split(",") if s.strip()]
flag = lambda n: os.environ.get(n, "0").strip().lower() in ("1", "true", "yes", "on")

expected = []
for s in seeds:
    expected += [f"{a}_s{s}" for a in arms]
    expected += [f"{b}_s{s}" for b in base if b in ("random_init", "supervised")]
if "dinov2" in base:
    expected.append("dinov2_s0")
if flag("CJ_ENABLE_ATTN"):
    expected += [f"attn_s{s}" for s in t2 if "attn" not in arms]
if flag("CJ_ENABLE_TAU_SWEEP"):
    expected += [f"veg_pos_tau{t}_s{s}" for s in t2 for t in ("0.25", "1")]
if flag("CJ_ENABLE_SIZE_AXIS"):
    expected += [f"{a}_frac{f}_s{s}" for s in t2 for f in ("0.1", "0.3") for a in ("ijepa", "veg_pos")]
if flag("CJ_ENABLE_LEJEPA"):
    expected += [f"lejepa_{a}_s{s}" for s in t2 for a in ("uniform", "veg_pos", "veg_neg", "pos_matched")]

runs = out / "runs"
done = [n for n in expected if (runs / n / "eval.json").exists()]
missing = [n for n in expected if n not in done]
errors = sorted(p.parent.name for p in runs.glob("*/error.txt")) if runs.exists() else []
# progress signature = finished evaluations + all logged training epochs (grows whenever any work is done)
n_eval = len(list(runs.glob("*/eval.json"))) if runs.exists() else 0
n_ep = 0
for f in (runs.glob("*/train_log.csv") if runs.exists() else []):
    try:
        n_ep += max(0, sum(1 for _ in open(f)) - 1)
    except OSError:
        pass
(out / "state").mkdir(parents=True, exist_ok=True)
(out / "state" / "progress").write_text(str(n_eval * 1000000 + n_ep))
print(f"FINISHED {len(done)}/{len(expected)} expected runs")
if missing:
    print("missing:", ", ".join(missing[:40]) + (" ..." if len(missing) > 40 else ""))
if errors:
    print("runs with error.txt (see outputs/runs/<name>/error.txt):", ", ".join(errors))
sys.exit(0 if not missing else 1)
