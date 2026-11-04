#!/usr/bin/env python
"""Usage: python check_complete_stageC.py <phase|all>
Expected runs come from the FROZEN pre-registration (its 'phases' plan), so this script and the experiment can never
disagree. Exit 0 = every expected run of the phase(s) has eval.json. The optional DINOv2 reference is reported but
never required. Also writes state/progress (finished evaluations + logged epochs) for the no-progress guard."""
import json, os, sys
from pathlib import Path

root = Path(os.environ.get("OUT_ROOT", "./outputs_stageC"))
prereg = Path(os.environ.get("CJ_PREREG_FILE", root / "phase0/stats/preregistration_stageC.json"))
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if not prereg.exists():
    print("no pre-registration yet (Phase 0 not finished)")
    sys.exit(1)
plan = json.load(open(prereg, encoding="utf-8"))["phases"]
phases = list(plan) if which == "all" else [which]
ok_all, n_eval, n_ep = True, 0, 0
for ph in phases:
    p = plan[ph]
    runs = root / ph / "runs"
    expected = [f"{a}_s{s}" for s in p["seeds"] for a in p["arms"]]
    expected += [f"random_init_s{s}" for s in p["seeds"] if "random_init" in p["baselines"]]
    expected += [f"supervised_s{s}" for s in p["sup_seeds"] if "supervised" in p["baselines"]]
    done = [n for n in expected if (runs / n / "eval.json").exists()]
    missing = [n for n in expected if n not in done]
    errors = sorted(e.parent.name for e in runs.glob("*/error.txt")) if runs.exists() else []
    extra = " | dinov2 reference: " + ("done" if (runs / "dinov2_s0" / "eval.json").exists() else "not run") if ph == "core" else ""
    print(f"[{ph}] FINISHED {len(done)}/{len(expected)} expected runs{extra}")
    if missing:
        print(f"[{ph}] missing: " + ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""))
    if errors:
        print(f"[{ph}] runs with error.txt: " + ", ".join(errors))
    ok_all &= not missing
for ph in plan:
    runs = root / ph / "runs"
    if runs.exists():
        n_eval += len(list(runs.glob("*/eval.json")))
        for f in runs.glob("*/train_log.csv"):
            try:
                n_ep += max(0, sum(1 for _ in open(f)) - 1)
            except OSError:
                pass
(root / "state").mkdir(parents=True, exist_ok=True)
(root / "state" / "progress").write_text(str(n_eval * 1000000 + n_ep))
sys.exit(0 if ok_all else 1)
