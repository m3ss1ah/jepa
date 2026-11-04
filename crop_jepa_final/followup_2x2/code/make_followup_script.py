#!/usr/bin/env python
"""Builds crop_jepa_followup.py from ../crop_jepa_stageC/crop_jepa_stageC.py by a few exact, checked text patches.
Everything else (model, masking, training, evaluation, statistics) is byte-identical to the Stage C script.
Run:  python make_followup_script.py"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "crop_jepa_stageC" / "crop_jepa_stageC.py"
DST = HERE / "crop_jepa_followup.py"
STAGEC_SHA = "859588be2b085008602897f4a5d10210ce25cbaeb850846c63866fbf7870324e"

FOLLOWUP_BLOCK = '''# ---- POST-HOC FOLLOW-UP (added after the Stage C results were known; exploratory, disclosed as such) ----
# Completes a 2x2 design (context-token cap x training length) for ijepa and veg_pos on split A, seeds 100-109:
#   capped / 26k = Stage C core (seeds 100-109)   uncapped / 9k = Stage C bridge
#   capped / 9k  = cap9k (new)                    uncapped / 26k = uncap26k (new)
FOLLOWUP_PLAN = {
    "cap9k": {"split": "A", "profile": "big_gpu", "arms": ["ijepa", "veg_pos"], "seeds": REPL_SEEDS, "sup_seeds": [],
              "baselines": [], "steps": 9000, "ctx_cap": "auto",
              "role": "post-hoc follow-up 2x2: Stage C token cap (K from Phase 0) at the Stage B length (9,030 steps)"},
    "uncap26k": {"split": "A", "profile": "big_gpu", "arms": ["ijepa", "veg_pos"], "seeds": REPL_SEEDS, "sup_seeds": [],
                 "baselines": [], "steps": 26000, "ctx_cap": "none",
                 "role": "post-hoc follow-up 2x2: Stage B token protocol (no cap) at the Stage C length (26,058 steps)"},
}
PLAN_ALL = {**PHASE_PLAN, **FOLLOWUP_PLAN}
IS_FOLLOWUP = PHASE in FOLLOWUP_PLAN
DECISION_MATRIX = {
'''

VERIFY_BLOCK = '''FOLLOWUP_FREEZE = _env("CJ_FOLLOWUP_FREEZE")
OUT_ROOT_FOLLOWUP = _env("CJ_FOLLOWUP_ROOT")
FOLLOWUP_RULES = {
    "unit": "seed (100-109); d = veg_pos - ijepa on the linear-probe primary endpoint (pp), per cell",
    "contrasts": {"cap_at_9k": "d(cap9k) - d(bridge)", "cap_at_26k": "d(core) - d(uncap26k)",
                  "length_uncapped": "d(uncap26k) - d(bridge)", "length_capped": "d(core) - d(cap9k)",
                  "interaction": "cap_at_26k - cap_at_9k"},
    "statistics": "mean, 95% seed-bootstrap CI (B = 10000), exact two-sided Wilcoxon; all EXPLORATORY (post hoc)",
    "reading": ("the cap explains the loss of the Stage B effect if cap_at_9k has a 95% CI below 0 and length_uncapped "
                "includes 0; training length explains it if length_uncapped has a CI below 0 and cap_at_9k includes 0; "
                "both if both are below 0; neither alone is identified otherwise"),
}


def verify_followup_freeze(g, code, stagec_prereg_sha):
    """Follow-up phases: the frozen Stage C plan and confirmation are intact (checked by the caller); the follow-up
    plan and the follow-up code are frozen by their own record, written once before the first follow-up run."""
    if not FOLLOWUP_FREEZE:
        raise RuntimeError("CJ_FOLLOWUP_FREEZE is not set (path of the follow-up freeze record).")
    p = Path(FOLLOWUP_FREEZE)
    sha_path = p.with_name(p.name + ".sha256")
    rec = {"what": "Crop-JEPA post-hoc follow-up (2x2: token cap x training length); exploratory, planned after the "
                   "Stage C results were known", "plan": FOLLOWUP_PLAN, "analysis": FOLLOWUP_RULES,
           "followup_code_sha256": code, "stageC_code_sha256": g.get("code_sha256"),
           "stageC_preregistration_sha256": stagec_prereg_sha, "stageC_token_cap": g["token_budget"]["ctx_tokens_cap"]}
    if not p.exists():
        n_runs = len(list(Path(OUT_ROOT_FOLLOWUP).glob("*/runs/*/eval.json"))) if OUT_ROOT_FOLLOWUP else None
        save_json({**rec, "created": datetime.datetime.now().isoformat(), "followup_runs_finished_at_freeze": n_runs}, p)
        save_text(f"{sha256_file(p)}  {p.name}\\n", sha_path)
        print(f"FOLLOW-UP plan frozen: {p} (sha256 {sha256_file(p)[:12]}; {n_runs} follow-up runs existed)")
    if not sha_path.exists() or sha_path.read_text().split()[0] != sha256_file(p):
        raise RuntimeError(f"{p} does not match its frozen hash {sha_path}: the follow-up plan was modified.")
    old = load_json(p)
    if old.get("followup_code_sha256") != code:
        raise RuntimeError(f"follow-up code hash {code} differs from the frozen {old.get('followup_code_sha256')}")
    if json.loads(json.dumps(FOLLOWUP_PLAN, default=_json_default)) != old["plan"]:
        raise RuntimeError("FOLLOWUP_PLAN in the code differs from the frozen follow-up plan.")
    if old.get("stageC_preregistration_sha256") != stagec_prereg_sha:
        raise RuntimeError("the Stage C pre-registration is not the one the follow-up was frozen against.")
    print(f"Stage C pre-registration intact (sha256 {stagec_prereg_sha[:12]}); follow-up plan verified "
          f"({p.name}, frozen {old.get('created')}).")
    return g


if RUN_MODE in ("prepare", "full"):
    check_phase_plan()
    GLOBAL_PREREG = verify_global_prereg()'''

PATCHES = [
    # 1. the two follow-up phases are allowed
    ('assert PHASE in ("core", "splitB", "vits", "bridge"), PHASE',
     'assert PHASE in ("core", "splitB", "vits", "bridge", "cap9k", "uncap26k"), PHASE'),
    # 2. follow-up plan, kept separate from the frozen Stage C PHASE_PLAN (which stays unchanged)
    ('DECISION_MATRIX = {\n', FOLLOWUP_BLOCK),
    # 3. the phase-plan check knows the follow-up phases
    ('    plan = PHASE_PLAN[PHASE]\n    problems = []', '    plan = PLAN_ALL[PHASE]\n    problems = []'),
    # 4. follow-up verification: Stage C plan intact (checks above it) + the follow-up's own freeze record
    ('    code = sha256_file(THIS_FILE) if THIS_FILE else None\n    if code != g.get("code_sha256"):',
     '    code = sha256_file(THIS_FILE) if THIS_FILE else None\n    if IS_FOLLOWUP:\n'
     '        return verify_followup_freeze(g, code, sha_path.read_text().split()[0])\n'
     '    if code != g.get("code_sha256"):'),
    ('if RUN_MODE in ("prepare", "full"):\n    check_phase_plan()\n    GLOBAL_PREREG = verify_global_prereg()',
     VERIFY_BLOCK),
    # 5. report labels / roles for the follow-up phases
    ('            "bridge": "BRIDGE (exploratory; Holm within phase shown for reference)"}[PHASE_EFF]',
     '            "bridge": "BRIDGE (exploratory; Holm within phase shown for reference)",\n'
     '            "cap9k": "FOLLOW-UP 2x2 (post hoc, exploratory)", "uncap26k": "FOLLOW-UP 2x2 (post hoc, exploratory)"}[PHASE_EFF]'),
    ('''DECISION["text"] = f"[{PHASE_EFF}: {PHASE_PLAN[PHASE_EFF]['role']}] " + DECISION["text"]''',
     '''DECISION["text"] = f"[{PHASE_EFF}: {PLAN_ALL[PHASE_EFF]['role']}] " + DECISION["text"]'''),
    ('G1 = {"phase": PHASE_EFF, "role": PHASE_PLAN[PHASE_EFF]["role"], "decision": DECISION,',
     'G1 = {"phase": PHASE_EFF, "role": PLAN_ALL[PHASE_EFF]["role"], "decision": DECISION,'),
]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == STAGEC_SHA, f"{SRC} is not the Stage C script that ran on the cluster (sha256 {sha})"
    s = raw.decode("utf-8")
    for old, new in PATCHES:
        n = s.count(old)
        assert n == 1, f"patch target found {n} times (expected 1):\n{old[:120]}"
        s = s.replace(old, new)
    header = ("# Crop-JEPA FOLLOW-UP (post-hoc 2x2: token cap x training length). Generated by make_followup_script.py\n"
              f"# from crop_jepa_stageC.py (sha256 {STAGEC_SHA}); only the follow-up phases and their freeze differ.\n")
    DST.write_text(header + s, encoding="utf-8", newline="\n")
    print(f"{DST.name} written ({len(PATCHES)} patches applied; sha256 {hashlib.sha256(DST.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()
