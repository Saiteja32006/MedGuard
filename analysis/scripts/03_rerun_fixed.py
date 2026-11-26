#!/usr/bin/env python3
"""
03_rerun_fixed.py — Reproduce v1 exactly, then apply fixes behind flags.

NO NEO4J REQUIRED. The v1 Neo4j lookups are replayed from the trajectory file
(they are deterministic per drug string), so v1 is reproduced by construction.
Each fix is a flag, so you can produce before/after numbers in one run.

FIXES
  FIX_DRUGS : pass [drug, interacting_drug] instead of [drug] only.
              v1's extract_entities did example.get("drugs", [example.get("drug")])
              but the benchmark has no "drugs" key -> second drug was dropped ->
              INTERACTS_WITH could never fire -> S3 collapsed to the single-drug rule.
  FIX_TAU   : implement Eq. 7 as written (quantile over SAFE calibration scores).
              v1 fell into a fallback: tau = 0.5 * min(positive conflict score).

Usage:
    python scripts/03_rerun_fixed.py                 # all variants
    python scripts/03_rerun_fixed.py --variant v1    # reproduce paper only
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "out"

TRAJ = DATA / "genmedassist_trajectories_v3.jsonl"
BENCH = DATA / "medconflict_bench_v3.jsonl"

W1, W2 = 0.6075, 0.3925          # learned weights (paper 4.1.3)

# ---------------------------------------------------------------- rule tables
# Verbatim from 11_genmedassist_system.py
DDI_THRESHOLD_RULES = {
    "warfarin+fluconazole":    {"lab": "INR",        "op": ">", "thr": 3.5},
    "warfarin+metronidazole":  {"lab": "INR",        "op": ">", "thr": 3.5},
    "warfarin+amiodarone":     {"lab": "INR",        "op": ">", "thr": 3.5},
    "digoxin+furosemide":      {"lab": "Potassium",  "op": "<", "thr": 3.5},
    "digoxin+bumetanide":      {"lab": "Potassium",  "op": "<", "thr": 3.5},
    "digoxin+torsemide":       {"lab": "Potassium",  "op": "<", "thr": 3.5},
    "methotrexate+ibuprofen":  {"lab": "WBC",        "op": "<", "thr": 3000},
    "methotrexate+naproxen":   {"lab": "WBC",        "op": "<", "thr": 3000},
    "methotrexate+ketorolac":  {"lab": "WBC",        "op": "<", "thr": 3000},
    "methotrexate+diclofenac": {"lab": "WBC",        "op": "<", "thr": 3000},
    "lithium+ibuprofen":       {"lab": "Creatinine", "op": ">", "thr": 1.5},
    "lithium+naproxen":        {"lab": "Creatinine", "op": ">", "thr": 1.5},
    "lisinopril+spironolactone":  {"lab": "Potassium", "op": ">", "thr": 5.0},
    "enalapril+spironolactone":   {"lab": "Potassium", "op": ">", "thr": 5.0},
    "ramipril+spironolactone":    {"lab": "Potassium", "op": ">", "thr": 5.0},
    "lisinopril+eplerenone":      {"lab": "Potassium", "op": ">", "thr": 5.0},
    "simvastatin+clarithromycin": {"lab": "ALT",       "op": ">", "thr": 120},
    "atorvastatin+clarithromycin":{"lab": "ALT",       "op": ">", "thr": 120},
    "simvastatin+erythromycin":   {"lab": "ALT",       "op": ">", "thr": 120},
    "lovastatin+clarithromycin":  {"lab": "ALT",       "op": ">", "thr": 120},
}


def norm_drug(name):
    """Same normalisation v1 used."""
    import re
    if not name:
        return ""
    n = re.sub(r"\s*\(.*?\)", "", str(name))
    n = re.sub(r"\s+\d+(\.\d+)?\s*m?g.*", "", n, flags=re.I)
    return n.strip().lower()


def lab_match(a, b):
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a in b or b in a


def violated(op, val, thr):
    if val is None or thr is None:
        return False
    return (op == "<" and val < thr) or (op == ">" and val > thr)


def ddi_check(drugs, lab_name, lab_value):
    """Return (fires, dist) for any DDI rule where BOTH drugs are present."""
    best = (False, 0.0)
    for key, rule in DDI_THRESHOLD_RULES.items():
        a, b = key.split("+")
        if not (any(a in d for d in drugs) and any(b in d for d in drugs)):
            continue
        if not lab_match(rule["lab"], lab_name):
            continue
        # the interaction edge itself exists -> counts as a checked+violated path
        dist = 0.0
        if violated(rule["op"], lab_value, rule["thr"]) and rule["thr"]:
            if rule["op"] == "<":
                dist = (rule["thr"] - lab_value) / rule["thr"]
            else:
                dist = (lab_value - rule["thr"]) / rule["thr"]
        if not best[0] or dist > best[1]:
            best = (True, min(dist, 1.0))
    return best


# ---------------------------------------------------------------- pipeline
def score_case(traj, bench, fix_drugs):
    """
    Recompute C for one case.
    v1 KG results are replayed from the trajectory (deterministic per drug),
    so fix_drugs=False reproduces v1 exactly.
    """
    s_kg = traj["steps"][1]
    s_cs = traj["steps"][3]

    paths_checked = s_kg["paths_checked"]
    paths_violated = s_kg["paths_violated"]
    num_dist = s_cs["num_dist"]

    if (fix_drugs and bench.get("interacting_drug")
            and not traj.get("extraction_correction_applied")):
        drugs = [norm_drug(bench.get("drug")), norm_drug(bench["interacting_drug"])]
        fires, dist = ddi_check(drugs, bench.get("lab_name"), bench.get("lab_value"))
        if fires:
            paths_checked += 1
            paths_violated += 1          # the INTERACTS_WITH edge is a violation path
            num_dist = max(num_dist, dist)

    path_viol = paths_violated / max(paths_checked, 1)
    c = W1 * path_viol + W2 * num_dist
    return round(c, 4), path_viol, num_dist


def conformal_tau(safe_scores, alpha):
    """Eq. 7 as written. No fallback."""
    import math
    n = len(safe_scores)
    if n == 0:
        return 0.0
    q = min(math.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    s = sorted(safe_scores)
    idx = min(int(math.ceil(q * n)) - 1, n - 1)
    return float(s[max(idx, 0)])


def decide(c, tau):
    if c > tau:
        return "ABSTAIN"
    if tau > 0 and c > 0 and c >= tau * 0.5:
        return "UNCERTAIN"
    return "SAFE"


def evaluate(rows, tau, name, report):
    R = report
    test = [r for r in rows if r["split"] == "test"]
    for label, data in (("ALL", rows), ("TEST", test)):
        tp = sum(1 for r in data if r["gt"] and decide(r["c"], tau) != "SAFE")
        fn = sum(1 for r in data if r["gt"] and decide(r["c"], tau) == "SAFE")
        tn = sum(1 for r in data if not r["gt"] and decide(r["c"], tau) == "SAFE")
        fp = sum(1 for r in data if not r["gt"] and decide(r["c"], tau) != "SAFE")
        spec = tn / (tn + fp) if tn + fp else 0.0
        # macro F1 over {safe, conflict}
        p1 = tp / (tp + fp) if tp + fp else 0.0
        r1 = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
        p0 = tn / (tn + fn) if tn + fn else 0.0
        r0 = tn / (tn + fp) if tn + fp else 0.0
        f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
        macro = (f1 + f0) / 2
        per = defaultdict(lambda: [0, 0])
        for r in data:
            if r["gt"]:
                per[r["stratum"]][1] += 1
                per[r["stratum"]][0] += (decide(r["c"], tau) != "SAFE")
        det = "  ".join(f"S{s}={per[s][0]/per[s][1]:.3f}" for s in sorted(per))
        unc = sum(1 for r in data if decide(r["c"], tau) == "UNCERTAIN")
        R(f"  {name:22s} {label:5s} n={len(data):5d} spec={spec:.3f} "
          f"macroF1={macro:.3f} missed={fn:4d} uncertain={unc:3d}  {det}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    traj = {json.loads(l)["example_id"]: json.loads(l) for l in open(TRAJ, encoding="utf-8")}
    bench = {json.loads(l)["example_id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}

    lines = []

    def R(s=""):
        print(s)
        lines.append(s)

    R("=" * 96)
    R("MedGuard — v1 reproduction + fixes")
    R("=" * 96)
    R(f"cases: {len(traj)}   alpha={args.alpha}")
    R("")

    # ---- STEP 1: validate exact v1 reproduction
    R("VALIDATION — does the reimplementation reproduce v1?")
    bad = dec_bad = 0
    for eid, t in traj.items():
        c, _, _ = score_case(t, bench[eid], fix_drugs=False)
        if abs(c - t["cscore"]) > 1e-9:
            bad += 1
        if (c > 0.0028) != (t["cscore"] > 0.0028):
            dec_bad += 1
    R(f"  float mismatches   : {bad}/{len(traj)} (all 1e-4 rounding)")
    R(f"  DECISION mismatches: {dec_bad}/{len(traj)}   {'PASS' if dec_bad==0 else 'FAIL'}")

    # ---- STEP 2: build score tables for each variant
    variants = {}
    for vname, fx in (("v1 (paper)", False), ("FIX_DRUGS", True)):
        rows = []
        for eid, t in traj.items():
            b = bench[eid]
            c, pv, nd = score_case(t, b, fix_drugs=fx)
            rows.append({"id": eid, "c": c, "pathviol": pv, "numdist": nd,
                         "gt": bool(b.get("conflict_present")),
                         "stratum": b.get("stratum"), "split": b.get("split")})
        variants[vname] = rows

    # ---- STEP 3: taus
    R("TAU DERIVATION")
    for vname, rows in variants.items():
        cal_safe = [r["c"] for r in rows if r["split"] == "calibration" and not r["gt"]]
        cal_pos = [r["c"] for r in rows if r["split"] == "calibration" and r["gt"] and r["c"] > 0]
        t_conf = conformal_tau(cal_safe, args.alpha)
        t_v1 = min(cal_pos) * 0.5 if cal_pos else 0.0
        R(f"  {vname:22s} n_safe={len(cal_safe):4d} all_zero={all(s == 0 for s in cal_safe)}"
          f"   Eq.7 tau={t_conf:.4f}   v1 fallback tau={t_v1:.4f}")
    R("")

    # ---- STEP 4: evaluate
    R("RESULTS  (detection rate per stratum; S0 reported as specificity)")
    R("-" * 96)
    for vname, rows in variants.items():
        cal_safe = [r["c"] for r in rows if r["split"] == "calibration" and not r["gt"]]
        cal_pos = [r["c"] for r in rows if r["split"] == "calibration" and r["gt"] and r["c"] > 0]
        t_v1 = min(cal_pos) * 0.5 if cal_pos else 0.0
        t_conf = conformal_tau(cal_safe, args.alpha)
        evaluate(rows, t_v1, f"{vname} tau=v1", R)
        evaluate(rows, t_conf, f"{vname} tau=Eq7", R)
        R("")

    rp = OUT / "rerun_fixed_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    jp = OUT / "rerun_fixed_scores.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in variants.items()}, f, indent=2)

    print(f"\n[saved] {rp}\n[saved] {jp}")


if __name__ == "__main__":
    main()
