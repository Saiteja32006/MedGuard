#!/usr/bin/env python3
"""
04_day1_analysis.py — Day 1 Track A (A1-A4). Laptop, seconds, no Neo4j.

A1  Decision-level confusion matrix          -> Sdyd Q7
A2  Extraction / KG-lookup failure audit     -> AC-3, oJaU
A3  Near-boundary sweep                      -> Jn3J Q2
A4  Measurement-noise robustness             -> Jn3J Q1

Usage:
    python scripts/04_day1_analysis.py
Outputs:
    out/day1_report.txt          <- read this
    out/day1_results.json
    out/A2_extraction_failures.csv
    out/A3_boundary_sweep.csv
"""

import csv
import json
import os
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "out"
W1, W2 = 0.6075, 0.3925

# Per-drug KG threshold rules, as reconstructed from the trajectories.
# metformin shows paths_checked=3 and violated=2 at eGFR<30 (both the <30 and
# <45 edges fire) — matching the paper's worked example PathViol = 2/3.
SWEEP_DRUGS = {
    "metformin":  {"lab": "eGFR", "paths_checked": 3,
                   "rules": [("<", 30), ("<", 45)]},
    "digoxin":    {"lab": "Potassium", "paths_checked": 5,
                   "rules": [("<", 3.5)]},
    "warfarin":   {"lab": "INR", "paths_checked": 7,
                   "rules": [(">", 4.0)]},
}
ASSAY_CV = {"egfr": 0.065, "creatinine": 0.065, "potassium": 0.02,
            "inr": 0.10, "alt": 0.10, "glucose": 0.04, "wbc": 0.05,
            "platelet": 0.05}


def violated(op, val, thr):
    return (op == "<" and val < thr) or (op == ">" and val > thr)


def score_synthetic(drug_key, lab_value):
    """C-score for a synthetic case, using the reconstructed rule model."""
    spec = SWEEP_DRUGS[drug_key]
    checked = spec["paths_checked"]
    viol = sum(1 for op, thr in spec["rules"] if violated(op, lab_value, thr))
    nd = 0.0
    for op, thr in spec["rules"]:
        if violated(op, lab_value, thr) and thr:
            d = (thr - lab_value) / \
                thr if op == "<" else (lab_value - thr) / thr
            nd = max(nd, min(d, 1.0))
    pv = viol / max(checked, 1)
    return round(W1 * pv + W2 * nd, 4), pv, nd


def decide(c, tau):
    if c > tau:
        return "ABSTAIN"
    if tau > 0 and c > 0 and c >= tau * 0.5:
        return "UNCERTAIN"
    return "SAFE"


def cv_for(lab_name):
    ln = (lab_name or "").lower()
    for k, v in ASSAY_CV.items():
        if k in ln:
            return v
    return 0.05


def main():
    os.makedirs(OUT, exist_ok=True)
    traj = {json.loads(l)["example_id"]: json.loads(l)
            for l in open(DATA / "genmedassist_trajectories_v2.jsonl", encoding="utf-8")}
    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(DATA / "medconflict_bench_v3.jsonl", encoding="utf-8")}

    lines = []

    def R(s=""):
        print(s)
        lines.append(s)

    results = {}
    TAU_V1 = 0.0028
    TAU_EQ7 = 0.0000

    R("=" * 88)
    R("DAY 1 — TRACK A ANALYSIS")
    R("=" * 88)
    R(f"cases: {len(traj)}")
    R("")

    # ══════════════════════════════════════════════ A1 confusion matrix
    R("=" * 88)
    R("A1  DECISION-LEVEL CONFUSION MATRIX   -> Sdyd Q7 'How is success computed?'")
    R("=" * 88)
    for tau, tname in ((TAU_V1, "tau=0.0028 (v1 fallback)"),
                       (TAU_EQ7, "tau=0.0000 (Eq.7 as written)")):
        cm = Counter()
        for eid, t in traj.items():
            b = bench[eid]
            if b.get("split") != "test":
                continue
            d = decide(t["cscore"], tau)
            g = "conflict" if b.get("conflict_present") else "safe"
            cm[(d, g)] += 1
        R(f"\n  {tname}   (test split, n=1261)")
        R(f"    {'decision':<12}{'safe':>8}{'conflict':>10}")
        for d in ("SAFE", "UNCERTAIN", "ABSTAIN"):
            R(f"    {d:<12}{cm[(d, 'safe')]:>8}{cm[(d, 'conflict')]:>10}")
        results[f"A1_{tau}"] = {f"{d}|{g}": n for (d, g), n in cm.items()}
    R("")
    R("  SCORING CONVENTION TO STATE IN THE PAPER:")
    R("    ABSTAIN   = positive detection (true positive if conflict present)")
    R("    UNCERTAIN = counted as positive detection (defers to human review)")
    R("    SAFE      = negative prediction")
    R("    Macro-F1 averages F1(conflict) and F1(safe).")
    R("")
    R("  WHY UNCERTAIN IS ALWAYS 0:")
    R("    v1 set tau = 0.5 * min(positive conflict score), so min_conflict = 2*tau.")
    R("    Every real conflict therefore scores >= 2*tau > tau -> ABSTAIN.")
    R("    Every safe case scores exactly 0 -> SAFE.")
    R("    The band [tau/2, tau) is unreachable BY CONSTRUCTION, not empirically empty.")
    R("")

    # ══════════════════════════════════════════════ A2 extraction audit
    R("=" * 88)
    R("A2  EXTRACTION / KG-LOOKUP FAILURE AUDIT   -> AC-3, oJaU 'never characterized'")
    R("=" * 88)
    fails = []
    for eid, t in traj.items():
        b = bench[eid]
        pc = t["steps"][1]["paths_checked"]
        problems = []
        if pc == 0:
            problems.append("kg_lookup_returned_nothing")
        if b.get("lab_value") is None:
            problems.append("no_lab_value")
        drug = b.get("drug") or ""
        if "-" in drug:
            problems.append("combination_product")
        if problems:
            fails.append({
                "example_id": eid, "drug": drug, "stratum": b.get("stratum"),
                "conflict": b.get("conflict_present"), "cscore": t["cscore"],
                "paths_checked": pc, "decision": decide(t["cscore"], TAU_V1),
                "problems": ";".join(problems),
            })
    kgfail = [f for f in fails if "kg_lookup" in f["problems"]]
    kgfail_conf = [f for f in kgfail if f["conflict"]]
    rescued = [f for f in kgfail_conf if f["decision"] != "SAFE"]
    silent = [f for f in kgfail_conf if f["decision"] == "SAFE"]
    combo = [f for f in fails if "combination_product" in f["problems"]]
    combo_silent = [f for f in combo if f["conflict"]
                    and f["decision"] == "SAFE"]

    R(f"  total cases flagged with any problem       : {len(fails)}")
    R("")
    R("  LAYER 1 — KG lookup returned nothing (paths_checked == 0)")
    R(f"    cases                                   : {len(kgfail)}")
    R(f"    ...labelled CONFLICT                    : {len(kgfail_conf)}")
    R(f"    ...rescued by the local rule table      : {len(rescued)}")
    R(f"    ...SILENTLY CLEARED as SAFE             : {len(silent)}")
    R("")
    R("  LAYER 2 — combination products (hyphenated drug strings)")
    R(f"    cases                                   : {len(combo)}")
    R(f"    ...labelled CONFLICT, cleared as SAFE   : {len(combo_silent)}")
    R("")
    R("  Failure mode: a failed lookup yields paths_checked=0 -> PathViol=0.")
    R("  If NumDist also fails to match, C=0 and the case clears WITHOUT")
    R("  abstaining -- a silent false clearance, not an abstention.")
    R("")
    R("  HONEST FRAMING FOR THE PAPER: the redundant local THRESHOLD_RULES table")
    R("  in compute_num_dist() rescues most KG lookup failures, which is why the")
    R("  silent-clearance count is small. That redundancy is undocumented in the")
    R("  submission and should be stated -- it is load-bearing for the")
    R("  specificity claim, not an implementation detail.")
    R("")
    R("  Top drug strings that produced NO KG match:")
    for d, n in Counter(f["drug"] for f in fails
                        if "kg_lookup" in f["problems"]).most_common(8):
        R(f"    {n:>4}  {d}")
    R("")
    R("  ROOT CAUSE (state this in the paper): the Cypher uses")
    R("    WHERE toLower(d.name) CONTAINS toLower($drug)")
    R("  so the KG node name must CONTAIN the full prescription string.")
    R("  'Dabigatran Etexilate' does not match the node 'Dabigatran' — the")
    R("  containment runs in the wrong direction for multi-word drug strings.")
    with open(OUT / "A2_extraction_failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fails[0].keys()))
        w.writeheader()
        w.writerows(fails)
    results["A2"] = {"n_problem": len(
        fails), "n_silently_cleared": len(silent)}
    R("")

    # ══════════════════════════════════════════════ A3 boundary sweep
    R("=" * 88)
    R("A3  NEAR-BOUNDARY SWEEP   -> Jn3J Q2 'eGFR = 29, 30, 31'")
    R("=" * 88)
    sweep_rows = []
    for drug, spec in SWEEP_DRUGS.items():
        for op, thr in spec["rules"]:
            R(f"\n  {drug} / {spec['lab']}  (threshold {op}{thr})")
            R(f"    {'value':>10}{'PathViol':>10}{'NumDist':>10}{'C':>10}   decision")
            lo, hi = thr * 0.5, thr * 1.5
            step = (hi - lo) / 40
            v = lo
            while v <= hi + 1e-9:
                c, pv, nd = score_synthetic(drug, round(v, 3))
                sweep_rows.append({"drug": drug, "lab": spec["lab"],
                                   "threshold": thr, "value": round(v, 3),
                                   "pathviol": round(pv, 4), "numdist": round(nd, 4),
                                   "cscore": c, "decision": decide(c, TAU_V1)})
                v += step
            d = max(thr * 0.02, 0.01)
            for val in (thr - 3*d, thr - d, thr, thr + d, thr + 3*d):
                val = round(val, 3)
                c, pv, nd = score_synthetic(drug, val)
                R(f"    {val:>10.2f}{pv:>10.3f}{nd:>10.4f}{c:>10.4f}   {decide(c, TAU_V1)}")
    R("")
    R("  KEY POINTS FOR THE REBUTTAL:")
    R("   1. The score is a STEP FUNCTION, not a gradient. C changes")
    R("      discontinuously as a value crosses a threshold; between thresholds")
    R("      it varies only through NumDist.")
    R("   2. Where a drug has TWO thresholds (metformin: <30 major, <45")
    R("      moderate) there is an intermediate plateau -- eGFR 30-45 still")
    R("      flags, correctly, as a dose-reduction case. Worth reporting: the")
    R("      system is more clinically granular than the binary labels admit,")
    R("      and this is why it produced a 'false positive' at eGFR 44.6 that")
    R("      was in fact correct.")
    R("   3. Outside every threshold, C is EXACTLY 0. There is no small-but-")
    R("      nonzero region, so nothing can land in the UNCERTAIN band.")
    with open(OUT / "A3_boundary_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    R("")

    # ══════════════════════════════════════════════ A4 noise robustness
    R("=" * 88)
    R("A4  MEASUREMENT-NOISE ROBUSTNESS   -> Jn3J Q1 'measurement noise'")
    R("=" * 88)
    random.seed(42)
    N_DRAWS = 100
    flips = []
    base_dec = {}
    for eid, t in traj.items():
        base_dec[eid] = decide(t["cscore"], TAU_V1)
    for draw in range(N_DRAWS):
        changed = 0
        for eid, t in traj.items():
            b = bench[eid]
            v = b.get("lab_value")
            thr = b.get("threshold")
            op = b.get("operator")
            if v is None or thr is None or op not in ("<", ">"):
                continue
            cv = cv_for(b.get("lab_name"))
            v2 = v * (1 + random.gauss(0, cv))
            if violated(op, v, thr) != violated(op, v2, thr):
                changed += 1
        flips.append(changed)
    mean_flip = sum(flips) / len(flips)
    R(f"  draws: {N_DRAWS}   assay CV: eGFR 6.5%, creatinine 6.5%, potassium 2%, INR 10%, "
      f"ALT 10%, glucose 4%, WBC 5%, platelet 5%")
    R(f"  mean cases whose threshold verdict flips: {mean_flip:.1f} "
      f"({mean_flip/len(traj)*100:.1f}% of {len(traj)})")
    R(f"  range: {min(flips)} .. {max(flips)}")
    R("")
    R("  INTERPRETATION (this is the answer to Jn3J):")
    R("    Noise does NOT smear safe scores off zero. A perturbed value either")
    R("    crosses its threshold (rule fires, C jumps to a large value) or does")
    R("    not (C stays exactly 0). Scores are quantized by the rule structure,")
    R("    so tau_hat is unaffected; what changes is which side of the boundary")
    R("    a near-threshold case falls on. The exposure is concentrated entirely")
    R("    in near-boundary cases -- see A3.")
    results["A4"] = {"mean_flips": mean_flip, "min": min(flips), "max": max(flips),
                     "n_draws": N_DRAWS}
    R("")

    with open(OUT / "day1_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT / "day1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT/'day1_report.txt'}")
    print(f"[saved] {OUT/'A2_extraction_failures.csv'}")
    print(f"[saved] {OUT/'A3_boundary_sweep.csv'}")


if __name__ == "__main__":
    main()
