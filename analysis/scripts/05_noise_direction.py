#!/usr/bin/env python3
"""
05_noise_direction.py — decompose A4's noise result by DIRECTION
=================================================================

A4 reported ~51.8 threshold-verdict flips (3.3% of 1576) under assay noise,
pooled. Pooled is the wrong summary: stratum-0 cases sit >=20% from every
threshold by construction, while 33-41% of conflict cases sit within 10%.
So the flips cannot be symmetric.

This splits them into:
    MISSED      conflict -> safe   (recall damage)
    FALSE ALARM safe -> conflict   (specificity damage)

Answers Jn3J Q1 with a directional claim instead of a pooled rate.

Usage:
    python scripts/05_noise_direction.py --bench data/raw/medconflict_bench_v2.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

# Assay coefficients of variation. First four match A4; the rest are
# assumptions and are labelled as such in the output.
# EXACTLY the ASSAY_CV dict from 04_day1_analysis.py lines 42-44. All eight
# labs are specified there; the A4 print statement listing only four is stale.
CV = {
    "egfr": 0.065,          # via creatinine; see caveat below
    "creatinine": 0.065,
    "potassium": 0.020,
    "inr": 0.100,
    "alt": 0.100,
    "glucose": 0.040,
    "platelet count": 0.050,
    "wbc": 0.050,
    "tsh": 0.080,           # not in A4; no TSH cases exist in the benchmark
}
ASSUMED = {"tsh"}

_paren = re.compile(r"\([^)]*\)")
_punct = re.compile(r"[^a-z0-9 ]+")


def canon(name):
    if not isinstance(name, str) or not name.strip():
        return None
    x = _punct.sub(" ", _paren.sub(" ", name.lower()))
    x = " ".join(x.split())
    if x.startswith("egfr"):
        return "egfr"
    for k in CV:
        if x == k or x.startswith(k) or k in x.split():
            return k
    return None


def violated(op, v, t):
    if op in ("<", "lt"):
        return v < t
    if op in ("<=", "le"):
        return v <= t
    if op in (">", "gt"):
        return v > t
    if op in (">=", "ge"):
        return v >= t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/raw/medconflict_bench_v2.jsonl")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-a4", action="store_true",
                    help="DEPRECATED. A4 specifies CVs for all eight labs; its "
                         "printed list of four is stale. Report the full run.")
    args = ap.parse_args()

    allowed = None
    if args.only_a4:
        allowed = {"egfr", "creatinine", "potassium", "inr", "alt"}
        print("  [--only-a4] DEPRECATED -- A4 specifies all eight CVs.")
        print("              Report the UNRESTRICTED run instead.")

    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = Counter()

    for line in open(args.bench, encoding="utf-8"):
        e = json.loads(line)
        v, t, op = e.get("lab_value"), e.get("threshold"), e.get("operator")
        lab = canon(e.get("lab_name"))
        if v is None or t is None or not op or lab is None:
            skipped[e.get("stratum")] += 1
            continue
        if allowed is not None and lab not in allowed:
            skipped[e.get("stratum")] += 1
            continue
        base = violated(op, float(v), float(t))
        if base is None:
            skipped[e.get("stratum")] += 1
            continue
        rows.append((e.get("stratum"), lab, float(v), float(t), op, base))

    print("=" * 74)
    print("NOISE FLIP DIRECTION  (Jn3J Q1)")
    print("=" * 74)
    print(f"  evaluable cases: {len(rows)}   draws: {args.draws}")
    print(f"  skipped (no lab/threshold/operator, mostly S1): {dict(skipped)}")
    used = sorted({r[1] for r in rows})
    print(f"  labs present: {used}")
    a = [l for l in used if l in ASSUMED]
    if a:
        print(f"  !! CV ASSUMED (not from A4) for: {a} — state this or drop them")
    print()

    flips = defaultdict(lambda: np.zeros(args.draws))
    per_case_flip = Counter()

    for i, (st, lab, v, t, op, base) in enumerate(rows):
        sd = CV[lab] * abs(v)
        pert = rng.normal(v, sd, args.draws)
        newv = np.array([violated(op, float(x), t) for x in pert])
        f = newv != base
        key = ("MISSED" if base else "FALSE ALARM", st)
        flips[key] += f.astype(float)
        if f.any():
            per_case_flip[(("MISSED" if base else "FALSE ALARM"), st)] += 1

    print("  Mean flips per draw, by direction and stratum:")
    tot_missed = tot_fa = 0.0
    for (direction, st), arr in sorted(flips.items()):
        m = arr.mean()
        if direction == "MISSED":
            tot_missed += m
        else:
            tot_fa += m
        print(f"    {direction:<12} stratum {st}   mean={m:8.2f}   "
              f"cases that ever flip={per_case_flip[(direction, st)]}")
    print()
    print(f"    TOTAL MISSED      (conflict -> safe) : {tot_missed:8.2f}")
    print(f"    TOTAL FALSE ALARM (safe -> conflict) : {tot_fa:8.2f}")
    print(f"    ratio missed:false-alarm             : "
          f"{(tot_missed / tot_fa) if tot_fa else float('inf'):.1f} : 1")
    print()
    print("  CLAIM FOR THE REBUTTAL:")
    print("    If false alarms are ~0, assay noise degrades RECALL, not")
    print("    specificity. The specificity=1.000 result is robust to")
    print("    measurement noise; the detection result is not. Report the")
    print("    directional split, not the pooled 3.3%.")
    print()
    print("  CAVEAT to state: eGFR is a nonlinear function of creatinine, so")
    print("  perturbing eGFR directly at creatinine's CV is an approximation.")
    print("  It is the same approximation A4 made; note it rather than hide it.")


if __name__ == "__main__":
    main()