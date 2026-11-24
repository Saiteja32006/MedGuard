#!/usr/bin/env python3
"""
23_verify_constants.py — audit the curated constants against REAL MIMIC data
=============================================================================

Every hand-written lookup table in this pipeline is a liability until it has
been checked against the data. One such table (the WBC itemid) was already
wrong. This script checks the rest, so that nothing reported in the rebuttal
rests on an unverified constant.

CHECK A  itemid identity
    For each canonical lab, list every d_labitems entry whose label matches,
    ranked by observed volume, and flag when the itemid in LAB_ITEMIDS is not
    the highest-volume Blood entry. This is the generalized form of the check
    that caught WBC (51300 vs 51301).

CHECK B  units
    Print the observed valueuom distribution and value percentiles per lab.
    This is where a threshold written in the wrong unit becomes visible.

CHECK C  threshold plausibility  <-- THE IMPORTANT ONE
    For every rule with a numeric threshold, compute the fraction of REAL
    MIMIC lab values that would violate it. A rule that fires on ~100% or
    ~0% of the population is almost certainly unit-mismatched or misdirected,
    not clinically rare. Suspected cases: platelets and WBC, which MIMIC
    reports in K/uL (250, 8.5) against thresholds written in absolute counts
    (100000, 3000).

CHECK D  drug near-misses
    Reads unmatched_drugs.csv from the coverage run and flags any unmatched
    string that contains a rule ingredient as a substring -- i.e. drugs the
    tokenizer should have caught and did not.

Nothing here modifies data. It reports; you decide.

Usage:
  python 23_verify_constants.py \
    --mimic-dir "/content/drive/MyDrive/NIPS_2026/Dataset/mimic-iv-hosp" \
    --bench-dir "/content/drive/MyDrive/NIPS_2026/MedConflict-Bench" \
    --cov-dir   "/content/drive/MyDrive/NIPS_2026/Rebuttal_Coverage/full"
"""

import argparse
import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
_dep = _here / "20_coverage_denominator.py"
if not _dep.exists():
    sys.exit(f"FATAL: {_dep} not found (shared constants live there).")
_spec = importlib.util.spec_from_file_location("cov20", _dep)
cov20 = importlib.util.module_from_spec(_spec)
sys.modules["cov20"] = cov20
_spec.loader.exec_module(cov20)

# Keyword used to find candidate itemids for each canonical lab.
LAB_KEYWORDS = {
    "creatinine": "creatinine",
    "potassium": "potassium",
    "inr": "INR",
    "alt": "ALT|Alanine",
    "glucose": "glucose",
    "platelet count": "platelet",
    "wbc": "WBC|White Blood",
    "tsh": "TSH|Thyroid Stimulating",
}


def sample_labs(mimic, itemids, sample_rows):
    """Chunked scan keeping only the itemids of interest, with units."""
    keep, scanned = [], 0
    for ch in pd.read_csv(mimic / "labevents.csv.gz",
                          usecols=["subject_id", "itemid", "valuenum",
                                   "valueuom"],
                          chunksize=2_000_000, compression="gzip",
                          low_memory=False):
        scanned += len(ch)
        sub = ch[ch["itemid"].isin(itemids)]
        if not sub.empty:
            keep.append(sub)
        if scanned >= sample_rows:
            break
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(
        columns=["subject_id", "itemid", "valuenum", "valueuom"])
    return df, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--cov-dir", default=None)
    ap.add_argument("--sample-rows", type=int, default=40_000_000)
    args = ap.parse_args()

    mimic = Path(args.mimic_dir)
    cov = Path(args.cov_dir) if args.cov_dir else Path(".")

    dli = pd.read_csv(mimic / "d_labitems.csv.gz", compression="gzip")

    # ---------------------------------------------------------------- CHECK A
    print("=" * 78)
    print("CHECK A — itemid identity")
    print("=" * 78)

    candidates = set()
    per_lab_cands = {}
    for lab, kw in LAB_KEYWORDS.items():
        hit = dli[dli["label"].astype(str).str.contains(kw, case=False,
                                                        na=False, regex=True)]
        per_lab_cands[lab] = hit
        candidates |= set(hit["itemid"].tolist())
    for v in cov20.LAB_ITEMIDS.values():
        candidates |= set(cov20._itemids(v))

    print(f"  scanning up to {args.sample_rows:,} lab rows for "
          f"{len(candidates)} candidate itemids...")
    samp, scanned = sample_labs(mimic, candidates, args.sample_rows)
    vol = samp["itemid"].value_counts().to_dict()
    print(f"  scanned {scanned:,} rows, kept {len(samp):,}")
    print()

    problems = []
    for lab, hit in per_lab_cands.items():
        used = set(cov20._itemids(cov20.LAB_ITEMIDS.get(lab, [])))
        rows = []
        for _, r in hit.iterrows():
            rows.append((int(r["itemid"]), str(r["label"]), str(r["fluid"]),
                         str(r["category"]), vol.get(int(r["itemid"]), 0)))
        rows.sort(key=lambda x: -x[4])
        print(f"  {lab.upper()}   (using {sorted(used)})")
        for iid, lbl, fluid, cat, n in rows[:6]:
            mark = " <-- USED" if iid in used else ""
            print(f"      {iid:>7}  {n:>10,}  {lbl[:34]:<34} {fluid:<7} {cat}{mark}")
        blood = [r for r in rows if r[2].lower() == "blood"]
        if blood and blood[0][0] not in used:
            problems.append((lab, blood[0][0], blood[0][1], blood[0][4]))
            print(f"      !! highest-volume Blood entry {blood[0][0]} "
                  f"({blood[0][1]}) is NOT in use")
        print()

    # ---------------------------------------------------------------- CHECK B
    print("=" * 78)
    print("CHECK B — units and value distributions")
    print("=" * 78)
    id2lab = {i: k for k, v in cov20.LAB_ITEMIDS.items()
              for i in cov20._itemids(v)}
    samp["lab"] = samp["itemid"].map(id2lab)
    dist = {}
    for lab, g in samp.dropna(subset=["lab"]).groupby("lab"):
        v = pd.to_numeric(g["valuenum"], errors="coerce").dropna()
        uom = collections.Counter(g["valueuom"].dropna().astype(str)).most_common(3)
        if v.empty:
            print(f"  {lab:<16} NO NUMERIC VALUES")
            continue
        dist[lab] = v
        print(f"  {lab:<16} n={len(v):>9,}  units={uom}")
        print(f"                   p01={v.quantile(.01):>10.3f} "
              f"median={v.median():>10.3f} p99={v.quantile(.99):>10.3f}")

    # eGFR is DERIVED, not measured. Comparing an eGFR threshold against raw
    # creatinine would falsely flag every renal rule, so build it properly.
    cr = samp[samp["lab"] == "creatinine"].copy()
    if not cr.empty:
        pts = pd.read_csv(mimic / "patients.csv.gz",
                          usecols=["subject_id", "gender", "anchor_age"],
                          compression="gzip")
        cr = cr.merge(pts, on="subject_id", how="left")
        cr["valuenum"] = pd.to_numeric(cr["valuenum"], errors="coerce")
        cr = cr.dropna(subset=["gender", "anchor_age", "valuenum"])
        cr = cr[cr["valuenum"] > 0]
        if not cr.empty:
            fem = cr["gender"].eq("F")
            kap = np.where(fem, 0.7, 0.9)
            alp = np.where(fem, -0.241, -0.302)
            scr = cr["valuenum"].to_numpy(dtype=float)
            age = cr["anchor_age"].to_numpy(dtype=float)
            eg = (142.0 * np.power(np.minimum(scr / kap, 1.0), alp)
                  * np.power(np.maximum(scr / kap, 1.0), -1.200)
                  * np.power(0.9938, age) * np.where(fem, 1.012, 1.0))
            dist["egfr"] = pd.Series(eg)
            e = dist["egfr"]
            print(f"  {'egfr (derived)':<16} n={len(e):>9,}  units=[('mL/min/1.73m2', "
                  f"{len(e)})]")
            print(f"                   p01={e.quantile(.01):>10.3f} "
                  f"median={e.median():>10.3f} p99={e.quantile(.99):>10.3f}")
    print()

    # ---------------------------------------------------------------- CHECK C
    print("=" * 78)
    print("CHECK C — threshold plausibility  (fires on what % of real values?)")
    print("=" * 78)
    cov20.BENCH = Path(args.bench_dir)
    rules, _ = cov20.stage0_rules(cov, verbose=False)

    def canon_for_rule(r):
        raw = (r.get("lab_name") or "").strip()
        if not raw:
            return None
        s = " ".join(cov20._punct.sub(" ", cov20._paren.sub(" ", raw.lower())).split())
        return "egfr" if s.startswith("egfr") else cov20.canon_lab(raw)

    suspect = []
    print(f"  {'rule':<14} {'lab':<15} {'op':<3} {'thresh':>10}  {'fires on':>9}")
    for r in sorted(rules, key=lambda x: x["rule_id"]):
        lab = canon_for_rule(r)
        t = r.get("threshold")
        op = r.get("operator")
        if lab is None or t is None or not op or lab not in dist:
            continue
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        v = dist[lab]
        frac = float((v < t).mean()) if op.startswith("<") else float((v > t).mean())
        flag = ""
        if frac > 0.90 or frac < 0.001:
            flag = "   <<< IMPLAUSIBLE"
            suspect.append((r["rule_id"], lab, op, t, frac))
        print(f"  {r['rule_id']:<14} {lab:<15} {op:<3} {t:>10.4g}  "
              f"{frac:>8.2%}{flag}")
    print()
    if suspect:
        print("  SUSPECT RULES — verify the unit before reporting anything that")
        print("  depends on them:")
        for rid, lab, op, t, frac in suspect:
            v = dist[lab]
            print(f"    {rid}: threshold {op}{t:g} on {lab}, whose real values run "
                  f"{v.quantile(.01):.3g}..{v.quantile(.99):.3g}")
            print(f"      -> fires on {frac:.1%} of the population. If the rule was")
            print(f"         written in a different unit, it is always/never active.")
    else:
        print("  No implausible thresholds detected.")
    print()

    # ---------------------------------------------------------------- CHECK D
    print("=" * 78)
    print("CHECK D — drug near-misses")
    print("=" * 78)
    um = cov / "unmatched_drugs.csv"
    if not um.exists():
        print(f"  SKIP: {um} not found (run the coverage job first).")
    else:
        d = pd.read_csv(um)
        _, expanded = cov20.stage0_rules(cov, verbose=False)
        ings = sorted({e["ingredient"] for e in expanded})
        near = []
        for _, row in d.iterrows():
            s = str(row["drug_norm"])
            for ing in ings:
                if ing in s and ing not in s.split():
                    near.append((int(row["count"]), s, ing))
                    break
        near.sort(reverse=True)
        if near:
            print(f"  {len(near)} unmatched strings CONTAIN a rule ingredient but")
            print("  did not tokenize to it — these are missed coverage:")
            for n, s, ing in near[:25]:
                print(f"    {n:>9,}  {s:<44} (contains {ing!r})")
        else:
            print("  No near-misses: no unmatched string contains a rule ingredient.")
    print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  itemid problems : {len(problems)}")
    for lab, iid, lbl, n in problems:
        print(f"      {lab}: consider {iid} ({lbl}, {n:,} obs)")
    print(f"  suspect rules   : {len(suspect)}")
    for rid, lab, op, t, frac in suspect:
        print(f"      {rid} fires on {frac:.1%} of real {lab} values")
    print()
    print("  Anything listed above is an unverified constant. Fix it or state")
    print("  it as a limitation; do not report a number that depends on it.")


if __name__ == "__main__":
    main()
