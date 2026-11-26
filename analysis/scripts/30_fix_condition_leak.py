#!/usr/bin/env python3
"""
30_fix_condition_leak.py — close the conflict_condition label leak
===================================================================

THE DEFECT
----------
`conflict_condition` is populated for every conflict case and "none" for every
safe case, so this one-line classifier scores 1261/1261 on the test split:

    predict_conflict = (conflict_condition != "none")

Any evaluation on the benchmark by a system that reads the field is therefore
uninformative, and baseline prompts that printed it handed the label to the
model.

THE FIX
-------
A safe case is not a case with no condition -- it is a case where the condition
is present and the threshold is NOT crossed. Metformin in a patient with renal
impairment and eGFR 98 is a genuine negative: the rule applies, the value is
safe. The field should say "Renal Impairment", not "none".

This script populates conflict_condition on stratum-0 cases from the condition
attached to the matching conflict rule, matched on (drug, lab) and falling back
to (lab, operator, threshold). Nothing else is altered: labels, values,
thresholds and rule ids are untouched.

WHAT THIS DOES NOT FIX
----------------------
Stratum 1 (categorical) has no negatives at all -- there is no case where an
allergy is documented and the prescribed drug is unrelated to it. So for
categorical conflicts the field remains perfectly predictive, because every
categorical case in the benchmark is a positive. Closing that requires
generating new cases, not relabelling existing ones. The script reports the
residual leak so the limitation is quantified rather than assumed away.

Usage:
    python scripts/30_fix_condition_leak.py
    python scripts/30_fix_condition_leak.py --apply
"""

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


def norm_drug(name):
    if not name:
        return ""
    n = re.sub(r"\s*\(.*?\)", "", str(name))
    n = re.sub(r"\s+\d+(\.\d+)?\s*m?g.*", "", n, flags=re.I)
    return n.strip().lower()


def drug_key(name):
    """First alphabetic token >2 chars: 'MetFORMIN XR' -> 'metformin'."""
    for t in norm_drug(name).split():
        if len(t) > 2 and t.isalpha():
            return t
    return norm_drug(name)


def canon_lab(s):
    if not s:
        return ""
    x = re.sub(r"\([^)]*\)", " ", str(s).lower())
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    return " ".join(x.split())


def strip_nan(o):
    """NaN/Infinity are not valid JSON. Coerce to null before writing."""
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: strip_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strip_nan(v) for v in o]
    return o


def leak_rate(rows, split="test"):
    """Accuracy of: predict conflict iff conflict_condition != none."""
    sub = [e for e in rows if e.get("split") == split]
    ok = sum(1 for e in sub
             if (str(e.get("conflict_condition") or "none").lower()
                 not in ("", "none")) == bool(e.get("conflict_present")))
    return ok, len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(DATA / "medconflict_bench_v3.jsonl"))
    ap.add_argument("--out", default=str(DATA / "medconflict_bench_v4.jsonl"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.bench, encoding="utf-8")]

    print("=" * 74)
    print("CONDITION LEAK FIX")
    print("=" * 74)
    ok, n = leak_rate(rows)
    print(f"  loaded {len(rows)} cases from {Path(args.bench).name}")
    print(f"  BEFORE — one-line classifier on test split: {ok}/{n} = {ok/max(n,1):.4f}")
    print()

    # ---- harvest conditions from conflict cases ---------------------------
    by_drug_lab = {}
    by_lab_thr = {}
    for e in rows:
        cond = str(e.get("conflict_condition") or "none")
        if cond.lower() in ("", "none") or not e.get("conflict_present"):
            continue
        dk, lk = drug_key(e.get("drug")), canon_lab(e.get("lab_name"))
        if dk and lk:
            by_drug_lab.setdefault((dk, lk), cond)
        thr, op = e.get("threshold"), e.get("operator")
        if lk and thr is not None and op:
            by_lab_thr.setdefault((lk, op, str(thr)), cond)

    print(f"  condition sources: {len(by_drug_lab)} by (drug, lab), "
          f"{len(by_lab_thr)} by (lab, op, threshold)")
    print()

    # ---- populate safe cases ----------------------------------------------
    filled = collections.Counter()
    unmatched = collections.Counter()
    for e in rows:
        if e.get("stratum") != 0:
            continue
        if str(e.get("conflict_condition") or "none").lower() not in ("", "none"):
            continue
        dk, lk = drug_key(e.get("drug")), canon_lab(e.get("lab_name"))
        cond = by_drug_lab.get((dk, lk))
        how = "drug+lab"
        if cond is None:
            cond = by_lab_thr.get((lk, e.get("operator"), str(e.get("threshold"))))
            how = "lab+threshold"
        if cond is None:
            unmatched[(dk, lk)] += 1
            continue
        e["conflict_condition"] = cond
        e["condition_source"] = (
            f"populated from matching conflict rule via {how}; the rule applies "
            "to this patient and the threshold is not crossed")
        filled[(dk, cond)] += 1

    total_filled = sum(filled.values())
    print(f"  safe cases populated : {total_filled}")
    for (dk, cond), c in filled.most_common(12):
        print(f"      {c:>4}  {dk:<16} -> {cond}")
    if unmatched:
        print()
        print(f"  !! {sum(unmatched.values())} safe case(s) had no matching rule:")
        for k, c in unmatched.most_common(8):
            print(f"      {c:>4}  drug={k[0]!r} lab={k[1]!r}")
        print("     These still read 'none' and remain separable. Investigate.")
    print()

    ok2, n2 = leak_rate(rows)
    print(f"  AFTER  — one-line classifier on test split: {ok2}/{n2} = {ok2/max(n2,1):.4f}")
    print()

    # ---- residual, by stratum ---------------------------------------------
    print("  residual separability by stratum (test split):")
    for s in sorted({e.get("stratum") for e in rows}):
        sub = [e for e in rows if e.get("stratum") == s and e.get("split") == "test"]
        if not sub:
            continue
        pop = sum(1 for e in sub
                  if str(e.get("conflict_condition") or "none").lower()
                  not in ("", "none"))
        print(f"    stratum {s}: {len(sub):>4} cases, {pop:>4} with a condition "
              f"({pop/max(len(sub),1):.0%})")
    print()
    print("  Stratum 1 has no negatives in this benchmark, so a condition is")
    print("  present on 100% of categorical cases by construction. That cannot be")
    print("  closed by relabelling; it needs new cases in which an allergy is")
    print("  documented and the prescribed drug is unrelated to it. State this")
    print("  as a remaining limitation.")
    print()

    if not args.apply:
        print("  DRY RUN — nothing written. Re-run with --apply.")
        return

    out = Path(args.out)
    n_nan = 0
    with open(out, "w", encoding="utf-8") as f:
        for e in rows:
            clean = strip_nan(e)
            if clean != e:
                n_nan += 1
            f.write(json.dumps(clean, default=str, allow_nan=False) + "\n")
    print(f"  [written] {out}")
    if n_nan:
        print(f"  ({n_nan} record(s) had NaN coerced to null -- the previous "
              f"files were not valid JSON)")

    with open(out.with_suffix(".changelog.json"), "w") as f:
        json.dump({"source": args.bench,
                   "safe_cases_populated": total_filled,
                   "leak_before": round(ok / max(n, 1), 4),
                   "leak_after": round(ok2 / max(n2, 1), 4),
                   "unmatched": {str(k): v for k, v in unmatched.items()},
                   "note": "stratum 1 has no negatives; residual leak there is "
                           "structural and needs new cases"}, f, indent=2)
    print(f"  [written] {out.with_suffix('.changelog.json')}")
    print()
    print("  NEXT: re-run 03_rerun_fixed.py and 04_day1_analysis.py against v4.")
    print("  Scores should NOT move -- the scorer does not read this field, and")
    print("  that is worth verifying rather than assuming.")


if __name__ == "__main__":
    main()