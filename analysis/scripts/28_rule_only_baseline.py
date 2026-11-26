#!/usr/bin/env python3
"""
28_rule_only_baseline.py — deterministic checker, no KG, no embeddings, no LLM
===============================================================================

Answers Reviewer Sdyd:
  W2  "why not directly use these functions to give the results? What is the
       use of LLMs?"
  W4  "there should be a clearer description of the data to evaluate whether a
       MLLM is at all necessary for this task."

The honest test of that question is to build the trivial system and report what
it scores. This is ~40 lines of rule matching over the same benchmark fields
MedGuard consumes. No graph, no retrieval, no model of any kind.

If it matches MedGuard, that is the answer to W2, and it is the result that
makes the MLLM failure worth reporting: the task has an exact deterministic
solution, and frontier models still fail it.

RULE TABLE
----------
Derived from the benchmark's own rule metadata (rule_id -> drug, lab, operator,
threshold), NOT from the case under test. A case is matched by (drug, lab), and
the threshold comes from the rule table, so no case supplies its own answer.

TWO CONDITIONS
--------------
  --no-condition (default)  uses drug, interacting_drug, lab_name, lab_value
  --use-condition           additionally uses conflict_condition

The second exists because conflict_condition is "none" for every safe case and
populated for every conflict, so any classifier reading it scores perfectly.
Running both quantifies that leak rather than leaving it implicit.

Usage:
    python scripts/28_rule_only_baseline.py
    python scripts/28_rule_only_baseline.py --use-condition
"""

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "out"


def norm_drug(name):
    if not name:
        return ""
    n = re.sub(r"\s*\(.*?\)", "", str(name))
    n = re.sub(r"\s+\d+(\.\d+)?\s*m?g.*", "", n, flags=re.I)
    return n.strip().lower()


def canon_lab(s):
    if not s:
        return ""
    x = re.sub(r"\([^)]*\)", " ", str(s).lower())
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    return " ".join(x.split())


def violated(op, val, thr):
    if val is None or thr is None or not op:
        return False
    return (op == "<" and val < thr) or (op == ">" and val > thr)


def build_rules(bench):
    """rule_id -> (drug, interacting_drug, lab, op, thr). One entry per rule."""
    rules = {}
    for e in bench:
        rid = e.get("rule_id")
        if not rid or rid in ("", "none", None):
            continue
        if rid in rules:
            continue
        rules[rid] = {
            "rule_id": rid,
            "drug": norm_drug(e.get("drug")),
            "interacting": norm_drug(e.get("interacting_drug")),
            "lab": canon_lab(e.get("lab_name")),
            "op": e.get("operator"),
            "thr": e.get("threshold"),
            "categorical": not e.get("lab_name"),
        }
    return rules


def predict(case, rules, use_condition):
    """True = conflict. Pure rule matching over structured fields."""
    d = norm_drug(case.get("drug"))
    inter = norm_drug(case.get("interacting_drug"))
    lab = canon_lab(case.get("lab_name"))
    val = case.get("lab_value")
    try:
        val = float(val) if val is not None else None
    except (TypeError, ValueError):
        val = None

    for r in rules.values():
        if not r["drug"] or r["drug"] not in d:
            continue
        if r["interacting"]:
            if not inter or r["interacting"] not in inter:
                continue
        if r["categorical"]:
            if use_condition:
                cond = str(case.get("conflict_condition") or "none").lower()
                if cond not in ("", "none"):
                    return True
            continue
        if not lab or r["lab"] != lab:
            continue
        try:
            thr = float(r["thr"])
        except (TypeError, ValueError):
            continue
        if violated(r["op"], val, thr):
            return True
    return False


def metrics(rows):
    tp = sum(1 for r in rows if r["pred"] and r["gt"])
    fp = sum(1 for r in rows if r["pred"] and not r["gt"])
    fn = sum(1 for r in rows if not r["pred"] and r["gt"])
    tn = sum(1 for r in rows if not r["pred"] and not r["gt"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1c = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ps = tn / (tn + fn) if tn + fn else 0.0
    rs = tn / (tn + fp) if tn + fp else 0.0
    f1s = 2 * ps * rs / (ps + rs) if ps + rs else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "specificity": round(rs, 4),
            "macro_f1": round((f1c + f1s) / 2, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(DATA / "medconflict_bench_v3.jsonl"))
    ap.add_argument("--use-condition", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    bench = [json.loads(l) for l in open(args.bench, encoding="utf-8")]
    rules = build_rules(bench)
    cases = [e for e in bench if e.get("split") == args.split]

    print("=" * 74)
    print("RULE-ONLY BASELINE — no KG, no embeddings, no LLM")
    print("=" * 74)
    print(f"  rules derived from benchmark metadata : {len(rules)}")
    print(f"  cases ({args.split} split)            : {len(cases)}")
    print(f"  conflict_condition used               : {args.use_condition}")
    print()

    rows = []
    for e in cases:
        rows.append({"stratum": e.get("stratum"),
                     "gt": bool(e.get("conflict_present")),
                     "pred": predict(e, rules, args.use_condition)})

    names = {0: "safe", 1: "categorical", 2: "numerical", 3: "multi-hop"}
    print(f"  {'stratum':<14}{'n':>6}{'detected':>10}{'rate':>9}")
    per = {}
    for s in sorted({r["stratum"] for r in rows}):
        sub = [r for r in rows if r["stratum"] == s]
        if s == 0:
            ok = sum(1 for r in sub if not r["pred"])
            print(f"  {names.get(s, s):<14}{len(sub):>6}{ok:>10}"
                  f"{ok/max(len(sub),1):>9.3f}   (specificity)")
            per[s] = round(ok / max(len(sub), 1), 4)
        else:
            ok = sum(1 for r in sub if r["pred"])
            print(f"  {names.get(s, s):<14}{len(sub):>6}{ok:>10}"
                  f"{ok/max(len(sub),1):>9.3f}")
            per[s] = round(ok / max(len(sub), 1), 4)
    print()

    m = metrics(rows)
    print(f"  specificity : {m['specificity']}")
    print(f"  macro-F1    : {m['macro_f1']}")
    print(f"  tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  tn={m['tn']}")
    print()
    print("  MedGuard (v3, test split) for comparison:")
    print("    specificity 1.000   macro-F1 0.991   S1 0.996  S2 0.973  S3 1.000")
    print()
    if not args.use_condition:
        print("  Categorical detection is expected to be ~0 here: without")
        print("  conflict_condition there is nothing to match a drug-allergy rule")
        print("  against. Re-run with --use-condition to see the leak.")
    else:
        print("  With conflict_condition, categorical detection is near-perfect")
        print("  because that field is 'none' for every safe case and populated")
        print("  for every conflict. Report this as a benchmark limitation.")
    print()
    print("  INTERPRETATION FOR THE RESPONSE:")
    print("   If the numerical and multi-hop rates match MedGuard, the reviewer")
    print("   is right that those strata need no learned component. That is the")
    print("   finding: the task has an exact deterministic solution and frontier")
    print("   MLLMs still fail it.")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "with_condition" if args.use_condition else "no_condition"
    with open(OUT / f"rule_only_baseline_{tag}.json", "w") as f:
        json.dump({"n_rules": len(rules), "n_cases": len(cases),
                   "use_condition": args.use_condition,
                   "per_stratum": {str(k): v for k, v in per.items()},
                   **m}, f, indent=2)
    print(f"  [saved] out/rule_only_baseline_{tag}.json")


if __name__ == "__main__":
    main()
