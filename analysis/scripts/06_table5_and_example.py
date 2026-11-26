#!/usr/bin/env python3
"""
06_table5_and_example.py — Day 2 Track A (A8, A9, A10). Laptop.

A8   Rebuild Table 5 with every row on the SAME denominator  -> Sdyd Q7, AC-2
A9   Worked end-to-end example (entities -> Cypher -> C)      -> Sdyd Q1/Q3/Q5
A10  Paired examples: MedGuard flags, baseline clears         -> Jn3J Q3

Usage:
    python scripts/06_table5_and_example.py
    python scripts/06_table5_and_example.py --baseline gpt-4o   # for A10
"""

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "out"


def decide(c, tau):
    if c > tau:
        return "ABSTAIN"
    if tau > 0 and c > 0 and c >= tau * 0.5:
        return "UNCERTAIN"
    return "SAFE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="gpt-4o")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    traj = {json.loads(l)["example_id"]: json.loads(l)
            for l in open(DATA / "genmedassist_trajectories_v1.jsonl", encoding="utf-8")}
    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(DATA / "medconflict_bench_v2.jsonl", encoding="utf-8")}
    test = [(e, t) for e, t in traj.items() if bench[e].get("split") == "test"]

    lines = []

    def R(s=""):
        print(s)
        lines.append(s)

    # ═══════════════════════════════════════════════ A8 Table 5 rebuild
    R("=" * 94)
    R("A8  TABLE 5 REBUILT — all rows on the same denominator")
    R("=" * 94)
    R("")
    R("  PROBLEM WITH THE SUBMITTED TABLE:")
    R("   'Sel. Acc.' for conformal rows = accuracy on NON-ABSTAINED cases only.")
    R("   'Sel. Acc.' for the no-abstention row = accuracy over ALL cases.")
    R("   Those are different denominators, so 0.822 vs 0.932 is not a")
    R("   like-for-like comparison. Every row below uses the same basis.")
    R("")
    R(f"  {'Method':<22}{'tau':>8}{'Flagged':>9}{'Cleared':>9}"
      f"{'Acc(all)':>10}{'Acc(clr)':>10}{'Recall':>8}{'Missed':>8}{'Spec':>7}")
    R("-" * 94)

    rows5 = []
    methods = [("Conformal (Eq.7)", 0.0000), ("v1 fallback", 0.0028),
               ("Fixed", 0.10), ("Fixed", 0.30), ("No abstention", -1.0)]
    for name, tau in methods:
        flagged = cleared = tp = tn = fn = fp = 0
        for e, t in test:
            gt = bool(bench[e].get("conflict_present"))
            if tau < 0:
                pred = t["cscore"] > 0
            else:
                pred = decide(t["cscore"], tau) != "SAFE"
            if pred:
                flagged += 1
                tp += gt
                fp += (not gt)
            else:
                cleared += 1
                fn += gt
                tn += (not gt)
        n = len(test)
        acc_all = (tp + tn) / n
        acc_cleared = tn / cleared if cleared else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        tshow = "—" if tau < 0 else f"{tau:.4f}"
        R(f"  {name:<22}{tshow:>8}{flagged:>9}{cleared:>9}"
          f"{acc_all:>10.3f}{acc_cleared:>10.3f}{rec:>8.3f}{fn:>8}{spec:>7.3f}")
        rows5.append({"method": name, "tau": tau, "flagged": flagged,
                      "cleared": cleared, "acc_all": acc_all,
                      "acc_cleared": acc_cleared, "recall": rec,
                      "missed": fn, "specificity": spec})
    R("-" * 94)
    R("  Acc(clr) = of the cases the system CLEARED, what fraction were truly safe.")
    R("             This is the clinically meaningful selective-accuracy number:")
    R("             it measures how much you can trust a 'SAFE' verdict.")
    R("")
    R("  WHY alpha=0.01 AND alpha=0.05 GAVE IDENTICAL ROWS in the submission:")
    R("    all safe calibration scores are exactly 0, so any quantile of them is")
    R("    0. tau_hat does not depend on alpha. Report this explicitly.")
    R("")

    # ═══════════════════════════════════════════════ A9 worked example
    R("=" * 94)
    R("A9  WORKED END-TO-END EXAMPLE   -> Sdyd 'show a detailed example'")
    R("=" * 94)
    pick = None
    for e, t in test:
        b = bench[e]
        if (b.get("stratum") == 2 and "metformin" in (b.get("drug") or "").lower()
                and b.get("conflict_present")):
            pick = (e, t, b)
            break
    if pick:
        e, t, b = pick
        R(f"\n  example_id: {e}")
        R(f"  MIMIC-IV   : subject_id={b.get('subject_id')} hadm_id={b.get('hadm_id')}")
        R("")
        R("  STEP 1 — Entity extraction (structured columns, NO model inference)")
        R(f"    drug       : {b.get('drug')!r}  ->  {t['steps'][0]['output']['drugs_normalised']}")
        R(f"    lab        : {b.get('lab_name')} = {b.get('lab_value')} {b.get('lab_unit')}")
        R("")
        R("  STEP 2 — KG traversal (Cypher, verbatim from 11_genmedassist_system.py)")
        R("    MATCH (d:Drug)-[r:CONTRAINDICATED_WITH]->(c:Condition)")
        R("    WHERE toLower(d.name) CONTAINS $drug")
        R("    RETURN d.name, c.name, r.threshold_lab, r.threshold_val,")
        R("           r.threshold_op, r.severity, r.action")
        R(f"    -> paths_checked  = {t['steps'][1]['paths_checked']}")
        R(f"    -> paths_violated = {t['steps'][1]['paths_violated']}")
        for v in t["steps"][1].get("violations", [])[:3]:
            R(f"       {json.dumps(v)[:150]}")
        R("")
        R("  STEP 3 — Arithmetic (pure Python, no model)")
        s4 = t["steps"][3]
        R(f"    PathViol = {t['steps'][1]['paths_violated']}/"
          f"{t['steps'][1]['paths_checked']} = {s4['path_viol']}")
        R(f"    NumDist  = {s4['num_dist']}")
        R(f"    C = {s4['w1']}*{s4['path_viol']} + {s4['w2']}*{s4['num_dist']}"
          f" = {s4['cscore']}")
        R("")
        R("  STEP 4 — Decision")
        R(f"    C = {t['cscore']}  vs  tau  ->  {t['decision']}")
        R(f"    ground truth: conflict_present = {b.get('conflict_present')}")
        R("")
        R("  NOTE FOR THE PAPER: no LLM is invoked at any step. Include this")
        R("  figure in the appendix — it answers Sdyd's Q1, Q3 and Q5 at once.")

    # ═══════════════════════════════════════════════ A10 paired examples
    R("")
    R("=" * 94)
    R(f"A10  PAIRED EXAMPLES — MedGuard flags, {args.baseline} clears")
    R("=" * 94)
    bpath = list(glob.glob(str(DATA / "eval" / f"*{args.baseline}*_results.jsonl")))
    if not bpath:
        R(f"  (no baseline file matching '{args.baseline}' in data/raw/eval/ —")
        R("   copy eval_results/*.jsonl from Drive, then re-run)")
    else:
        recs = [json.loads(l) for l in open(bpath[0], encoding="utf-8") if l.strip()]
        ik = next((k for k in ("example_id", "id", "case_id") if k in recs[0]), None)
        pk = next((k for k in ("prediction", "pred", "conflict_detected",
                               "answer", "model_answer", "output") if k in recs[0]), None)
        bl = {r[ik]: r for r in recs} if ik else {}
        found = 0
        for e, t in test:
            if found >= 8:
                break
            b = bench[e]
            r = bl.get(e)
            if not r or not b.get("conflict_present"):
                continue
            pv = str(r.get(pk, "")).lower()
            if "no" in pv or "safe" in pv or pv in ("false", "0"):
                if t["decision"] != "SAFE":
                    found += 1
                    R(f"\n  [{found}] {e}  stratum S{b.get('stratum')}")
                    R(f"    case      : {b.get('drug')} + {b.get('lab_name')}="
                      f"{b.get('lab_value')} (threshold {b.get('operator')}{b.get('threshold')})")
                    R(f"    MedGuard  : C={t['cscore']} -> {t['decision']}  CORRECT")
                    R(f"    {args.baseline:<10}: {str(r.get(pk))[:90]}  WRONG")
        if found == 0:
            R("  (none found — check the prediction key name in the baseline file)")

    with open(OUT / "day2_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT / "table5_rebuilt.json", "w", encoding="utf-8") as f:
        json.dump(rows5, f, indent=2)
    print(f"\n[saved] {OUT/'day2_report.txt'}")


if __name__ == "__main__":
    main()
