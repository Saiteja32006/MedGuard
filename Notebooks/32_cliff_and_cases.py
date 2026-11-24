#!/usr/bin/env python3
"""
32_cliff_and_cases.py — rebuild the cliff table, and find illustrative cases
=============================================================================

Two deliverables from the baseline result files.

PART A — the cliff table  (Sdyd W1: "the authors just dont report it")
    Per-stratum detection for every evaluated model, plus MedGuard's own row,
    plus each model's own S1-to-S2 drop. The submission reported baseline
    cliffs but not its own; this reports all of them on the same axis.

PART B — illustrative cases  (Jn3J Q3: "cases where MedGuard correctly
    abstained and the baseline model confidently made mistakes")
    Cases where MedGuard abstained and was right, and a baseline returned a
    confident negative. Ranked by how many baselines missed each case, so the
    most illustrative appear first.

Usage:
    python scripts/32_cliff_and_cases.py
    python scripts/32_cliff_and_cases.py --top 15
"""

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "out"

NAMES = {0: "safe (specificity)", 1: "categorical", 2: "numerical",
         3: "multi-hop"}


def load_baselines(eval_dir):
    models = {}
    for f in sorted(Path(eval_dir).glob("*_results.jsonl")):
        name = f.stem.replace("_results", "")
        models[name] = [json.loads(l) for l in open(f, encoding="utf-8")]
    return models


def per_stratum(recs):
    out = {}
    for s in sorted({r.get("stratum") for r in recs if r.get("stratum") is not None}):
        sub = [r for r in recs if r.get("stratum") == s]
        if s == 0:
            ok = sum(1 for r in sub if not r.get("predicted_conflict"))
        else:
            ok = sum(1 for r in sub if r.get("predicted_conflict"))
        out[s] = (ok / max(len(sub), 1), len(sub))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(DATA / "eval"))
    ap.add_argument("--bench", default=str(DATA / "medconflict_bench_v4.jsonl"))
    ap.add_argument("--traj", default=str(DATA / "genmedassist_trajectories_v3.jsonl"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--medguard", default="0.996,0.973,1.000,1.000",
                    help="S1,S2,S3,specificity for MedGuard (test split)")
    args = ap.parse_args()

    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(args.bench, encoding="utf-8")}
    traj = {json.loads(l)["example_id"]: json.loads(l)
            for l in open(args.traj, encoding="utf-8")}
    models = load_baselines(args.eval_dir)

    # ------------------------------------------------------------- PART A
    print("=" * 78)
    print("PART A — CLIFF TABLE")
    print("=" * 78)
    print(f"  {'model':<18}{'S0 spec':>9}{'S1 cat':>9}{'S2 num':>9}"
          f"{'S3 multi':>10}{'cliff S1-S2':>13}")
    rows = []
    for name, recs in models.items():
        ps = per_stratum(recs)
        s0 = ps.get(0, (None,))[0]
        s1 = ps.get(1, (None,))[0]
        s2 = ps.get(2, (None,))[0]
        s3 = ps.get(3, (None,))[0]
        cliff = (s1 - s2) if (s1 is not None and s2 is not None) else None
        rows.append((name, s0, s1, s2, s3, cliff))
        fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else "—"
        print(f"  {name:<18}{fmt(s0):>9}{fmt(s1):>9}{fmt(s2):>9}"
              f"{fmt(s3):>10}{fmt(cliff):>13}")

    mg = [float(x) for x in args.medguard.split(",")]
    mg_s1, mg_s2, mg_s3, mg_s0 = mg
    print(f"  {'MedGuard (ours)':<18}{mg_s0:>9.3f}{mg_s1:>9.3f}{mg_s2:>9.3f}"
          f"{mg_s3:>10.3f}{mg_s1-mg_s2:>13.3f}")
    print()
    print("  Our own cliff is reported here alongside the baselines. The")
    print("  submission omitted it, which the reviewer is right to flag.")
    print()

    # ------------------------------------------------------------- PART B
    print("=" * 78)
    print("PART B — CASES MEDGUARD CAUGHT AND BASELINES MISSED")
    print("=" * 78)

    missed_by = collections.defaultdict(list)
    for name, recs in models.items():
        for r in recs:
            eid = r.get("example_id")
            if not r.get("ground_truth_conflict"):
                continue
            if r.get("predicted_conflict"):
                continue
            t = traj.get(eid)
            if not t or t.get("decision") != "ABSTAIN":
                continue
            missed_by[eid].append((name, r.get("predicted_explanation") or ""))

    ranked = sorted(missed_by.items(), key=lambda kv: -len(kv[1]))
    print(f"  {len(ranked)} conflict case(s) MedGuard abstained on that at least")
    print(f"  one baseline cleared as safe.")
    print()

    lines = ["# Illustrative cases — MedGuard abstained, baselines cleared",
             "",
             "Cases where the ground truth is a conflict, MedGuard abstained,",
             "and one or more evaluated models returned a confident negative.",
             "Ranked by number of models that missed the case.",
             ""]

    for eid, missers in ranked[:args.top]:
        b = bench.get(eid, {})
        t = traj.get(eid, {})
        print(f"  {eid}  missed by {len(missers)}/{len(models)}")
        print(f"     {b.get('drug')}"
              f"{' + ' + str(b.get('interacting_drug')) if b.get('interacting_drug') else ''}"
              f"   {b.get('lab_name')} = {b.get('lab_value')} "
              f"(rule {b.get('rule_id')}: {b.get('operator')}{b.get('threshold')})")
        print(f"     MedGuard C = {t.get('cscore')} -> {t.get('decision')}")
        print(f"     missed by: {', '.join(m for m, _ in missers)}")
        print()

        lines += [
            f"## `{eid}` — missed by {len(missers)} of {len(models)} models", "",
            f"- **Order:** {b.get('drug')}"
            + (f" + {b.get('interacting_drug')}" if b.get("interacting_drug") else ""),
            f"- **Lab:** {b.get('lab_name')} = {b.get('lab_value')}",
            f"- **Rule:** {b.get('rule_id')} ({b.get('operator')}{b.get('threshold')})",
            f"- **MedGuard:** C = {t.get('cscore')} → {t.get('decision')} (correct)",
            f"- **Cleared as safe by:** {', '.join(m for m, _ in missers)}", ""]
        for m, expl in missers[:3]:
            if expl:
                lines += [f"  > *{m}:* {str(expl)[:220]}", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "illustrative_cases.md").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT / "cliff_table.json", "w") as f:
        json.dump({"baselines": [
            {"model": r[0], "S0": r[1], "S1": r[2], "S2": r[3], "S3": r[4],
             "cliff": r[5]} for r in rows],
            "medguard": {"S0": mg_s0, "S1": mg_s1, "S2": mg_s2, "S3": mg_s3,
                         "cliff": mg_s1 - mg_s2}}, f, indent=2)
    print(f"  [saved] {OUT / 'illustrative_cases.md'}")
    print(f"  [saved] {OUT / 'cliff_table.json'}")
    print()
    print("  CAUTION: these baseline numbers come from the ORIGINAL evaluation,")
    print("  whose prompt showed a patient-history block only for categorical")
    print("  cases. Any S1-vs-S2 comparison across strata inherits that")
    print("  confound. Report the table with that caveat, or re-run the")
    print("  baselines under the uniform B1 prompt.")


if __name__ == "__main__":
    main()
