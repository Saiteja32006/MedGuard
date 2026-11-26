#!/usr/bin/env python3
"""
05_baseline_metrics.py — B3: fill the missing Table 4 rows. ZERO API COST.

Table 3 lists 7 models; Table 4 shows only 5. claude-sonnet and medgemma
results already exist in Drive. This computes their rows.

Put all *_results.jsonl from BOTH eval_results folders into data/raw/eval/:
    NIPS_2026/MedConflict-Bench/eval_results/
    NIPS_2026/CODE_NIPS/genmedassist/eval_results/

Usage:
    python scripts/05_baseline_metrics.py
"""

import glob
import json
import math
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "raw" / "eval"
BENCH = ROOT / "data" / "raw" / "medconflict_bench_v2.jsonl"
OUT = ROOT / "out"

PRED_KEYS = ["prediction", "pred", "conflict_detected", "model_conflict",
             "answer", "predicted", "model_answer", "response_label", "output"]
ID_KEYS = ["example_id", "id", "case_id"]
TRUE = {"yes", "true", "conflict", "1", 1, True, "unsafe", "y"}
FALSE = {"no", "false", "safe", "0", 0, False, "n"}


def get(rec, keys):
    for k in keys:
        if k in rec:
            return rec[k]
        if "." in k:
            cur, ok = rec, True
            for p in k.split("."):
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    ok = False
                    break
            if ok:
                return cur
    return None


def to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {str(x).lower() for x in TRUE}:
        return True
    if s in {str(x).lower() for x in FALSE}:
        return False
    if "yes" in s or "conflict" in s:
        return True
    if "no" in s or "safe" in s:
        return False
    return None


def two_prop_z(x1, n1, x2, n2):
    """Two-proportion z-test, S1 vs S2 detection."""
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval


def main():
    os.makedirs(OUT, exist_ok=True)
    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(BENCH, encoding="utf-8")}

    files = sorted(glob.glob(str(EVAL / "*_results.jsonl")))
    if not files:
        print(f"No *_results.jsonl found in {EVAL}")
        print("Copy the eval_results files from Drive into data/raw/eval/")
        return

    lines = []

    def R(s=""):
        print(s)
        lines.append(s)

    R("=" * 92)
    R("TABLE 4 — detection rates per stratum (recomputed from eval_results)")
    R("=" * 92)
    R(f"  {'Model':<20}{'S0-Spec':>9}{'S1-Cat':>9}{'S2-Num':>9}{'S3-MH':>9}"
      f"{'Cliff':>8}{'MacroF1':>9}{'p(S1vS2)':>10}")
    R("-" * 92)

    table = {}
    for path in files:
        name = os.path.basename(path).replace("_results.jsonl", "")
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        pk = next((k for k in PRED_KEYS if k in recs[0]), None)
        ik = next((k for k in ID_KEYS if k in recs[0]), None)
        if pk is None or ik is None:
            R(f"  {name:<20} !! could not find prediction/id key. keys={list(recs[0])[:8]}")
            continue

        per = defaultdict(lambda: [0, 0])
        tp = fp = tn = fn = 0
        for r in recs:
            b = bench.get(r[ik])
            if not b or b.get("split") != "test":
                continue
            pred = to_bool(get(r, [pk]))
            if pred is None:
                continue
            gt = bool(b.get("conflict_present"))
            s = b.get("stratum")
            per[s][1] += 1
            per[s][0] += int(pred)
            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1

        spec = per[0][0] and (1 - per[0][0] / per[0][1]) or (1.0 if per[0][1] else 0.0)
        det = {s: (per[s][0] / per[s][1] if per[s][1] else 0.0) for s in (1, 2, 3)}
        cliff = det[1] - det[2]
        p1 = tp / (tp + fp) if tp + fp else 0.0
        r1 = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
        p0 = tn / (tn + fn) if tn + fn else 0.0
        r0 = tn / (tn + fp) if tn + fp else 0.0
        f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
        macro = (f1 + f0) / 2
        _, pval = two_prop_z(per[1][0], per[1][1], per[2][0], per[2][1])

        R(f"  {name:<20}{spec:>9.3f}{det[1]:>9.3f}{det[2]:>9.3f}{det[3]:>9.3f}"
          f"{cliff:>+8.3f}{macro:>9.3f}{(pval if pval is not None else float('nan')):>10.4f}")
        table[name] = {"S0_spec": spec, "S1": det[1], "S2": det[2], "S3": det[3],
                       "cliff": cliff, "macro_f1": macro, "p_S1_vs_S2": pval,
                       "n_scored": tp + fp + tn + fn}

    R("-" * 92)
    R("  MedGuard v1 (paper)   1.000    0.996    0.975    0.672  +0.021    0.925")
    R("  MedGuard FIX_DRUGS    1.000    0.996    0.975    1.000  +0.021    0.991")
    R("")
    R("  NOTE: report MedGuard's own cliff (+0.021) — reviewer Sdyd caught that")
    R("  it was omitted. Also reconcile the abstract's 'ubiquitous p<0.001'")
    R("  against the per-model p-values above.")

    with open(OUT / "table4_recomputed.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT / "table4_recomputed.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"\n[saved] {OUT/'table4_recomputed.txt'}")


if __name__ == "__main__":
    main()
