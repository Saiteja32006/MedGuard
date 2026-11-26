#!/usr/bin/env python3
"""
25_fix_trajectories.py — recompute C under corrected K/uL thresholds
=====================================================================

WHY THIS EXISTS
---------------
03_rerun_fixed.py REPLAYS path_viol / paths_violated / num_dist from the
recorded trajectories (lines 117-119); it never recomputes them from the
benchmark. Those trajectories were produced with two wrong thresholds:

    Platelet Count   <100000   should be <100     (MIMIC reports K/uL)
    WBC              <3000     should be <3.0

Both are 1000x too high, so every real value violated them. The error sits in
THREE places at once -- the Neo4j threshold edges, the redundant local
THRESHOLD_RULES table inside compute_num_dist(), and the benchmark labels --
all wrong in the same direction, which is exactly why every reported metric
agreed with itself and none could catch it.

Correcting the benchmark labels ALONE would be wrong: the scores would stay at
C~1.0 while the labels flipped to safe, turning 25 true positives into false
positives and pushing tau_hat from 0 to ~1.0. That measures the BROKEN system
against CORRECT labels. This script fixes the other side.

WHAT IT DOES
------------
For every trajectory, re-evaluates each recorded violation and the num_dist
computation against the corrected threshold:

  * a violation that no longer holds is dropped, and paths_violated decremented
    (paths_checked is left alone -- the path WAS checked)
  * path_viol   = paths_violated / max(paths_checked, 1)
  * num_dist    recomputed from num_dist_details under the corrected threshold
  * cscore      = w1 * path_viol + w2 * num_dist   (weights read per-case)

Decisions are NOT rewritten -- tau_hat is re-derived downstream by
03_rerun_fixed.py, which is the point.

USAGE
-----
    python scripts/25_fix_trajectories.py \
        --traj data/raw/genmedassist_trajectories_v1.jsonl

    python scripts/25_fix_trajectories.py \
        --traj data/raw/genmedassist_trajectories_v1.jsonl \
        --out  data/raw/genmedassist_trajectories_v2.jsonl --apply

Dry run by default. Nothing is written without --apply.
"""

import argparse
import collections
import json
import re
from pathlib import Path

# (lab lowercased, broken threshold) -> corrected threshold
CORRECTIONS = {
    ("platelet count", 100000.0): 100.0,
    ("platelets", 100000.0): 100.0,
    ("wbc", 3000.0): 3.0,
    ("white blood cells", 3000.0): 3.0,
}

_THR = re.compile(r"\s*([<>]=?)\s*([0-9.]+)")


def parse_thr(s):
    """'<100000' -> ('<', 100000.0)"""
    if s is None:
        return None, None
    if isinstance(s, (int, float)):
        return None, float(s)
    m = _THR.match(str(s))
    return (m.group(1), float(m.group(2))) if m else (None, None)


def corrected(lab, thr_val):
    if lab is None or thr_val is None:
        return None
    return CORRECTIONS.get((str(lab).strip().lower(), float(thr_val)))


def violates(op, actual, thr):
    if actual is None or thr is None or op is None:
        return False
    if op.startswith("<"):
        return actual < thr if op == "<" else actual <= thr
    return actual > thr if op == ">" else actual >= thr


def num_dist_of(op, actual, thr):
    """Normalised violation magnitude, 0 when not violated."""
    if not violates(op, actual, thr) or not thr:
        return 0.0
    d = (thr - actual) / thr if op.startswith("<") else (actual - thr) / thr
    return round(min(max(d, 0.0), 1.0), 4)


def lab_of(v):
    return v.get("lab") or v.get("lab_test") or v.get("lab_name")


# ============================================================================
def fix_one(t):
    """Returns (changed, note_dict). Mutates t in place."""
    steps = t.get("steps") or []
    kg = next((s for s in steps if s.get("name") == "graphrag"), None)
    cs = next((s for s in steps if s.get("name") == "cscore_computation"), None)
    if cs is None:
        return False, None

    note = {"example_id": t.get("example_id"),
            "cscore_before": t.get("cscore"),
            "dropped_violations": 0}
    changed = False

    # ---- 1. re-evaluate recorded KG violations -----------------------------
    if kg is not None:
        keep = []
        for v in kg.get("violations", []):
            op, thr = parse_thr(v.get("threshold"))
            new_thr = corrected(lab_of(v), thr)
            if new_thr is None:
                keep.append(v)
                continue
            changed = True
            v["threshold_original"] = v.get("threshold")
            v["threshold"] = f"{op or '<'}{new_thr:g}"
            v["unit_correction"] = "K/uL: threshold was 1000x too high"
            if violates(op, v.get("actual"), new_thr):
                keep.append(v)
            else:
                note["dropped_violations"] += 1
        if changed:
            n_drop = note["dropped_violations"]
            kg["violations"] = keep
            kg["paths_violated"] = max(0, kg.get("paths_violated", 0) - n_drop)
            # paths_checked is deliberately unchanged: the path WAS checked.

    # ---- 2. recompute num_dist --------------------------------------------
    nd = cs.get("num_dist_details") or {}
    op, thr = parse_thr(nd.get("threshold"))
    new_thr = corrected(nd.get("lab"), thr)
    if new_thr is not None:
        changed = True
        nd["threshold_original"] = nd.get("threshold")
        nd["threshold"] = f"{op or '<'}{new_thr:g}"
        nd["dist_original"] = nd.get("dist")
        nd["dist"] = num_dist_of(op, nd.get("actual"), new_thr)
        cs["num_dist_original"] = cs.get("num_dist")
        cs["num_dist"] = nd["dist"]

    if not changed:
        return False, None

    # ---- 3. recompute path_viol and cscore ---------------------------------
    if kg is not None:
        pv = kg.get("paths_violated", 0) / max(kg.get("paths_checked", 0), 1)
        cs["path_viol_original"] = cs.get("path_viol")
        cs["path_viol"] = round(pv, 4)

    w1 = cs.get("w1", 0.6075)
    w2 = cs.get("w2", 0.3925)
    c = w1 * cs.get("path_viol", 0.0) + w2 * cs.get("num_dist", 0.0)
    c = round(c, 4)
    cs["cscore_original"] = cs.get("cscore")
    cs["cscore"] = c
    t["cscore_original"] = t.get("cscore")
    t["cscore"] = c
    t["unit_correction_applied"] = True

    note.update({"cscore_after": c,
                 "path_viol": cs.get("path_viol"),
                 "num_dist": cs.get("num_dist"),
                 "stratum": t.get("stratum"),
                 "rule_id": (t.get("ground_truth") or {}).get("rule_id")})
    return True, note


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    traj_path = Path(args.traj)
    out_path = Path(args.out) if args.out else traj_path.with_name(
        traj_path.name.replace("_v1", "_v2"))

    rows = [json.loads(l) for l in open(traj_path, encoding="utf-8")]
    print("=" * 74)
    print("TRAJECTORY CORRECTION — K/uL threshold unit error")
    print("=" * 74)
    print(f"  loaded {len(rows)} trajectories from {traj_path}")
    print()

    notes = []
    for t in rows:
        ch, n = fix_one(t)
        if ch and n:
            notes.append(n)

    print(f"  trajectories corrected: {len(notes)}")
    if not notes:
        print("  Nothing matched. Check that thresholds are stored as '<100000'.")
        return

    by_rule = collections.Counter(n["rule_id"] for n in notes)
    print(f"  by rule_id: {dict(by_rule)}")
    print()

    to_zero = [n for n in notes if (n["cscore_after"] or 0) == 0]
    stays = [n for n in notes if (n["cscore_after"] or 0) > 0]
    print(f"  C -> 0.0    (now correctly SAFE)   : {len(to_zero)}")
    print(f"  C  > 0      (genuine violation)    : {len(stays)}")
    print()

    print("  sample of corrected cases:")
    print(f"    {'example_id':<16} {'rule':<10} {'C before':>9} {'C after':>9} "
          f"{'pathviol':>9} {'numdist':>8}")
    for n in (notes[:6] + notes[-4:] if len(notes) > 10 else notes):
        print(f"    {str(n['example_id'])[:15]:<16} {str(n['rule_id']):<10} "
              f"{n['cscore_before'] or 0:>9.4f} {n['cscore_after'] or 0:>9.4f} "
              f"{n['path_viol'] or 0:>9.4f} {n['num_dist'] or 0:>8.4f}")
    print()

    drops = sum(n["dropped_violations"] for n in notes)
    print(f"  KG violation records dropped (no longer violations): {drops}")
    print()

    if not args.apply:
        print("  DRY RUN — nothing written. Re-run with --apply.")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"  [written] {out_path}")

    log = out_path.with_suffix(".changelog.json")
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"source": str(traj_path),
                   "corrections": {f"{k[0]} {k[1]:g}": v
                                   for k, v in CORRECTIONS.items()},
                   "n_trajectories": len(rows),
                   "n_corrected": len(notes),
                   "n_now_zero": len(to_zero),
                   "n_still_violating": len(stays),
                   "violation_records_dropped": drops,
                   "by_rule": dict(by_rule),
                   "cases": notes}, f, indent=2, default=str)
    print(f"  [written] {log}")
    print()
    print("  NEXT — in 03_rerun_fixed.py set BOTH:")
    print("      TRAJ  = DATA / 'genmedassist_trajectories_v2.jsonl'")
    print("      BENCH = DATA / 'medconflict_bench_v3.jsonl'")
    print("  Fixing only one of the two gives a meaningless result.")


if __name__ == "__main__":
    main()
