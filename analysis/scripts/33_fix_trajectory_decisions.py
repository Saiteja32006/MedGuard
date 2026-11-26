#!/usr/bin/env python3
"""
33_fix_trajectory_decisions.py — recompute `decision` from `cscore`
===================================================================

25_fix_trajectories.py rewrote cscore for the 34 unit-corrected cases but
deliberately left `decision` alone, expecting it to be re-derived downstream.
27_fix_trajectory_extraction.py then reconciled step 5 against that stale
top-level value, propagating it. The released file therefore contains records
reading `cscore: 0.0, decision: ABSTAIN`, which is self-contradictory at
tau = 0.0028.

Reported metrics are unaffected -- 03_rerun_fixed.py recomputes decisions from
cscore -- but the released artefact must be self-consistent, and any analysis
that trusts the stored field (as 32_cliff_and_cases.py did) gets wrong answers.

Usage:
    python scripts/33_fix_trajectory_decisions.py
    python scripts/33_fix_trajectory_decisions.py --apply
"""
import argparse, collections, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
TAU = 0.0028


def strip_nan(o):
    """NaN/Infinity are not valid JSON; coerce to null before writing."""
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: strip_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strip_nan(v) for v in o]
    return o


def decide(c, tau=TAU):
    if c is None:
        return None
    c = float(c)
    if c > tau:
        return "ABSTAIN"
    if tau > 0 and c > 0 and c >= tau * 0.5:
        return "UNCERTAIN"
    return "SAFE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=str(DATA / "genmedassist_trajectories_v3.jsonl"))
    ap.add_argument("--out", default=str(DATA / "genmedassist_trajectories_v4.jsonl"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.traj, encoding="utf-8")]
    print("=" * 74)
    print("TRAJECTORY DECISION RECONCILIATION")
    print("=" * 74)
    print(f"  {len(rows)} trajectories, tau = {TAU}")
    print()

    changed = collections.Counter()
    examples = []
    for t in rows:
        want = decide(t.get("cscore"))
        if want is None or want == t.get("decision"):
            continue
        changed[(t.get("decision"), want)] += 1
        if len(examples) < 6:
            examples.append((t.get("example_id"), t.get("cscore"),
                             t.get("decision"), want))
        t["decision_stale"] = t.get("decision")
        t["decision"] = want
        for st in t.get("steps", []):
            if st.get("name") == "decision":
                st["decision_stale"] = st.get("decision")
                st["decision"] = want
                st["cscore"] = t.get("cscore")
                st["tau"] = TAU

    print(f"  decisions corrected: {sum(changed.values())}")
    for (old, new), n in changed.most_common():
        print(f"      {old:>9} -> {new:<9} {n}")
    print()
    for e, c, old, new in examples:
        print(f"      {e}  cscore={c}  {old} -> {new}")
    print()

    if not args.apply:
        print("  DRY RUN — nothing written. Re-run with --apply.")
        return
    n_nan = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for t in rows:
            clean = strip_nan(t)
            if clean != t:
                n_nan += 1
            f.write(json.dumps(clean, default=str, allow_nan=False) + "\n")
    print(f"  [written] {args.out}  ({len(rows)} records)")
    if n_nan:
        print(f"  ({n_nan} record(s) had NaN coerced to null)")
    print()
    print("  Then re-run 32_cliff_and_cases.py with --traj pointing at v4.")


if __name__ == "__main__":
    main()