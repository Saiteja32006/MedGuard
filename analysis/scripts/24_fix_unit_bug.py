#!/usr/bin/env python3
"""
24_fix_unit_bug.py — correct the K/uL unit error in NUM-012 / NUM-015
======================================================================

MIMIC-IV reports platelets and WBC in K/uL (219, 7.6), but two rules were
written in absolute counts:

    NUM-012  heparin      Platelet Count  < 100000   should be < 100
    NUM-015  methotrexate WBC             < 3000     should be < 3.0

Both therefore fired on 100% of real values, and because the benchmark labels
were generated from the same rule table, the labels are wrong in the same
direction as the predictions. Every metric in the submission agreed with
itself and none could have caught it.

This script does two things:

  --scan DIR    find every occurrence of the broken constants in your code,
                so no duplicated copy of the rule table is missed. The
                redundant local THRESHOLD_RULES inside compute_num_dist()
                is a known duplicate (see the A2 audit).

  --apply       write a corrected benchmark:
                  * fix the threshold field on all affected cases
                  * cases that no longer violate are relabelled SAFE
                    (stratum 0, conflict_present False, new SAFE-NUM rule id)
                  * everything else untouched
                Default is a DRY RUN; nothing is written without --apply.

Usage:
    python scripts/24_fix_unit_bug.py --scan .
    python scripts/24_fix_unit_bug.py --bench data/raw/medconflict_bench_v2.jsonl
    python scripts/24_fix_unit_bug.py --bench data/raw/medconflict_bench_v2.jsonl \\
        --out data/raw/medconflict_bench_v3.jsonl --apply
"""

import argparse
import collections
import json
import re
from pathlib import Path

# rule_id -> (broken threshold, corrected threshold, new safe rule id, lab)
FIXES = {
    "NUM-012": (100000, 100.0, "SAFE-NUM-009", "Platelet Count"),
    "NUM-015": (3000, 3.0, "SAFE-NUM-010", "WBC"),
}

SAFE_MARGIN = 0.20   # the floor the builder enforced on safe cases


# ============================================================================
def scan(root):
    print("=" * 74)
    print("SCAN — where do the broken constants appear?")
    print("=" * 74)
    pats = [(re.compile(r"\b100[_,]?000\b"), "100000 (platelet threshold)"),
            (re.compile(r"\b3[_,]?000\b"), "3000 (WBC threshold)")]
    skip = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
            ".venv", "venv", "out"}
    hits = 0
    for p in sorted(Path(root).rglob("*")):
        if p.is_dir() or any(s in p.parts for s in skip):
            continue
        if p.suffix.lower() not in {".py", ".ipynb", ".json", ".yml", ".yaml",
                                    ".cypher", ".md", ".txt"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for rx, label in pats:
                if rx.search(line):
                    print(f"  {p}:{i}")
                    print(f"      [{label}]  {line.strip()[:100]}")
                    hits += 1
    print()
    if hits:
        print(f"  {hits} occurrence(s). Check EACH one — a bare 3000 may be an")
        print("  unrelated constant (max_tokens, a row limit). Only change the")
        print("  ones that are drug-lab thresholds.")
    else:
        print("  No occurrences found. Verify you scanned the right directory.")
    print()
    print("  Known places the rule table is duplicated:")
    print("    * 03_enrich_thresholds.py        -> writes threshold-enriched")
    print("                                        edges into Neo4j")
    print("    * compute_num_dist() THRESHOLD_RULES -> the redundant local")
    print("                                        table found by the A2 audit")
    print("    * the MedConflict-Bench builder notebook")
    print()


# ============================================================================
def relabel(bench_path, out_path, apply):
    print("=" * 74)
    print("RELABEL — correct thresholds and fix wrong ground truth")
    print("=" * 74)

    rows = [json.loads(l) for l in open(bench_path, encoding="utf-8")]
    print(f"  loaded {len(rows)} cases from {bench_path}")
    print()

    before = collections.Counter(e.get("stratum") for e in rows)
    changed, flipped, kept = [], [], []

    for e in rows:
        rid = e.get("rule_id")
        if rid not in FIXES:
            continue
        broken, correct, safe_id, lab = FIXES[rid]
        old_t = e.get("threshold")
        e["threshold"] = correct
        e["unit_correction"] = {"field": "threshold", "from": old_t,
                                "to": correct, "reason": "K/uL unit error"}
        changed.append(e)

        try:
            v = float(e.get("lab_value"))
        except (TypeError, ValueError):
            continue

        if v < correct:
            kept.append((rid, v))
        else:
            e["stratum"] = 0
            e["stratum_name"] = "safe"
            e["conflict_present"] = False
            e["conflict_condition"] = "none"
            e["rule_id"] = safe_id
            e["relabelled_from"] = rid
            flipped.append((rid, safe_id, v, correct))

    for rid, (broken, correct, safe_id, lab) in FIXES.items():
        f = [x for x in flipped if x[0] == rid]
        k = [x for x in kept if x[0] == rid]
        print(f"  {rid}  ({lab})  threshold {broken} -> {correct}")
        print(f"      remain conflict : {len(k)}")
        print(f"      relabelled safe : {len(f)}  -> {safe_id}")
        if f:
            margins = sorted(abs(v - t) / t for _, _, v, t in f)
            near = [m for m in margins if m < SAFE_MARGIN]
            print(f"      new safe-case margins: min {margins[0]:.1%} "
                  f"median {margins[len(margins)//2]:.1%} max {margins[-1]:.1%}")
            if near:
                print(f"      !! {len(near)} new safe case(s) sit INSIDE the "
                      f"{SAFE_MARGIN:.0%} margin the builder enforced")
                print(f"         margins: {[f'{m:.1%}' for m in near]}")
                print(f"         These are the first near-boundary negatives in")
                print(f"         the benchmark. Report them: they test whether")
                print(f"         tau_hat=0 is structural or constructional.")
        print()

    after = collections.Counter(e.get("stratum") for e in rows)
    print(f"  stratum counts before: {dict(sorted(before.items()))}")
    print(f"  stratum counts after : {dict(sorted(after.items()))}")
    print()

    sp = collections.Counter(
        e.get("split") for e in rows if "relabelled_from" in e)
    print(f"  relabelled cases by split: {dict(sp)}")
    print("    calibration-split changes affect tau_hat derivation;")
    print("    test-split changes affect the reported detection rate.")
    print()

    s2t = [e for e in rows if e.get("stratum") == 2 and e.get("split") == "test"]
    print(f"  corrected S2 test denominator: {len(s2t)}")
    print()

    if not apply:
        print("  DRY RUN — nothing written. Re-run with --apply to write.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, default=str) + "\n")
    print(f"  [written] {out_path}")

    log = out_path.with_suffix(".changelog.json")
    with open(log, "w", encoding="utf-8") as f:
        json.dump({
            "source": str(bench_path),
            "fixes": {k: {"from": v[0], "to": v[1], "safe_rule_id": v[2],
                          "lab": v[3]} for k, v in FIXES.items()},
            "n_threshold_corrected": len(changed),
            "n_relabelled_safe": len(flipped),
            "n_remain_conflict": len(kept),
            "stratum_before": {str(k): v for k, v in sorted(before.items())},
            "stratum_after": {str(k): v for k, v in sorted(after.items())},
            "relabelled_by_split": dict(sp),
        }, f, indent=2)
    print(f"  [written] {log}")
    print()
    print("  NEXT: point 03_rerun_fixed.py at the v3 file and re-run.")
    print("  Keep v2 -- the rebuttal must show both, not silently replace one.")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=None, metavar="DIR")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.scan:
        scan(args.scan)
    if args.bench:
        out = args.out or str(Path(args.bench).with_name(
            Path(args.bench).name.replace("_v2", "_v3")))
        relabel(args.bench, out, args.apply)
    if not args.scan and not args.bench:
        ap.error("give --scan DIR and/or --bench FILE")


if __name__ == "__main__":
    main()
