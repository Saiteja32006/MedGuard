#!/usr/bin/env python3
"""
31_b1_analysis.py — split the B1 result by distance to threshold
=================================================================

The headline B1 numbers (S2 ~1.00 recall, S0 specificity 0.66-0.85) are
averages over cases that are mostly far from their cutoff. Two questions the
average cannot answer:

  1. Does recall hold NEAR the boundary, where the comparison is actually hard?
  2. Where do the false positives sit -- are safe cases near their threshold
     the ones being over-flagged?

This joins the B1 per-case records back to the benchmark and splits by relative
distance |v - tau| / tau.

Usage:
    python scripts/31_b1_analysis.py \
        --b1-dir out/B1 --bench data/raw/medconflict_bench_v4.jsonl
"""

import argparse
import collections
import json
from pathlib import Path

BANDS = [(0.00, 0.05, "<5%"), (0.05, 0.10, "5-10%"),
         (0.10, 0.20, "10-20%"), (0.20, 1e9, ">20%")]


def band_of(v, t):
    try:
        d = abs(float(v) - float(t)) / abs(float(t))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    for lo, hi, name in BANDS:
        if lo <= d < hi:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-dir", default="out/B1")
    ap.add_argument("--bench", default="data/raw/medconflict_bench_v4.jsonl")
    args = ap.parse_args()

    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(args.bench, encoding="utf-8")}

    files = sorted(Path(args.b1_dir).glob("B1_*.jsonl"))
    if not files:
        raise SystemExit(f"no B1_*.jsonl in {args.b1_dir}")

    print("=" * 78)
    print("B1 — RECALL AND SPECIFICITY BY DISTANCE TO THRESHOLD")
    print("=" * 78)
    print("  A model that compares should degrade near the boundary.")
    print("  A model that pattern-matches on drug identity should not.")
    print()

    out = {}
    for f in files:
        stem = f.stem.replace("B1_", "")
        model, _, cond = stem.rpartition("_")
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]

        # ---- S2 recall by band --------------------------------------------
        rec_band = collections.defaultdict(lambda: [0, 0])
        spec_band = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            b = bench.get(r["example_id"])
            if not b:
                continue
            band = band_of(b.get("lab_value"), b.get("threshold"))
            if band is None:
                continue
            if r["stratum"] == 2 and r["gt"]:
                rec_band[band][1] += 1
                if r["pred"] is True:
                    rec_band[band][0] += 1
            elif r["stratum"] == 0 and not r["gt"]:
                spec_band[band][1] += 1
                if r["pred"] is False:
                    spec_band[band][0] += 1

        print(f"  {model}  /  {cond}")
        print(f"    {'band':<10}{'S2 recall':>22}{'S0 specificity':>24}")
        for _, _, name in BANDS:
            k, n = rec_band.get(name, [0, 0])
            ks, ns = spec_band.get(name, [0, 0])
            rs = f"{k}/{n} = {k/n:.3f}" if n else "—"
            ss = f"{ks}/{ns} = {ks/ns:.3f}" if ns else "—"
            print(f"    {name:<10}{rs:>22}{ss:>24}")
        print()

        out[f"{model}|{cond}"] = {
            "recall_by_band": {k: {"correct": v[0], "n": v[1]}
                               for k, v in rec_band.items()},
            "specificity_by_band": {k: {"correct": v[0], "n": v[1]}
                                    for k, v in spec_band.items()},
        }

    Path(args.b1_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.b1_dir) / "B1_by_distance.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print("  HOW TO READ THIS")
    print("   S2 recall flat across bands, including <5%:")
    print("     the model is not doing the comparison -- it flags the drug.")
    print("     Recall of 1.00 then reflects a high flag rate, not competence,")
    print("     and the specificity column is where the cost shows up.")
    print("   S2 recall dropping in the <5% and 5-10% bands:")
    print("     the model IS comparing and fails near the boundary. That is a")
    print("     narrower but genuine version of the numerical cliff.")
    print("   S0 specificity worst in the <5% and 5-10% bands:")
    print("     over-flagging concentrates near the cutoff, which is exactly")
    print("     where a clinician least wants a false alarm.")
    print()
    print(f"  [saved] {Path(args.b1_dir) / 'B1_by_distance.json'}")


if __name__ == "__main__":
    main()
