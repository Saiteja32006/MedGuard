#!/usr/bin/env python3
"""
27_fix_trajectory_extraction.py — bake FIX_DRUGS into the released trajectories
================================================================================

PROBLEM
-------
v1's extract_entities did `example.get("drugs", [example.get("drug")])`, but the
benchmark has no "drugs" key, so the second drug was silently dropped. Every
multi-hop trajectory therefore records a single drug, no interaction path, and
paths_violated = 0.

03_rerun_fixed.py corrects this AT SCORING TIME (fix_drugs=True) but never
writes the correction back. Consequence: the released trajectory file shows
zero successful multi-hop detections, and any worked example drawn from it
displays the bug rather than the system.

WHAT THIS DOES
--------------
Applies the same correction 03_rerun_fixed.py applies, and persists it:

  step 1 entity_extraction : both drugs in drugs_raw / drugs_normalised
  step 2 graphrag          : +1 paths_checked, +1 paths_violated, and an
                             explicit interaction violation record
  step 4 cscore_computation: path_viol, num_dist, cscore recomputed
  top level                : cscore, decision

VALIDATION
----------
Running 03_rerun_fixed.py on the output with fix_drugs=False must reproduce the
FIX_DRUGS row exactly (S3 = 1.000, missed = 10, macro-F1 = 0.991 on test). If it
does not, this script and the scorer disagree and one of them is wrong.

Usage:
    python scripts/27_fix_trajectory_extraction.py
    python scripts/27_fix_trajectory_extraction.py --apply
"""

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"

W1, W2 = 0.6075, 0.3925
TAU_V1 = 0.0028

# Copied verbatim from 03_rerun_fixed.py (WBC thresholds in K/uL).
DDI_THRESHOLD_RULES = {
    "warfarin+fluconazole":       {"lab": "INR",        "op": ">", "thr": 3.5},
    "warfarin+metronidazole":     {"lab": "INR",        "op": ">", "thr": 3.5},
    "warfarin+amiodarone":        {"lab": "INR",        "op": ">", "thr": 3.5},
    "digoxin+furosemide":         {"lab": "Potassium",  "op": "<", "thr": 3.5},
    "digoxin+bumetanide":         {"lab": "Potassium",  "op": "<", "thr": 3.5},
    "digoxin+torsemide":          {"lab": "Potassium",  "op": "<", "thr": 3.5},
    "methotrexate+ibuprofen":     {"lab": "WBC",        "op": "<", "thr": 3.0},
    "methotrexate+naproxen":      {"lab": "WBC",        "op": "<", "thr": 3.0},
    "methotrexate+ketorolac":     {"lab": "WBC",        "op": "<", "thr": 3.0},
    "methotrexate+diclofenac":    {"lab": "WBC",        "op": "<", "thr": 3.0},
    "lithium+ibuprofen":          {"lab": "Creatinine", "op": ">", "thr": 1.5},
    "lithium+naproxen":           {"lab": "Creatinine", "op": ">", "thr": 1.5},
    "lisinopril+spironolactone":  {"lab": "Potassium",  "op": ">", "thr": 5.0},
    "enalapril+spironolactone":   {"lab": "Potassium",  "op": ">", "thr": 5.0},
    "ramipril+spironolactone":    {"lab": "Potassium",  "op": ">", "thr": 5.0},
    "lisinopril+eplerenone":      {"lab": "Potassium",  "op": ">", "thr": 5.0},
    "simvastatin+clarithromycin": {"lab": "ALT",        "op": ">", "thr": 120},
    "atorvastatin+clarithromycin": {"lab": "ALT",       "op": ">", "thr": 120},
    "simvastatin+erythromycin":   {"lab": "ALT",        "op": ">", "thr": 120},
    "lovastatin+clarithromycin":  {"lab": "ALT",        "op": ">", "thr": 120},
}


def norm_drug(name):
    if not name:
        return ""
    n = re.sub(r"\s*\(.*?\)", "", str(name))
    n = re.sub(r"\s+\d+(\.\d+)?\s*m?g.*", "", n, flags=re.I)
    return n.strip().lower()


def lab_match(a, b):
    if not a or not b:
        return False
    a, b = str(a).lower(), str(b).lower()
    return a in b or b in a


def violated(op, val, thr):
    if val is None or thr is None:
        return False
    return (op == "<" and val < thr) or (op == ">" and val > thr)


def ddi_check(drugs, lab_name, lab_value):
    """Identical to 03_rerun_fixed.py: returns (fires, dist, key, rule)."""
    best = (False, 0.0, None, None)
    for key, rule in DDI_THRESHOLD_RULES.items():
        a, b = key.split("+")
        if not (any(a in d for d in drugs) and any(b in d for d in drugs)):
            continue
        if not lab_match(rule["lab"], lab_name):
            continue
        dist = 0.0
        if violated(rule["op"], lab_value, rule["thr"]) and rule["thr"]:
            dist = ((rule["thr"] - lab_value) / rule["thr"] if rule["op"] == "<"
                    else (lab_value - rule["thr"]) / rule["thr"])
        if not best[0] or dist > best[1]:
            best = (True, min(dist, 1.0), key, rule)
    return best


def decide(c, tau=TAU_V1):
    if c > tau:
        return "ABSTAIN"
    if tau > 0 and c > 0 and c >= tau * 0.5:
        return "UNCERTAIN"
    return "SAFE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(DATA / "medconflict_bench_v3.jsonl"))
    ap.add_argument("--traj", default=str(DATA / "genmedassist_trajectories_v2.jsonl"))
    ap.add_argument("--out", default=str(DATA / "genmedassist_trajectories_v3.jsonl"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(args.bench, encoding="utf-8")}
    rows = [json.loads(l) for l in open(args.traj, encoding="utf-8")]

    print("=" * 74)
    print("TRAJECTORY EXTRACTION FIX — persist FIX_DRUGS into the JSONL")
    print("=" * 74)
    print(f"  {len(rows)} trajectories, {len(bench)} benchmark cases")
    print()

    n_changed = 0
    n_fired = 0
    by_stratum = collections.Counter()
    safe_with_ddi = []

    for t in rows:
        eid = t.get("example_id")
        b = bench.get(eid)
        if not b or not b.get("interacting_drug"):
            continue

        steps = {s.get("name"): s for s in t.get("steps", [])}
        ex = steps.get("entity_extraction", {}).get("output", {})
        kg = steps.get("graphrag", {})
        cs = steps.get("cscore_computation", {})
        if not (ex and kg and cs):
            continue

        d1, d2 = norm_drug(b.get("drug")), norm_drug(b["interacting_drug"])
        ex["drugs_raw"] = [b.get("drug"), b.get("interacting_drug")]
        ex["drugs_normalised"] = [d1, d2]
        ex["extraction_correction"] = ("second drug recovered; v1 read a "
                                       "non-existent 'drugs' key")
        n_changed += 1

        fires, dist, key, rule = ddi_check([d1, d2], b.get("lab_name"),
                                           b.get("lab_value"))
        if not fires:
            continue
        n_fired += 1
        by_stratum[b.get("stratum")] += 1

        kg["paths_checked"] = kg.get("paths_checked", 0) + 1
        kg["paths_violated"] = kg.get("paths_violated", 0) + 1
        kg.setdefault("violations", []).append({
            "type": "drug_interaction_threshold",
            "drug_a": b.get("drug"), "drug_b": b.get("interacting_drug"),
            "rule": key, "lab": rule["lab"],
            "threshold": f"{rule['op']}{rule['thr']:g}",
            "actual": b.get("lab_value"),
            "source": "curated_DDI_table",
        })

        if not violated(rule["op"], b.get("lab_value"), rule["thr"]):
            safe_with_ddi.append((eid, b.get("stratum"), key,
                                  b.get("lab_value")))

        pv = kg["paths_violated"] / max(kg["paths_checked"], 1)
        nd = max(cs.get("num_dist", 0.0), dist)
        c = round(W1 * pv + W2 * nd, 4)

        cs["path_viol"] = round(pv, 4)
        cs["num_dist"] = round(nd, 4)
        cs["cscore"] = c
        t["cscore"] = c
        t["decision"] = decide(c)
        t["extraction_correction_applied"] = True
        t["_justification_hint"] = (
            f"Interaction {key} with {rule['lab']} = {b.get('lab_value')} "
            f"violates threshold {rule['op']}{rule['thr']:g}.")

    # ---- reconcile the recorded decision step with the final score --------
    # 25_fix_trajectories.py and the block above both rewrite cscore and the
    # top-level decision but neither touched step 5, leaving records that
    # contradict themselves (e.g. "No conflicts detected" on an ABSTAIN case).
    n_stale = 0
    for t in rows:
        dec_step = next((s_ for s_ in t.get("steps", [])
                         if s_.get("name") == "decision"), None)
        if dec_step is None:
            continue
        final_c = t.get("cscore")
        final_d = t.get("decision") or decide(final_c or 0.0)
        if (dec_step.get("cscore") == final_c
                and dec_step.get("decision") == final_d):
            continue
        n_stale += 1
        dec_step["cscore_original"] = dec_step.get("cscore")
        dec_step["decision_original"] = dec_step.get("decision")
        dec_step["cscore"] = final_c
        dec_step["decision"] = final_d
        dec_step["tau"] = TAU_V1
        hint = t.pop("_justification_hint", None)
        if final_d == "ABSTAIN":
            dec_step["justification"] = (
                hint or f"Conflict score {final_c} exceeds abstention "
                        f"threshold {TAU_V1}; referred for human review.")
        elif final_d == "SAFE":
            dec_step["justification"] = (
                f"No constraint violated; conflict score {final_c} is at or "
                f"below the abstention threshold {TAU_V1}.")
        t["decision_step_reconciled"] = True
    for t in rows:
        t.pop("_justification_hint", None)

    print(f"  trajectories with a second drug recovered : {n_changed}")
    print(f"  interaction rule fired                    : {n_fired}")
    print(f"  by stratum                                : {dict(by_stratum)}")
    print(f"  decision steps reconciled with final score: {n_stale}")
    if n_stale:
        print("     (step 5 previously contradicted the top-level outcome --")
        print("      this also repairs the 34 unit-corrected cases from")
        print("      25_fix_trajectories.py, which had the same omission)")
    print()

    if safe_with_ddi:
        print(f"  !! {len(safe_with_ddi)} case(s) where the drug pair matches but the")
        print("     lab is NOT violated, yet paths_violated is still incremented.")
        print("     ddi_check() returns fires=True on pair+lab match alone, so an")
        print("     interaction edge counts as a violation regardless of the value.")
        print("     Report this: it is invisible today only because no SAFE case")
        print("     pairs two interacting drugs.")
        for r in safe_with_ddi[:5]:
            print(f"       {r}")
        print()

    if not args.apply:
        print("  DRY RUN — nothing written. Re-run with --apply.")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"  [written] {args.out}")
    print()
    print("  VALIDATE: point 03_rerun_fixed.py at trajectories_v3 and confirm the")
    print("  v1-column now equals the old FIX_DRUGS column (S3 = 1.000,")
    print("  missed = 10, macro-F1 = 0.991 on test). If it does not, this script")
    print("  and the scorer disagree.")


if __name__ == "__main__":
    main()