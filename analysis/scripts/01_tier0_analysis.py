#!/usr/bin/env python3
"""
01_tier0_analysis.py  —  Tier 0 diagnostics (T0.1 - T0.8)

Runs entirely on your laptop. No Neo4j, no GPU, no MIMIC-IV CSVs.
Inputs: the trajectory JSONL + benchmark JSONL you already have.

Usage (from anywhere — paths resolve to the project root):
    python scripts/01_tier0_analysis.py --tau 0.003

Outputs (in out/):
    tier0_report.txt          human-readable report  <- read this
    tier0_results.json        machine-readable numbers
    t0_4_missed_conflicts.csv every conflict scored C=0
    t0_5_s3_failures.csv      S3 misses with cause classification
    t0_2_extraction_fail.csv  extraction failures
"""

import argparse
import csv
import glob
import gzip
import json
import math
import os
from pathlib import Path
from collections import Counter, defaultdict

# Project root = parent of scripts/ — so this works from any CWD
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "out"

# ============================================================================
# FIELD_MAP — EDIT THIS after running 00_inspect_schema.py
# Each entry is a list of candidate key paths (dotted). First match wins.
# Add your real key path to the FRONT of the relevant list.
# ============================================================================
FIELD_MAP = {
    "case_id":      ["case_id", "id", "example_id", "uid", "index"],
    "stratum":      ["stratum", "strata", "conflict_type", "type", "label_type",
                     "meta.stratum"],
    "gt_label":     ["label", "ground_truth", "gt", "y", "conflict_label",
                     "is_conflict", "meta.label"],
    "c_score":      ["c_score", "C", "conflict_score", "score", "cscore",
                     "scores.c_score"],
    "pathviol":     ["pathviol", "path_viol", "PathViol", "scores.pathviol",
                     "path_violation", "path_violation_score"],
    "numdist":      ["numdist", "num_dist", "NumDist", "scores.numdist",
                     "numerical_distance"],
    "decision":     ["decision", "prediction", "pred", "final_decision",
                     "outcome", "action"],
    "drugs":        ["drugs", "entities.drugs", "extracted.drugs", "drug",
                     "entities.drug"],
    "labs":         ["labs", "entities.labs", "extracted.labs", "lab_values",
                     "entities.lab_values"],
    "cuis":         ["cuis", "entities.cuis", "drug_cuis", "cui"],
    "kg_paths":     ["kg_paths", "paths", "graph_paths", "paths_found",
                     "kg.paths"],
    "kg_queries":   ["kg_queries", "queries", "cypher", "kg.queries"],
    "rule_id":      ["rule_id", "rule", "matched_rule", "rule_name",
                     "meta.rule_id"],
    "egfr":         ["egfr", "eGFR", "labs.egfr", "entities.egfr"],
}

# Values that mean "this case contains a genuine conflict"
CONFLICT_TOKENS = {"conflict", "yes", "true", "1", "unsafe", "positive", 1, True}
SAFE_TOKENS = {"safe", "no", "false", "0", "negative", 0, False}

# Decision vocabulary
FLAG_TOKENS = {"abstain", "flag", "conflict", "flagged", "abstained", "unsafe"}
SAFE_DEC_TOKENS = {"safe", "pass", "clear", "accept", "ok"}
UNCERTAIN_TOKENS = {"uncertain", "review", "request_review", "unsure"}

EPS = 1e-12


# ---------------------------------------------------------------- utilities
def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_records(path):
    with open_maybe_gzip(path) as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            try:
                for rec in json.load(f):
                    yield rec
            except Exception:
                return
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def dig(rec, path):
    """Fetch a dotted key path from a nested dict. Returns (found, value)."""
    cur = rec
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def resolve(rec, logical):
    """Resolve a logical field to its value using FIELD_MAP candidates."""
    for cand in FIELD_MAP.get(logical, []):
        found, val = dig(rec, cand)
        if found:
            return val
    return None


def resolved_key(records, logical):
    """Which candidate key actually resolved (for the report)."""
    for cand in FIELD_MAP.get(logical, []):
        for r in records[:200]:
            if dig(r, cand)[0]:
                return cand
    return None


def norm(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower()
    return v


def is_conflict(gt):
    n = norm(gt)
    if n is None:
        return None
    if n in CONFLICT_TOKENS:
        return True
    if n in SAFE_TOKENS:
        return False
    if isinstance(n, str):
        if n.startswith("s0") or "safe" in n:
            return False
        if n.startswith(("s1", "s2", "s3")):
            return True
    return None


def decision_bucket(d):
    n = norm(d)
    if n is None:
        return None
    if isinstance(n, str):
        for t in UNCERTAIN_TOKENS:
            if t in n:
                return "UNCERTAIN"
        for t in FLAG_TOKENS:
            if t in n:
                return "FLAG"
        for t in SAFE_DEC_TOKENS:
            if t in n:
                return "SAFE"
    return str(n).upper()


def stratum_of(rec):
    s = resolve(rec, "stratum")
    n = norm(s)
    if n is None:
        return "UNKNOWN"
    n = str(n)
    for tag in ("s0", "s1", "s2", "s3"):
        if n.startswith(tag) or n == tag:
            return tag.upper()
    if "safe" in n:
        return "S0"
    if "categor" in n:
        return "S1"
    if "numer" in n or "threshold" in n:
        return "S2"
    if "multi" in n or "hop" in n:
        return "S3"
    return str(s)


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def nonempty(v):
    if v is None:
        return False
    if isinstance(v, (list, dict, str)):
        return len(v) > 0
    return True


# ---------------------------------------------------------------- analyses
def run(records, tau, out_dir, report):
    R = report
    results = {}
    n = len(records)

    R(f"records loaded: {n}")
    R(f"tau_hat used  : {tau}")
    R("")
    R("RESOLVED FIELDS")
    for logical in FIELD_MAP:
        k = resolved_key(records, logical)
        status = k if k else "*** NOT FOUND ***"
        R(f"  {logical:<12} -> {status}")
    R("")

    # ---- derived per-record view
    rows = []
    for r in records:
        c = as_float(resolve(r, "c_score"))
        rows.append({
            "case_id":  resolve(r, "case_id"),
            "stratum":  stratum_of(r),
            "gt":       is_conflict(resolve(r, "gt_label")),
            "c":        c,
            "pathviol": as_float(resolve(r, "pathviol")),
            "numdist":  as_float(resolve(r, "numdist")),
            "decision": decision_bucket(resolve(r, "decision")),
            "drugs":    resolve(r, "drugs"),
            "labs":     resolve(r, "labs"),
            "cuis":     resolve(r, "cuis"),
            "paths":    resolve(r, "kg_paths"),
            "rule_id":  resolve(r, "rule_id"),
            "egfr":     as_float(resolve(r, "egfr")),
            "_raw":     r,
        })

    have_c = sum(1 for x in rows if x["c"] is not None)
    R(f"records with a usable C-score: {have_c}/{n}")
    if have_c == 0:
        R("!! No C-score resolved. Fix FIELD_MAP['c_score'] and re-run.")
        return results, rows
    R("")

    # ================================================== T0.1 Uncertain band
    R("=" * 74)
    R("T0.1  UNCERTAIN-BAND COUNT")
    R("=" * 74)
    lo, hi = tau / 2.0, tau
    band = [x for x in rows if x["c"] is not None and lo - EPS <= x["c"] <= hi + EPS]
    R(f"  band = [{lo:.6f}, {hi:.6f}]")
    R(f"  cases in band : {len(band)}")
    if band:
        R(f"  by stratum    : {dict(Counter(x['stratum'] for x in band))}")
        R(f"  example C     : {[x['c'] for x in band[:10]]}")
    else:
        R("  -> ZERO cases. The three-way decision (Safe/Uncertain/Abstain)")
        R("     describes a state the system cannot enter. REPORT THIS.")
    # smallest nonzero score actually achievable
    nz = sorted({x["c"] for x in rows if x["c"] is not None and x["c"] > EPS})
    if nz:
        R(f"  smallest nonzero C observed: {nz[0]:.6f}")
        if nz[0] > hi:
            R(f"  -> smallest nonzero C ({nz[0]:.6f}) EXCEEDS band top ({hi:.6f}).")
            R("     The Uncertain band is unreachable by construction.")
    results["T0_1"] = {
        "band_lo": lo, "band_hi": hi, "n_in_band": len(band),
        "smallest_nonzero_c": nz[0] if nz else None,
        "band_by_stratum": dict(Counter(x["stratum"] for x in band)),
    }
    R("")

    # ============================================ T0.3 confusion matrix
    R("=" * 74)
    R("T0.3  DECISION-LEVEL CONFUSION MATRIX")
    R("=" * 74)
    cm = defaultdict(int)
    for x in rows:
        if x["gt"] is None or x["decision"] is None:
            continue
        gtk = "conflict" if x["gt"] else "safe"
        cm[(x["decision"], gtk)] += 1
    if cm:
        decs = sorted({d for d, _ in cm})
        R(f"  {'decision':<12} {'safe':>8} {'conflict':>10}")
        for d in decs:
            R(f"  {d:<12} {cm[(d,'safe')]:>8} {cm[(d,'conflict')]:>10}")
        R("")
        R("  NOTE for the paper: state explicitly whether FLAG/ABSTAIN counts as")
        R("  a positive detection in Macro-F1, and how UNCERTAIN is scored.")
    else:
        R("  !! decision or gt_label unresolved — fix FIELD_MAP.")
    results["T0_3"] = {f"{d}|{g}": v for (d, g), v in cm.items()}
    R("")

    # =================================== T0.4 conflicts scored exactly zero
    R("=" * 74)
    R("T0.4  CONFLICTS SCORED C = 0  (the invariant misses)")
    R("=" * 74)
    missed = [x for x in rows if x["gt"] is True and x["c"] is not None
              and abs(x["c"]) <= EPS]
    R(f"  conflicts with C == 0 : {len(missed)}")
    R(f"  by stratum            : {dict(Counter(x['stratum'] for x in missed))}")
    below_tau = [x for x in rows if x["gt"] is True and x["c"] is not None
                 and x["c"] <= tau + EPS]
    R(f"  conflicts with C <= tau ({tau}) : {len(below_tau)}")
    R("  -> if these two counts are equal, NO threshold can catch them.")
    R("     They are not near-misses; the rule engine found nothing at all.")
    _dump_csv(os.path.join(out_dir, "t0_4_missed_conflicts.csv"), missed,
              ["case_id", "stratum", "c", "pathviol", "numdist", "decision",
               "rule_id"])
    results["T0_4"] = {
        "n_conflict_c_zero": len(missed),
        "n_conflict_below_tau": len(below_tau),
        "by_stratum": dict(Counter(x["stratum"] for x in missed)),
    }
    R("")

    # ================================= T0.5 S3 failure cause classification
    R("=" * 74)
    R("T0.5  S3 FAILURE CLASSIFICATION")
    R("=" * 74)
    s3_missed = [x for x in rows if x["stratum"] == "S3" and x["gt"] is True
                 and x["c"] is not None and x["c"] <= tau + EPS]
    R(f"  S3 conflicts not flagged: {len(s3_missed)}")
    causes = Counter()
    for x in s3_missed:
        drugs = x["drugs"]
        cuis = x["cuis"]
        paths = x["paths"]
        ndrug = len(drugs) if isinstance(drugs, list) else (1 if nonempty(drugs) else 0)
        ncui = len(cuis) if isinstance(cuis, list) else (1 if nonempty(cuis) else 0)
        npath = len(paths) if isinstance(paths, list) else (1 if nonempty(paths) else 0)

        if ndrug < 2:
            cause = "A_drug_normalization_miss (<2 drugs extracted)"
        elif ncui and ncui < ndrug:
            cause = "A_drug_normalization_miss (CUI missing for a drug)"
        elif not nonempty(x["rule_id"]):
            cause = "B_pairing_no_rule_matched"
        elif npath == 0:
            cause = "C_query_unreachable (rule matched, 0 paths returned)"
        else:
            cause = "D_true_edge_absence_or_other"
        causes[cause] += 1
        x["_cause"] = cause
    for c, k in causes.most_common():
        R(f"    {c:<52} {k}")
    if not s3_missed:
        R("    (none — check stratum/gt resolution)")
    R("")
    R("  Interpretation: A/B are EXTRACTION-stage problems, C is a QUERY-design")
    R("  problem, D is edge density. Section 7.2 currently claims D. If A/B/C")
    R("  dominate, 7.2 is wrong and richer ontologies would NOT fix it.")
    _dump_csv(os.path.join(out_dir, "t0_5_s3_failures.csv"), s3_missed,
              ["case_id", "stratum", "c", "pathviol", "numdist", "rule_id",
               "_cause"])
    results["T0_5"] = dict(causes)
    R("")

    # =================================== T0.2 extraction-failure audit
    R("=" * 74)
    R("T0.2  EXTRACTION-FAILURE AUDIT")
    R("=" * 74)
    fails = []
    n_no_drug = n_no_cui = n_no_lab = n_no_egfr = 0
    for x in rows:
        problems = []
        if not nonempty(x["drugs"]):
            problems.append("no_drug_extracted"); n_no_drug += 1
        if x["cuis"] is not None and not nonempty(x["cuis"]):
            problems.append("no_cui_match"); n_no_cui += 1
        if not nonempty(x["labs"]):
            problems.append("no_lab_extracted"); n_no_lab += 1
        if x["egfr"] is not None and (math.isnan(x["egfr"]) if isinstance(x["egfr"], float) else False):
            problems.append("egfr_nan"); n_no_egfr += 1
        if problems:
            x["_problems"] = ";".join(problems)
            fails.append(x)
    R(f"  cases with >=1 extraction problem : {len(fails)}")
    R(f"    no drug extracted   : {n_no_drug}")
    R(f"    no CUI match        : {n_no_cui}")
    R(f"    no lab extracted    : {n_no_lab}")
    R(f"    eGFR NaN            : {n_no_egfr}")
    silent = [x for x in fails if x["c"] is not None and x["c"] <= tau + EPS]
    R(f"  of those, cleared as SAFE (C <= tau) : {len(silent)}")
    R("  -> THIS is the failure mode oJaU/AC-3 asked for: a failed extraction")
    R("     yields C = 0 and clears WITHOUT abstaining. Report the count.")
    _dump_csv(os.path.join(out_dir, "t0_2_extraction_fail.csv"), fails,
              ["case_id", "stratum", "gt", "c", "decision", "_problems"])
    results["T0_2"] = {
        "n_with_problem": len(fails), "no_drug": n_no_drug,
        "no_cui": n_no_cui, "no_lab": n_no_lab, "egfr_nan": n_no_egfr,
        "silently_cleared": len(silent),
    }
    R("")

    # ============================= T0.6 S0 decomposition (real vs vacuous)
    R("=" * 74)
    R("T0.6  S0 DECOMPOSITION — real negatives vs vacuous negatives")
    R("=" * 74)
    s0 = [x for x in rows if x["gt"] is False]
    checked = [x for x in s0 if nonempty(x["rule_id"]) or nonempty(x["paths"])]
    vacuous = [x for x in s0 if x not in checked]
    R(f"  safe cases total            : {len(s0)}")
    R(f"  rule evaluated, not violated: {len(checked)}   <- REAL negatives")
    R(f"  nothing found to check      : {len(vacuous)}   <- VACUOUS negatives")
    R("  -> only the first group supports a specificity claim.")
    results["T0_6"] = {"safe_total": len(s0), "real_negative": len(checked),
                       "vacuous_negative": len(vacuous)}
    R("")

    # ================================== T0.7 label vs eval consistency
    R("=" * 74)
    R("T0.7  LABEL-vs-EVAL CONSISTENCY")
    R("=" * 74)
    R(f"  cases labelled CONFLICT but scored C=0 : {len(missed)}")
    with_rule = [x for x in missed if nonempty(x["rule_id"])]
    R(f"    ...of which a rule_id IS recorded    : {len(with_rule)}")
    R(f"    ...of which NO rule_id recorded      : {len(missed) - len(with_rule)}")
    R("  Label generation found a violation; evaluation did not.")
    R("  Same rules, same data -> pipeline inconsistency, not a reasoning limit.")
    R("  ACTION: open 5-10 rows of t0_4_missed_conflicts.csv and diff the two")
    R("  code paths (label generator vs evaluator) for those case_ids.")
    results["T0_7"] = {"conflict_c_zero": len(missed),
                       "with_rule_id": len(with_rule)}
    R("")

    # ================================================= T0.8 rule inventory
    R("=" * 74)
    R("T0.8  RULE INVENTORY")
    R("=" * 74)
    rule_ids = Counter(str(x["rule_id"]) for x in rows if nonempty(x["rule_id"]))
    R(f"  distinct rule_ids observed in trajectories: {len(rule_ids)}")
    R("  (paper says 19 in Limitations, 36 in Figure 2 — reconcile)")
    for rid, k in rule_ids.most_common(50):
        R(f"    {rid:<45} {k}")
    results["T0_8"] = {"n_distinct_rules": len(rule_ids),
                       "rule_usage": dict(rule_ids)}
    R("")

    # ============================== bonus: C-score distribution (feeds T1.1)
    R("=" * 74)
    R("BONUS  C-SCORE DISTRIBUTION (feeds T1.1 bimodality argument)")
    R("=" * 74)
    safe_c = [x["c"] for x in rows if x["gt"] is False and x["c"] is not None]
    conf_c = [x["c"] for x in rows if x["gt"] is True and x["c"] is not None]
    if safe_c:
        R(f"  SAFE  n={len(safe_c)}  min={min(safe_c):.6f} max={max(safe_c):.6f}"
          f"  #exactly_zero={sum(1 for v in safe_c if abs(v) <= EPS)}")
    if conf_c:
        R(f"  CONF  n={len(conf_c)}  min={min(conf_c):.6f} max={max(conf_c):.6f}"
          f"  #exactly_zero={sum(1 for v in conf_c if abs(v) <= EPS)}")
    if safe_c and all(abs(v) <= EPS for v in safe_c):
        R("  -> EVERY safe case scores exactly 0. tau_hat is a quantile of a")
        R("     constant. This is structural, not a small-sample artifact.")
    results["BONUS"] = {
        "safe_all_zero": bool(safe_c) and all(abs(v) <= EPS for v in safe_c),
        "n_safe": len(safe_c), "n_conflict": len(conf_c),
    }
    R("")

    return results, rows


def _dump_csv(path, rows, cols):
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
    except Exception as e:
        print(f"  (csv write failed for {path}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--tau", type=float, default=0.003)
    ap.add_argument("--file", default=None,
                    help="specific trajectory file; otherwise auto-selected")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.file:
        path = args.file
    else:
        cands = []
        for p in ("*.jsonl", "*.jsonl.gz", "*.json"):
            cands.extend(glob.glob(os.path.join(args.data_dir, "**", p),
                                   recursive=True))
        if not cands:
            print("No JSONL found under", args.data_dir)
            print("See DOWNLOAD_CHECKLIST.md for what to pull from Drive.")
            return
        # Prefer a scored/trajectory file over a bare benchmark file
        preferred = [c for c in cands
                     if any(t in os.path.basename(c).lower()
                            for t in ("trajector", "scored"))]
        pool = preferred if preferred else cands
        path = max(pool, key=os.path.getsize)

    print(f"[using] {path}")
    records = list(iter_records(path))

    lines = []

    def report(s=""):
        print(s)
        lines.append(s)

    report("=" * 74)
    report("TIER 0 DIAGNOSTIC REPORT")
    report("=" * 74)
    report(f"source: {path}")
    report("")

    results, rows = run(records, args.tau, args.out_dir, report)

    rp = os.path.join(args.out_dir, "tier0_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    jp = os.path.join(args.out_dir, "tier0_results.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[saved] {rp}")
    print(f"[saved] {jp}")


if __name__ == "__main__":
    main()
