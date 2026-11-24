#!/usr/bin/env python3
import os
r"""
04b_cscore_benchmark.py — FIXED v2: Run C-Score on MedConflict-Bench
=====================================================================

FIXES from v1:
  1. KG CONTRAINDICATED_WITH edges: now checks if lab value ACTUALLY
     violates the threshold before counting as PathViol. Previously
     it counted the edge existing as a violation regardless of lab value.
  2. Added categorical conflict detection: drug-allergy and drug-disease
     rules checked via ICD codes in the benchmark data.

RUN ON LAB MACHINE:
  cd C:\Users\AIML\Sai_Teja _Honors\genmedassist
  python 04b_cscore_benchmark.py

Author: Ala Sai Teja, IIIT Sricity
"""

import json
import sys
import re
from pathlib import Path
from neo4j import GraphDatabase

# ============================================================================
# CONFIG
# ============================================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "changeme")

BENCH_INPUT = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "medconflict_bench_v2.jsonl"
BENCH_OUTPUT = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "medconflict_bench_v2_scored.jsonl"

W1 = 0.7
W2 = 0.3


# ============================================================================
# THRESHOLD RULES (must check lab value against these)
# ============================================================================

THRESHOLD_RULES = {
    "metformin":      [{"lab": "eGFR",      "op": "<", "thr": 30},
                       {"lab": "eGFR",      "op": "<", "thr": 45}],
    "gabapentin":     [{"lab": "eGFR",      "op": "<", "thr": 60}],
    "dabigatran":     [{"lab": "eGFR",      "op": "<", "thr": 30}],
    "pradaxa":        [{"lab": "eGFR",      "op": "<", "thr": 30}],
    "rivaroxaban":    [{"lab": "eGFR",      "op": "<", "thr": 15}],
    "digoxin":        [{"lab": "Potassium", "op": "<", "thr": 3.5}],
    "spironolactone": [{"lab": "Potassium", "op": ">", "thr": 5.0}],
    "lisinopril":     [{"lab": "Potassium", "op": ">", "thr": 5.5}],
    "enalapril":      [{"lab": "Potassium", "op": ">", "thr": 5.5}],
    "ramipril":       [{"lab": "Potassium", "op": ">", "thr": 5.5}],
    "captopril":      [{"lab": "Potassium", "op": ">", "thr": 5.5}],
    "benazepril":     [{"lab": "Potassium", "op": ">", "thr": 5.5}],
    "acetaminophen":  [{"lab": "ALT",       "op": ">", "thr": 120}],
    "atorvastatin":   [{"lab": "ALT",       "op": ">", "thr": 120}],
    "simvastatin":    [{"lab": "ALT",       "op": ">", "thr": 120}],
    "rosuvastatin":   [{"lab": "ALT",       "op": ">", "thr": 120}],
    "pravastatin":    [{"lab": "ALT",       "op": ">", "thr": 120}],
    "lovastatin":     [{"lab": "ALT",       "op": ">", "thr": 120}],
    "warfarin":       [{"lab": "INR",       "op": ">", "thr": 4.0}],
    "heparin":        [{"lab": "Platelet Count", "op": "<", "thr": 100000}],
    "glipizide":      [{"lab": "Glucose",   "op": "<", "thr": 70}],
    "glyburide":      [{"lab": "Glucose",   "op": "<", "thr": 70}],
    "glimepiride":    [{"lab": "Glucose",   "op": "<", "thr": 70}],
    "insulin":        [{"lab": "Glucose",   "op": "<", "thr": 54}],
    "methotrexate":   [{"lab": "WBC",       "op": "<", "thr": 3000},
                       {"lab": "eGFR",      "op": "<", "thr": 30}],
    "fluconazole":    [{"lab": "INR",       "op": ">", "thr": 3.5}],
}


# ============================================================================
# CATEGORICAL RULES (drug-allergy, drug-disease — no lab needed)
# These match the benchmark's Stratum 1 rules.
# The C-score for these is purely from PathViol (KG edge exists + condition matches)
# ============================================================================

CATEGORICAL_RULES = [
    {"drugs": ["amoxicillin","ampicillin","piperacillin","nafcillin","oxacillin","penicillin"],
     "condition": "Penicillin Allergy", "type": "drug_allergy"},
    {"drugs": ["sulfamethoxazole","sulfasalazine","sulfadiazine","bactrim","septra"],
     "condition": "Sulfonamide Allergy", "type": "drug_allergy"},
    {"drugs": ["cephalexin","cefazolin","ceftriaxone","cefepime","cefuroxime","cephalosporin"],
     "condition": "Penicillin Allergy", "type": "drug_allergy_cross"},
    {"drugs": ["metformin"],
     "condition": "Heart Failure", "type": "drug_disease"},
    {"drugs": ["ibuprofen","naproxen","ketorolac","indomethacin","diclofenac","meloxicam","celecoxib"],
     "condition": "GI Bleeding", "type": "drug_disease"},
    {"drugs": ["methotrexate"],
     "condition": "Pregnancy", "type": "drug_disease"},
    {"drugs": ["warfarin","coumadin"],
     "condition": "Pregnancy", "type": "drug_disease"},
    {"drugs": ["ciprofloxacin","levofloxacin","moxifloxacin","fluoroquinolone"],
     "condition": "Myasthenia Gravis", "type": "drug_disease"},
    {"drugs": ["aspirin","asa"],
     "condition": "Analgesic Allergy", "type": "drug_allergy"},
    {"drugs": ["metoprolol","atenolol","propranolol","carvedilol","bisoprolol","nadolol","labetalol"],
     "condition": "Bradycardia", "type": "drug_disease"},
    {"drugs": ["morphine","oxycodone","hydromorphone","fentanyl","codeine","hydrocodone","tramadol","meperidine"],
     "condition": "Respiratory Failure", "type": "drug_disease"},
    {"drugs": ["glipizide","glyburide","glimepiride"],
     "condition": "Sulfonamide Allergy", "type": "drug_allergy_cross"},
]

LAB_ALIASES = {
    "eGFR (CKD-EPI 2021)": "eGFR",
    "Platelet Count": "Platelet Count",
}


def lab_matches(rule_lab, example_lab):
    """Check if a rule's lab name matches the example's lab name."""
    if not rule_lab or not example_lab:
        return False
    rl = rule_lab.lower()
    el = example_lab.lower()
    return rl in el or el in rl


def check_threshold_violated(op, lab_value, threshold):
    """Check if a lab value actually violates the threshold."""
    if lab_value is None or threshold is None:
        return False
    if op == "<" and lab_value < threshold:
        return True
    if op == ">" and lab_value > threshold:
        return True
    return False


def extract_drug_key(drug_name):
    """
    Extract the base drug name from MIMIC's drug field.
    'MetFORMIN (Glucophage)' → 'metformin'
    'Piperacillin-Tazobactam' → 'piperacillin'
    'Amoxicillin-Clavulanic Acid' → 'amoxicillin'
    'Metoprolol Succinate XL' → 'metoprolol'
    'Dabigatran Etexilate' → 'dabigatran'
    'OxyCODONE (Immediate Release)' → 'oxycodone'
    """
    if not drug_name:
        return ""
    # Remove parenthetical brand names
    name = re.sub(r'\(.*?\)', '', drug_name).strip()
    # Take first word, split on hyphen
    name = name.split()[0] if name.split() else ""
    name = name.split('-')[0]
    return name.lower()


# ============================================================================
# C-SCORE COMPUTATION
# ============================================================================

def compute_cscore(session, example):
    """
    Compute C-score for a benchmark example using Neo4j KG + local rules.

    KEY FIX: KG edges (CONTRAINDICATED_WITH) are only counted as violations
    if the patient's lab value actually violates the threshold on that edge.
    Finding the edge alone is NOT a violation — the lab must be abnormal.
    """
    drug_name = example.get("drug", "")
    interacting_drug = example.get("interacting_drug")
    lab_name_raw = example.get("lab_name")
    lab_value = example.get("lab_value")
    lab_name = LAB_ALIASES.get(lab_name_raw, lab_name_raw) if lab_name_raw else None

    drug_key = extract_drug_key(drug_name)
    stratum = example.get("stratum", -1)
    conflict_condition = example.get("conflict_condition", "")

    violations = []
    paths_checked = 0
    paths_violated = 0

    # ================================================================
    # PART A: Categorical conflict detection (Stratum 1)
    # Check if the drug + condition combination matches a known rule
    # ================================================================
    if conflict_condition and conflict_condition != "none":
        for cat_rule in CATEGORICAL_RULES:
            # Check if drug matches any in the rule's drug list
            drug_matched = any(d in drug_key or drug_key in d for d in cat_rule["drugs"] if len(d) >= 3)

            if not drug_matched:
                # Also try matching against the full drug name (lowered)
                drug_lower = drug_name.lower()
                drug_matched = any(d in drug_lower for d in cat_rule["drugs"] if len(d) >= 3)

            if drug_matched:
                # Check if condition matches
                cond_matched = (cat_rule["condition"].lower() in conflict_condition.lower() or
                               conflict_condition.lower() in cat_rule["condition"].lower())

                paths_checked += 1
                if cond_matched:
                    paths_violated += 1
                    violations.append({
                        "type": "categorical_" + cat_rule["type"],
                        "drug": drug_name,
                        "condition": conflict_condition,
                        "severity": "major",
                    })

    # ================================================================
    # PART B: KG CONTRAINDICATED_WITH / REQUIRES_MONITORING edges
    # CRITICAL FIX: Only count as violation if lab value is abnormal
    # ================================================================
    if drug_key:
        # Query KG
        kg_results = []
        try:
            records = session.run("""
                MATCH (d:Drug)-[r:CONTRAINDICATED_WITH]->(c)
                WHERE toLower(d.name) CONTAINS toLower($drug)
                RETURN c.name AS condition, r.severity AS severity,
                       r.threshold_lab AS lab, r.threshold_val AS threshold,
                       r.threshold_op AS operator
            """, drug=drug_key)
            for rec in records:
                kg_results.append(dict(rec))

            records = session.run("""
                MATCH (d:Drug)-[r:REQUIRES_MONITORING]->(c)
                WHERE toLower(d.name) CONTAINS toLower($drug)
                RETURN c.name AS condition, r.severity AS severity,
                       r.threshold_lab AS lab, r.threshold_val AS threshold,
                       r.threshold_op AS operator
            """, drug=drug_key)
            for rec in records:
                kg_results.append(dict(rec))
        except Exception:
            pass

        for kg_r in kg_results:
            paths_checked += 1
            kg_lab = kg_r.get("lab")
            kg_thr = kg_r.get("threshold")
            kg_op = kg_r.get("operator")

            # FIX: Only count as violation if lab actually violates threshold
            if kg_lab and kg_thr and kg_op and lab_value is not None:
                if lab_matches(kg_lab, lab_name):
                    try:
                        thr_val = float(kg_thr)
                        if check_threshold_violated(kg_op, lab_value, thr_val):
                            paths_violated += 1
                            violations.append({
                                "type": "kg_contraindication",
                                "condition": kg_r.get("condition"),
                                "severity": kg_r.get("severity"),
                                "lab": kg_lab,
                                "threshold": thr_val,
                                "operator": kg_op,
                                "patient_value": lab_value,
                            })
                        # else: edge exists but lab is normal → NOT a violation
                    except (ValueError, TypeError):
                        pass
            # If no lab info on the edge, don't count (these are threshold edges)

    # ================================================================
    # PART C: Local threshold rules (same as KG but faster)
    # ================================================================
    if drug_key in THRESHOLD_RULES and lab_value is not None and lab_name:
        for rule in THRESHOLD_RULES[drug_key]:
            if lab_matches(rule["lab"], lab_name):
                paths_checked += 1
                if check_threshold_violated(rule["op"], lab_value, rule["thr"]):
                    paths_violated += 1
                    violations.append({
                        "type": "threshold_rule",
                        "lab": rule["lab"],
                        "threshold": rule["thr"],
                        "operator": rule["op"],
                        "patient_value": lab_value,
                    })

    # ================================================================
    # PART D: Drug-drug interactions (Stratum 3)
    # ================================================================
    if interacting_drug:
        inter_key = extract_drug_key(interacting_drug)

        # Query KG for INTERACTS_WITH
        try:
            records = session.run("""
                MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
                WHERE toLower(d1.name) CONTAINS toLower($drug1)
                  AND toLower(d2.name) CONTAINS toLower($drug2)
                RETURN r.severity AS severity, r.mechanism AS mechanism,
                       r.affected_lab AS affected_lab
            """, drug1=drug_key, drug2=inter_key)

            interactions = [dict(rec) for rec in records]
            paths_checked += max(1, len(interactions))

            if interactions:
                paths_violated += 1  # interaction exists
                violations.append({
                    "type": "drug_interaction",
                    "drug1": drug_name,
                    "drug2": interacting_drug,
                    "severity": interactions[0].get("severity"),
                    "mechanism": interactions[0].get("mechanism", ""),
                })
        except Exception:
            paths_checked += 1

        # Check interacting drug's threshold rules
        if inter_key in THRESHOLD_RULES and lab_value is not None and lab_name:
            for rule in THRESHOLD_RULES[inter_key]:
                if lab_matches(rule["lab"], lab_name):
                    paths_checked += 1
                    if check_threshold_violated(rule["op"], lab_value, rule["thr"]):
                        paths_violated += 1
                        violations.append({
                            "type": "interaction_threshold",
                            "drug": interacting_drug,
                            "lab": rule["lab"],
                            "threshold": rule["thr"],
                            "operator": rule["op"],
                            "patient_value": lab_value,
                        })

    # ================================================================
    # COMPUTE C-SCORE
    # ================================================================
    if paths_checked == 0:
        path_viol = 0.0
    else:
        path_viol = min(paths_violated / paths_checked, 1.0)

    # NumDist: max normalized distance across violated thresholds
    num_dist = 0.0
    for v in violations:
        if "threshold" in v and "operator" in v and "patient_value" in v:
            thr = float(v["threshold"])
            val = float(v["patient_value"])
            if thr != 0:
                if v["operator"] == "<":
                    d = max(0, (thr - val) / thr)
                elif v["operator"] == ">":
                    d = max(0, (val - thr) / thr)
                else:
                    d = 0
                num_dist = max(num_dist, min(d, 1.0))

    cscore = round(W1 * path_viol + W2 * num_dist, 4)

    trace = {
        "paths_checked": paths_checked,
        "paths_violated": paths_violated,
        "path_viol": round(path_viol, 4),
        "num_dist": round(num_dist, 4),
        "cscore": cscore,
        "n_violations": len(violations),
        "violations_summary": [
            {k: v for k, v in viol.items() if k in ["type", "severity", "lab", "threshold", "patient_value", "condition"]}
            for viol in violations[:5]
        ],
    }

    return cscore, trace


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("C-Score Benchmark — FIXED v2 (proper threshold checking)")
    print("=" * 70)

    if not BENCH_INPUT.exists():
        print(f"\n❌ Not found: {BENCH_INPUT}")
        sys.exit(1)

    examples = [json.loads(line) for line in open(BENCH_INPUT, encoding="utf-8")]
    print(f"\nLoaded: {len(examples)} examples")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        result = session.run("MATCH (n:Drug) RETURN count(n) AS cnt")
        print(f"Neo4j: {result.single()['cnt']:,} Drug nodes")

        scored = []
        stats = {"s0_correct": 0, "s1_detected": 0, "s1_missed": 0,
                 "s2_detected": 0, "s2_missed": 0,
                 "s3_detected": 0, "s3_missed": 0}

        for i, ex in enumerate(examples):
            cscore, trace = compute_cscore(session, ex)
            ex["kg_cscore"] = cscore
            ex["kg_trace"] = trace
            scored.append(ex)

            s = ex.get("stratum", -1)
            is_conflict = ex.get("conflict_present", False)

            if s == 0:
                if cscore == 0: stats["s0_correct"] += 1
            elif s == 1:
                if cscore > 0: stats["s1_detected"] += 1
                else: stats["s1_missed"] += 1
            elif s == 2:
                if cscore > 0: stats["s2_detected"] += 1
                else: stats["s2_missed"] += 1
            elif s == 3:
                if cscore > 0: stats["s3_detected"] += 1
                else: stats["s3_missed"] += 1

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(examples)}] S0={stats['s0_correct']}, "
                      f"S1={stats['s1_detected']}/{stats['s1_detected']+stats['s1_missed']}, "
                      f"S2={stats['s2_detected']}/{stats['s2_detected']+stats['s2_missed']}, "
                      f"S3={stats['s3_detected']}/{stats['s3_detected']+stats['s3_missed']}")

    driver.close()

    with open(BENCH_OUTPUT, "w", encoding="utf-8") as f:
        for ex in scored:
            f.write(json.dumps(ex, default=str) + "\n")

    n_safe = sum(1 for e in examples if e["stratum"] == 0)
    print(f"\n{'='*70}")
    print(f"SCORING COMPLETE")
    print(f"{'='*70}")
    print(f"  S0 (Safe):        {stats['s0_correct']}/{n_safe} correctly scored 0")
    print(f"  S1 (Categorical): {stats['s1_detected']}/{stats['s1_detected']+stats['s1_missed']} detected")
    print(f"  S2 (Numerical):   {stats['s2_detected']}/{stats['s2_detected']+stats['s2_missed']} detected")
    print(f"  S3 (Multi-hop):   {stats['s3_detected']}/{stats['s3_detected']+stats['s3_missed']} detected")
    print(f"\n  Output: {BENCH_OUTPUT}")
    print(f"  → Copy to Google Drive → Run 05_calibrate_v3.py")


if __name__ == "__main__":
    main()
