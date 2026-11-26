#!/usr/bin/env python3
"""
02_colab_temporal_and_coverage.py  —  T0.9 + T3.1

RUN THIS ON COLAB (needs MIMIC-IV CSVs from Drive; too big for a 16GB laptop).
Paste into a notebook cell, or upload and run with %run.

T0.9  Temporal alignment audit  — was the lab drawn AFTER the prescription?
T3.1  Coverage denominator      — how much of MIMIC-IV can 19 rules even see?

Both read prescriptions.csv.gz and labevents.csv.gz in chunks.
Results checkpoint to Drive so a runtime disconnect costs nothing.
"""

import os
import json
import gzip
import pandas as pd

# ------------------------------------------------------------------ CONFIG
DRIVE = os.getenv("DATA_ROOT", ".")
MIMIC = f"{DRIVE}/Dataset/mimic-iv-hosp"
OUT = f"{DRIVE}/Tier0_Results"
os.makedirs(OUT, exist_ok=True)

CHUNK = 2_000_000

# The lab itemids your 19 rules use (from PROJECT_HANDOFF.md)
RULE_ITEMIDS = {
    50912: "Creatinine",   # -> eGFR
    50971: "Potassium",
    51237: "INR",
    50861: "ALT",
    50931: "Glucose",
    51265: "Platelet",
    51300: "WBC",
}

# Drug name fragments your rules cover (lowercase, substring match)
RULE_DRUGS = [
    "metformin", "gabapentin", "dabigatran", "rivaroxaban", "digoxin",
    "spironolactone", "lisinopril", "enalapril", "ramipril",   # ACE inhibitors
    "acetaminophen", "atorvastatin", "simvastatin", "rosuvastatin",  # statins
    "warfarin", "heparin", "levothyroxine", "glipizide", "glyburide",
    "insulin", "amiodarone", "fluconazole", "methotrexate",
]


def ckpt(name, obj):
    p = f"{OUT}/{name}.json"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {p}")


def load_prescriptions():
    print("loading prescriptions...")
    cols = ["subject_id", "hadm_id", "drug", "starttime", "stoptime"]
    df = pd.read_csv(f"{MIMIC}/prescriptions.csv.gz", usecols=cols,
                     parse_dates=["starttime", "stoptime"],
                     low_memory=False)
    df["drug_l"] = df["drug"].astype(str).str.lower()
    print(f"  {len(df):,} rows, {df.hadm_id.nunique():,} admissions")
    return df


def scan_labevents(itemids):
    """Chunked scan of labevents keeping only rule-relevant itemids."""
    print("scanning labevents (chunked)...")
    keep = []
    cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    reader = pd.read_csv(f"{MIMIC}/labevents.csv.gz", usecols=cols,
                         parse_dates=["charttime"], chunksize=CHUNK,
                         low_memory=False)
    total = 0
    for i, ch in enumerate(reader):
        total += len(ch)
        keep.append(ch[ch.itemid.isin(itemids)])
        print(f"  chunk {i}: {total:,} rows scanned", end="\r")
    labs = pd.concat(keep, ignore_index=True)
    print(f"\n  kept {len(labs):,} rule-relevant lab rows")
    return labs, total


# ============================================================== T3.1
def coverage_denominator(rx, labs, total_lab_rows):
    print("\n" + "=" * 70)
    print("T3.1  COVERAGE DENOMINATOR")
    print("=" * 70)

    covered_mask = rx.drug_l.apply(
        lambda s: any(d in s for d in RULE_DRUGS))
    drug_cov = covered_mask.mean()
    n_unique = rx.drug.nunique()
    n_unique_cov = rx.loc[covered_mask, "drug"].nunique()

    print(f"(a) DRUG COVERAGE")
    print(f"    prescription rows touching a covered drug : "
          f"{covered_mask.sum():,}/{len(rx):,} = {drug_cov:.2%}")
    print(f"    unique drug strings covered               : "
          f"{n_unique_cov:,}/{n_unique:,} = {n_unique_cov/n_unique:.2%}")

    print(f"\n(b) LAB COVERAGE")
    print(f"    rule-relevant lab rows : {len(labs):,}/{total_lab_rows:,} "
          f"= {len(labs)/total_lab_rows:.2%}")
    print(f"    lab tests covered      : {len(RULE_ITEMIDS)} itemids")

    # (c) opportunity coverage: admissions where a rule is even evaluable
    adm_with_drug = set(rx.loc[covered_mask, "hadm_id"].dropna().unique())
    adm_with_lab = set(labs["hadm_id"].dropna().unique())
    adm_all = set(rx["hadm_id"].dropna().unique())
    evaluable = adm_with_drug & adm_with_lab

    print(f"\n(c) OPPORTUNITY COVERAGE")
    print(f"    admissions with any prescription     : {len(adm_all):,}")
    print(f"    ...with a covered drug               : {len(adm_with_drug):,}")
    print(f"    ...AND a rule-relevant lab           : {len(evaluable):,}")
    print(f"    evaluable fraction                   : "
          f"{len(evaluable)/max(len(adm_all),1):.2%}")
    print("\n    ^ THIS is the number to report. It measures how often the")
    print("      system has anything at all to say about an admission.")

    res = {
        "drug_row_coverage": float(drug_cov),
        "unique_drugs_total": int(n_unique),
        "unique_drugs_covered": int(n_unique_cov),
        "lab_row_coverage": float(len(labs) / max(total_lab_rows, 1)),
        "n_lab_tests_covered": len(RULE_ITEMIDS),
        "admissions_total": len(adm_all),
        "admissions_with_covered_drug": len(adm_with_drug),
        "admissions_evaluable": len(evaluable),
        "opportunity_coverage": len(evaluable) / max(len(adm_all), 1),
    }
    ckpt("T3_1_coverage", res)
    return res


# ============================================================== T0.9
def temporal_audit(rx, labs):
    print("\n" + "=" * 70)
    print("T0.9  TEMPORAL ALIGNMENT AUDIT")
    print("=" * 70)

    covered = rx[rx.drug_l.apply(lambda s: any(d in s for d in RULE_DRUGS))]
    covered = covered.dropna(subset=["hadm_id", "starttime"])
    labs2 = labs.dropna(subset=["hadm_id", "charttime", "valuenum"])

    m = covered.merge(labs2, on="hadm_id", suffixes=("_rx", "_lab"))
    if m.empty:
        print("  no overlapping admissions found — check column names")
        return {}

    m["delta_h"] = (m["charttime"] - m["starttime"]).dt.total_seconds() / 3600.0

    before = (m.delta_h < 0).mean()        # lab drawn BEFORE prescription
    after = (m.delta_h > 0).mean()         # lab drawn AFTER prescription
    within24 = (m.delta_h.abs() <= 24).mean()

    print(f"  drug-lab pairs examined       : {len(m):,}")
    print(f"  lab drawn BEFORE prescription : {before:.2%}  "
          f"(conflict was knowable at prescribing time)")
    print(f"  lab drawn AFTER prescription  : {after:.2%}  "
          f"(conflict was NOT knowable -> not a prescribing error)")
    print(f"  within +/- 24h                : {within24:.2%}")
    print("\n  delta (hours, lab - prescription) percentiles:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"    p{int(q*100):<3} {m.delta_h.quantile(q):>12.1f}")

    print("\n  ^ If a large share of your labelled conflicts come from labs")
    print("    drawn AFTER the prescription, the 68.5% conflict rate is")
    print("    inflated by cases that were not prescribing errors at all.")
    print("    State your pairing rule explicitly in the paper.")

    res = {
        "n_pairs": int(len(m)),
        "frac_lab_before_rx": float(before),
        "frac_lab_after_rx": float(after),
        "frac_within_24h": float(within24),
        "delta_h_p50": float(m.delta_h.quantile(0.5)),
        "delta_h_p05": float(m.delta_h.quantile(0.05)),
        "delta_h_p95": float(m.delta_h.quantile(0.95)),
    }
    ckpt("T0_9_temporal", res)
    return res


def main():
    rx = load_prescriptions()
    labs, total_lab_rows = scan_labevents(set(RULE_ITEMIDS))
    coverage_denominator(rx, labs, total_lab_rows)
    temporal_audit(rx, labs)
    print("\nDONE. Results checkpointed to", OUT)


if __name__ == "__main__":
    main()
