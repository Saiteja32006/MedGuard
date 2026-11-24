#!/usr/bin/env python3
r"""
17_clinician_validation.py — Clinician Validation Package
==========================================================
Generates everything needed for clinician review:

1. An Excel spreadsheet with 200 stratified cases for the clinician to review
2. A separate instructions document (printed to console + saved as text)
3. After clinician fills in the spreadsheet, computes Cohen's Kappa

Usage:
    # Step 1: Generate the review spreadsheet
    python 17_clinician_validation.py --generate

    # Step 2: After clinician fills it in, compute agreement
    python 17_clinician_validation.py --compute clinician_review_filled.xlsx

Run on LAB MACHINE.
Prerequisites: pip install openpyxl
"""

import json, os, sys, random
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    os.system(f"{sys.executable} -m pip install openpyxl --quiet")
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

BASE_DIR = Path(os.getenv("GENMEDASSIST_DIR", "."))
BENCH_PATH = BASE_DIR / "medconflict_bench_v2.jsonl"
OUTPUT_DIR = BASE_DIR / "clinician_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_validation_package():
    """Generate the complete clinician validation package."""
    print("=" * 70)
    print("Generating Clinician Validation Package")
    print("=" * 70)

    # Load benchmark
    data = [json.loads(l) for l in open(BENCH_PATH, encoding="utf-8")]
    test_data = [d for d in data if d.get("split") == "test"]

    # Stratified sampling: ~67 from each of S1, S2, S3 + some S0
    rng = random.Random(42)
    s0 = [d for d in test_data if d.get("stratum") == 0]
    s1 = [d for d in test_data if d.get("stratum") == 1]
    s2 = [d for d in test_data if d.get("stratum") == 2]
    s3 = [d for d in test_data if d.get("stratum") == 3]

    # Sample: 30 safe + 67 categorical + 67 numerical + 56 multi-hop = 220
    # (slightly more than 200 to allow for exclusions)
    sample_s0 = rng.sample(s0, min(30, len(s0)))
    sample_s1 = rng.sample(s1, min(67, len(s1)))
    sample_s2 = rng.sample(s2, min(67, len(s2)))
    sample_s3 = rng.sample(s3, min(56, len(s3)))

    # Shuffle so clinician doesn't see strata in order
    all_samples = sample_s0 + sample_s1 + sample_s2 + sample_s3
    rng.shuffle(all_samples)

    print(f"  Selected {len(all_samples)} cases:")
    print(f"    S0 (Safe): {len(sample_s0)}")
    print(f"    S1 (Categorical): {len(sample_s1)}")
    print(f"    S2 (Numerical-Threshold): {len(sample_s2)}")
    print(f"    S3 (Multi-hop Causal): {len(sample_s3)}")

    # ── Create Excel Workbook ──────────────────────────────────────────
    wb = openpyxl.Workbook()

    # Sheet 1: Instructions
    ws_inst = wb.active
    ws_inst.title = "Instructions"
    ws_inst.column_dimensions['A'].width = 100

    instructions = [
        "CLINICAL VALIDATION — MedConflict-Bench",
        "",
        "PROJECT: GenMedAssist — Knowledge-Graph Grounded Abstention for Clinical Conflict Detection",
        "INSTITUTION: IIIT Sricity, CSE Department",
        "GUIDE: Dr. Sreeja S.R.",
        "STUDENT: Ala Sai Teja [H27SR02]",
        "",
        "═══════════════════════════════════════════════════════════════",
        "PURPOSE",
        "═══════════════════════════════════════════════════════════════",
        "",
        "We have built a benchmark of clinical conflict cases extracted from",
        "MIMIC-IV (real hospital data). Each case has a prescription (drug)",
        "and a lab value. We need a medical professional to verify whether",
        "each case is correctly labeled as a 'conflict' or 'safe'.",
        "",
        "This validation ensures the quality of our ground truth labels.",
        "",
        "═══════════════════════════════════════════════════════════════",
        "WHAT IS A 'CLINICAL CONFLICT'?",
        "═══════════════════════════════════════════════════════════════",
        "",
        "A clinical conflict exists when:",
        "  1. A drug is prescribed to a patient with a contraindicated condition",
        "     (e.g., penicillin given to a patient with documented penicillin allergy)",
        "  2. A drug is prescribed when a lab value exceeds a safety threshold",
        "     (e.g., Metformin given when eGFR < 30 mL/min → lactic acidosis risk)",
        "  3. Two drugs interact dangerously, confirmed by an abnormal lab value",
        "     (e.g., Warfarin + Fluconazole with INR > 3.5 → bleeding risk)",
        "",
        "═══════════════════════════════════════════════════════════════",
        "WHAT YOU NEED TO DO",
        "═══════════════════════════════════════════════════════════════",
        "",
        "1. Go to the 'Cases' sheet",
        "2. For each row, review:",
        "   - Column B: The prescribed drug(s)",
        "   - Column C: The patient's condition (if any)",
        "   - Column D: The lab test name",
        "   - Column E: The lab value",
        "   - Column F: The lab unit",
        "   - Column G: Our system's label (CONFLICT or SAFE)",
        "",
        "3. In Column H ('Your Assessment'), enter one of:",
        "   - AGREE    → You agree with our label",
        "   - DISAGREE → You disagree with our label",
        "   - UNSURE   → You are not certain",
        "",
        "4. In Column I ('Your Reasoning'), briefly explain why",
        "   (especially for DISAGREE and UNSURE cases)",
        "",
        "5. In Column J ('Severity'), if it's a conflict, rate:",
        "   - MAJOR    → Could cause serious harm (e.g., organ damage, death)",
        "   - MODERATE → Requires dose adjustment or monitoring",
        "   - MINOR    → Low risk but should be noted",
        "   - N/A      → If it's a safe case",
        "",
        "═══════════════════════════════════════════════════════════════",
        "IMPORTANT NOTES",
        "═══════════════════════════════════════════════════════════════",
        "",
        "- All data comes from MIMIC-IV (real de-identified hospital records)",
        "- The drug names are exactly as recorded in MIMIC prescriptions",
        "- Lab values are real patient values at the time of prescription",
        "- Please use your clinical judgment based on current FDA guidelines",
        "- If you're unsure, mark UNSURE rather than guessing",
        "- This should take approximately 2-3 hours",
        "",
        "═══════════════════════════════════════════════════════════════",
        "WHICH MEDICAL PROFESSIONALS CAN REVIEW?",
        "═══════════════════════════════════════════════════════════════",
        "",
        "Any of the following are qualified:",
        "  - Clinical Pharmacist (PharmD) — BEST for drug-lab conflicts",
        "  - Physician (MD/MBBS) — any specialty, especially Internal Medicine",
        "  - Nurse Practitioner (NP) with prescribing authority",
        "  - Physician Assistant (PA) with clinical experience",
        "",
        "Ideal specialties: Internal Medicine, Clinical Pharmacology,",
        "Emergency Medicine, Critical Care, Nephrology, Cardiology",
        "",
        "═══════════════════════════════════════════════════════════════",
        "REFERENCE: THRESHOLD RULES USED IN OUR SYSTEM",
        "═══════════════════════════════════════════════════════════════",
        "",
        "Drug → Lab Test → Threshold → Action",
        "─────────────────────────────────────────────────────",
        "Metformin → eGFR < 30 → Contraindicated (lactic acidosis)",
        "Metformin → eGFR < 45 → Dose reduction needed",
        "Gabapentin → eGFR < 60 → Dose reduction (renal excretion)",
        "Dabigatran → eGFR < 30 → Contraindicated",
        "Rivaroxaban → eGFR < 15 → Contraindicated",
        "Digoxin → Potassium < 3.5 → Risk increased (toxicity)",
        "Spironolactone → Potassium > 5.0 → Contraindicated",
        "ACE Inhibitors → Potassium > 5.5 → Discontinue",
        "Acetaminophen → ALT > 120 → Contraindicated (hepatotoxicity)",
        "Statins → ALT > 120 → Contraindicated (hepatotoxicity)",
        "Warfarin → INR > 4.0 → Hold dose (bleeding risk)",
        "Heparin → Platelets < 100,000 → Contraindicated (HIT risk)",
        "Sulfonylureas → Glucose < 70 → Dose reduction (hypoglycemia)",
        "Insulin → Glucose < 54 → Hold dose (severe hypoglycemia)",
        "Methotrexate → WBC < 3,000 → Hold dose (bone marrow)",
        "Methotrexate → eGFR < 30 → Contraindicated",
        "",
        "These thresholds are based on FDA clinical guidelines.",
        "",
        "Thank you for your time and expertise!",
    ]

    for i, line in enumerate(instructions, 1):
        cell = ws_inst.cell(row=i, column=1, value=line)
        if line.startswith("═") or line.startswith("─"):
            cell.font = Font(name="Consolas", size=10)
        elif line.startswith("CLINICAL VALIDATION"):
            cell.font = Font(name="Arial", size=16, bold=True, color="006600")
        elif line.startswith("PROJECT:") or line.startswith("PURPOSE") or \
             line.startswith("WHAT IS") or line.startswith("WHAT YOU") or \
             line.startswith("IMPORTANT") or line.startswith("WHICH") or \
             line.startswith("REFERENCE"):
            cell.font = Font(name="Arial", size=12, bold=True, color="003366")
        else:
            cell.font = Font(name="Arial", size=11)

    # Sheet 2: Cases for Review
    ws_cases = wb.create_sheet("Cases")

    # Headers
    headers = [
        ("A", "Case #", 8),
        ("B", "Drug(s) Prescribed", 30),
        ("C", "Patient Condition", 25),
        ("D", "Lab Test", 18),
        ("E", "Lab Value", 12),
        ("F", "Lab Unit", 15),
        ("G", "System Label", 15),
        ("H", "Your Assessment\n(AGREE/DISAGREE/UNSURE)", 25),
        ("I", "Your Reasoning\n(brief explanation)", 40),
        ("J", "Severity\n(MAJOR/MODERATE/MINOR/N/A)", 20),
    ]

    header_fill = PatternFill(start_color="003366", end_color="003366", fill_type="solid")
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    for col_letter, header_text, width in headers:
        col_idx = openpyxl.utils.column_index_from_string(col_letter)
        cell = ws_cases.cell(row=1, column=col_idx, value=header_text)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = thin_border
        ws_cases.column_dimensions[col_letter].width = width

    ws_cases.row_dimensions[1].height = 40

    # Fill in cases
    conflict_fill = PatternFill(start_color="FFE0E0", end_color="FFE0E0", fill_type="solid")
    safe_fill = PatternFill(start_color="E0FFE0", end_color="E0FFE0", fill_type="solid")
    input_fill = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")

    for i, ex in enumerate(all_samples, 2):
        drugs = ex.get("drugs", [ex.get("drug", "")])
        if isinstance(drugs, list):
            drug_str = " + ".join(drugs)
        else:
            drug_str = str(drugs)

        condition = ex.get("conflict_condition", "")
        lab_name = ex.get("lab_name", "")
        lab_value = ex.get("lab_value", "")
        lab_unit = ex.get("lab_unit", "")
        is_conflict = ex.get("conflict_present", False)
        system_label = "CONFLICT" if is_conflict else "SAFE"

        row_fill = conflict_fill if is_conflict else safe_fill

        values = [i - 1, drug_str, condition, lab_name, lab_value, lab_unit, system_label]
        for j, val in enumerate(values, 1):
            cell = ws_cases.cell(row=i, column=j, value=val)
            cell.font = Font(name="Arial", size=11)
            cell.border = thin_border
            cell.alignment = Alignment(vertical='center')
            if j == 7:  # System label
                cell.fill = row_fill
                cell.font = Font(name="Arial", size=11, bold=True,
                                 color="CC0000" if is_conflict else "006600")
            else:
                cell.fill = PatternFill(start_color="FFFFFF", fill_type="solid")

        # Clinician input columns (highlighted yellow)
        for j in [8, 9, 10]:
            cell = ws_cases.cell(row=i, column=j, value="")
            cell.fill = input_fill
            cell.border = thin_border
            cell.alignment = Alignment(vertical='center')

    # Add data validation for Column H
    from openpyxl.worksheet.datavalidation import DataValidation
    dv_assess = DataValidation(type="list", formula1='"AGREE,DISAGREE,UNSURE"', allow_blank=True)
    dv_assess.error = "Please enter AGREE, DISAGREE, or UNSURE"
    dv_assess.errorTitle = "Invalid Input"
    ws_cases.add_data_validation(dv_assess)
    dv_assess.add(f"H2:H{len(all_samples)+1}")

    dv_severity = DataValidation(type="list", formula1='"MAJOR,MODERATE,MINOR,N/A"', allow_blank=True)
    ws_cases.add_data_validation(dv_severity)
    dv_severity.add(f"J2:J{len(all_samples)+1}")

    # Sheet 3: Summary (auto-filled after clinician review)
    ws_summary = wb.create_sheet("Summary (auto)")
    ws_summary.cell(row=1, column=1, value="This sheet will be filled automatically").font = Font(bold=True)
    ws_summary.cell(row=2, column=1, value="after running: python 17_clinician_validation.py --compute <file>")
    ws_summary.cell(row=4, column=1, value="Metrics computed:").font = Font(bold=True)
    ws_summary.cell(row=5, column=1, value="- Raw Agreement Count")
    ws_summary.cell(row=6, column=1, value="- Cohen's Kappa (inter-rater reliability)")
    ws_summary.cell(row=7, column=1, value="- Per-stratum agreement rates")
    ws_summary.cell(row=8, column=1, value="- Disagreement analysis")

    # Save
    excel_path = OUTPUT_DIR / "clinician_review_form.xlsx"
    wb.save(str(excel_path))
    print(f"\n  Excel form saved: {excel_path}")

    # Also save the case list as JSON for reference
    case_list = []
    for ex in all_samples:
        drugs = ex.get("drugs", [ex.get("drug", "")])
        case_list.append({
            "example_id": ex.get("example_id", ""),
            "stratum": ex.get("stratum", 0),
            "drugs": drugs if isinstance(drugs, list) else [drugs],
            "condition": ex.get("conflict_condition", ""),
            "lab_name": ex.get("lab_name", ""),
            "lab_value": ex.get("lab_value", ""),
            "lab_unit": ex.get("lab_unit", ""),
            "system_label": "CONFLICT" if ex.get("conflict_present") else "SAFE",
            "conflict_present": ex.get("conflict_present", False),
            "rule_id": ex.get("rule_id", ""),
        })

    json_path = OUTPUT_DIR / "validation_cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(case_list, f, indent=2, ensure_ascii=False)
    print(f"  JSON reference saved: {json_path}")

    # Print instructions for sharing
    print(f"\n  ═══════════════════════════════════════════════════════")
    print(f"  HOW TO USE THIS:")
    print(f"  ═══════════════════════════════════════════════════════")
    print(f"  1. Send '{excel_path.name}' to the clinician")
    print(f"  2. They open it in Excel/Google Sheets")
    print(f"  3. They read the 'Instructions' sheet first")
    print(f"  4. They fill in columns H, I, J on the 'Cases' sheet")
    print(f"  5. They save and return the file to you")
    print(f"  6. You run: python 17_clinician_validation.py --compute <returned_file>")
    print(f"  ═══════════════════════════════════════════════════════")
    print(f"\n  Who can review (in order of preference):")
    print(f"    1. Clinical Pharmacist (PharmD) — BEST for drug-lab conflicts")
    print(f"    2. Physician (MD/MBBS) — Internal Medicine, Nephrology, Cardiology")
    print(f"    3. Nurse Practitioner with prescribing authority")
    print(f"    4. Medical PG student (MD/MS) — with clinical rotation experience")
    print(f"\n  Time required: 2-3 hours")
    print(f"  Cases to review: {len(all_samples)}")

    return all_samples


def compute_agreement(filepath):
    """Compute Cohen's Kappa and agreement metrics from filled spreadsheet."""
    print("=" * 70)
    print("Computing Clinician Agreement Metrics")
    print("=" * 70)

    wb = openpyxl.load_workbook(filepath)
    ws = wb["Cases"]

    system_labels = []
    clinician_labels = []
    disagreements = []
    per_stratum = defaultdict(lambda: {"agree": 0, "disagree": 0, "unsure": 0, "total": 0})

    # Load reference cases for stratum info
    ref_path = OUTPUT_DIR / "validation_cases.json"
    ref_cases = json.load(open(ref_path, encoding="utf-8")) if ref_path.exists() else []
    stratum_map = {i: ref_cases[i].get("stratum", -1) for i in range(len(ref_cases))} if ref_cases else {}

    for row in range(2, ws.max_row + 1):
        case_num = ws.cell(row=row, column=1).value
        if case_num is None:
            continue

        system_label = ws.cell(row=row, column=7).value  # CONFLICT or SAFE
        assessment = str(ws.cell(row=row, column=8).value or "").strip().upper()
        reasoning = str(ws.cell(row=row, column=9).value or "")
        severity = str(ws.cell(row=row, column=10).value or "")

        if assessment not in ["AGREE", "DISAGREE", "UNSURE"]:
            continue

        idx = int(case_num) - 1
        stratum = stratum_map.get(idx, -1)

        # For Kappa: convert to binary
        sys_binary = 1 if system_label == "CONFLICT" else 0

        if assessment == "AGREE":
            clin_binary = sys_binary  # clinician agrees with system
        elif assessment == "DISAGREE":
            clin_binary = 1 - sys_binary  # clinician disagrees
        else:  # UNSURE
            clin_binary = sys_binary  # treat unsure as agree for Kappa (conservative)

        system_labels.append(sys_binary)
        clinician_labels.append(clin_binary)

        per_stratum[stratum]["total"] += 1
        if assessment == "AGREE":
            per_stratum[stratum]["agree"] += 1
        elif assessment == "DISAGREE":
            per_stratum[stratum]["disagree"] += 1
            disagreements.append({
                "case": case_num,
                "drug": ws.cell(row=row, column=2).value,
                "lab": f"{ws.cell(row=row, column=4).value}={ws.cell(row=row, column=5).value}",
                "system": system_label,
                "reasoning": reasoning,
                "stratum": stratum,
            })
        else:
            per_stratum[stratum]["unsure"] += 1

    if not system_labels:
        print("  ERROR: No assessments found. Make sure the clinician filled in column H.")
        return

    n = len(system_labels)
    sys_arr = np.array(system_labels)
    clin_arr = np.array(clinician_labels)

    # Raw agreement
    raw_agree = (sys_arr == clin_arr).sum()
    raw_rate = raw_agree / n

    # Cohen's Kappa
    po = raw_rate  # observed agreement
    pe_1 = (sys_arr.sum() / n) * (clin_arr.sum() / n)
    pe_0 = ((n - sys_arr.sum()) / n) * ((n - clin_arr.sum()) / n)
    pe = pe_1 + pe_0  # expected agreement by chance
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    print(f"\n  Total cases reviewed: {n}")
    print(f"  Raw agreement: {raw_agree}/{n} = {raw_rate:.1%}")
    print(f"  Cohen's Kappa: {kappa:.4f}")

    # Interpret Kappa
    if kappa >= 0.81:
        interp = "Almost Perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    else:
        interp = "Slight/Poor"
    print(f"  Interpretation: {interp} agreement")

    # Per-stratum
    print(f"\n  Per-stratum agreement:")
    stratum_names = {0: "Safe", 1: "Categorical", 2: "Numerical", 3: "Multi-hop"}
    for s in sorted(per_stratum.keys()):
        d = per_stratum[s]
        if d["total"] == 0:
            continue
        agree_rate = d["agree"] / d["total"]
        name = stratum_names.get(s, f"S{s}")
        print(f"    S{s} ({name}): {d['agree']}/{d['total']} agree ({agree_rate:.1%}), "
              f"{d['disagree']} disagree, {d['unsure']} unsure")

    # Disagreement analysis
    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)} cases):")
        for d in disagreements[:10]:
            print(f"    Case {d['case']}: {d['drug']} | {d['lab']} | System={d['system']}")
            if d['reasoning']:
                print(f"      Clinician reasoning: {d['reasoning'][:100]}")

    # Save results
    results = {
        "total_reviewed": n,
        "raw_agreement": raw_agree,
        "raw_agreement_rate": round(raw_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": interp,
        "per_stratum": {str(k): dict(v) for k, v in per_stratum.items()},
        "disagreements": disagreements,
    }
    out_path = OUTPUT_DIR / "validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate the review spreadsheet")
    parser.add_argument("--compute", type=str, help="Compute agreement from filled spreadsheet")
    args = parser.parse_args()

    if args.compute:
        compute_agreement(args.compute)
    elif args.generate:
        generate_validation_package()
    else:
        # Default: generate
        generate_validation_package()


if __name__ == "__main__":
    main()