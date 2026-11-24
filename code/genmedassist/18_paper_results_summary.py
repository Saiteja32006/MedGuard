#!/usr/bin/env python3
import os
r"""
18_paper_results_summary.py — Paper-Ready Results for Co-Author
================================================================
Reads ALL result files and generates a clean, accurate summary.
No computation — just reads existing JSONs and formats them.

Run on LAB MACHINE:
    python 18_paper_results_summary.py

Outputs:
    - Console: all tables formatted for copy-paste
    - paper_results_summary.json: machine-readable
    - paper_results_summary.txt: human-readable for co-author
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(os.getenv("GENMEDASSIST_DIR", "."))
EVAL_DIR = BASE_DIR / "eval_results"
SYSTEM_DIR = BASE_DIR / "system_output"
ADDITIONAL_DIR = BASE_DIR / "additional_eval"
BENCH_PATH = BASE_DIR / "medconflict_bench_v2.jsonl"

OUTPUT_PATH = BASE_DIR / "paper_results_summary.txt"
JSON_PATH = BASE_DIR / "paper_results_summary.json"


def load_baseline_results():
    """Load all 7 baseline model results."""
    models = {}
    for f in EVAL_DIR.glob("*_results.jsonl"):
        name = f.stem.replace("_results", "")
        preds = [json.loads(l) for l in open(f, encoding="utf-8")]
        if not preds or "stratum" not in preds[0]:
            continue
        models[name] = preds
    return models


def compute_model_metrics(preds):
    """Compute per-stratum metrics from predictions."""
    from sklearn.metrics import f1_score
    y_true = np.array([1 if r["gt_conflict"] else 0 for r in preds])
    y_pred = np.array([1 if r["pred_conflict"] else 0 for r in preds])

    result = {
        "n": len(preds),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }

    for s in [0, 1, 2, 3]:
        mask = np.array([r["stratum"] == s for r in preds])
        if mask.sum() == 0:
            continue
        st = y_true[mask]
        sp = y_pred[mask]
        if s == 0:
            result[f"S{s}_specificity"] = round(1 - sp.mean(), 4)
        else:
            result[f"S{s}_detection"] = round(sp.mean(), 4)

    # Cliff
    s1_mask = np.array([r["stratum"] == 1 for r in preds])
    s2_mask = np.array([r["stratum"] == 2 for r in preds])
    if s1_mask.sum() > 0 and s2_mask.sum() > 0:
        s1_det = y_pred[s1_mask].mean()
        s2_det = y_pred[s2_mask].mean()
        result["cliff"] = round(s1_det - s2_det, 4)

    return result


def compute_significance(preds):
    """Compute z-test and bootstrap CI for the cliff."""
    from scipy import stats

    s1 = [p for p in preds if p.get("stratum") == 1]
    s2 = [p for p in preds if p.get("stratum") == 2]
    if not s1 or not s2:
        return {}

    s1_det = sum(1 for p in s1 if p.get("pred_conflict")) / len(s1)
    s2_det = sum(1 for p in s2 if p.get("pred_conflict")) / len(s2)
    cliff = s1_det - s2_det

    s1_k = sum(1 for p in s1 if p.get("pred_conflict"))
    s2_k = sum(1 for p in s2 if p.get("pred_conflict"))
    p_pool = (s1_k + s2_k) / (len(s1) + len(s2))
    se = np.sqrt(p_pool * (1 - p_pool) * (1/len(s1) + 1/len(s2))) if p_pool > 0 and p_pool < 1 else 1
    z = cliff / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    rng = np.random.RandomState(42)
    s1_arr = np.array([1 if p.get("pred_conflict") else 0 for p in s1])
    s2_arr = np.array([1 if p.get("pred_conflict") else 0 for p in s2])
    boots = [rng.choice(s1_arr, len(s1_arr), True).mean() - rng.choice(s2_arr, len(s2_arr), True).mean()
             for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    return {
        "cliff": round(cliff, 4),
        "z": round(z, 3),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05,
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    lines = []
    all_data = {}

    def pr(text=""):
        print(text)
        lines.append(text)

    pr("=" * 80)
    pr("GenMedAssist — COMPLETE PAPER RESULTS SUMMARY")
    pr("=" * 80)
    pr(f"Generated from: {BASE_DIR}")
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 1: MedConflict-Bench Statistics
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 1: MedConflict-Bench v2 Statistics")
    pr("=" * 80)

    bench = [json.loads(l) for l in open(BENCH_PATH, encoding="utf-8")]
    test = [d for d in bench if d.get("split") == "test"]
    cal = [d for d in bench if d.get("split") == "calibration"]

    strata_counts = defaultdict(int)
    for d in bench:
        strata_counts[d.get("stratum", -1)] += 1

    conflict_count = sum(1 for d in bench if d.get("conflict_present"))
    safe_count = len(bench) - conflict_count

    # Drug and lab diversity
    drugs = set()
    labs = set()
    for ex in bench:
        d = ex.get("drug", ex.get("drugs", ""))
        if isinstance(d, list):
            for x in d:
                drugs.add(x.lower().strip())
        elif d:
            drugs.add(str(d).lower().strip())
        lab = ex.get("lab_name", "")
        if lab:
            labs.add(lab)

    bench_stats = {
        "total": len(bench), "calibration": len(cal), "test": len(test),
        "S0_safe": strata_counts[0], "S1_categorical": strata_counts[1],
        "S2_numerical": strata_counts[2], "S3_multihop": strata_counts[3],
        "conflict_count": conflict_count, "safe_count": safe_count,
        "positive_rate": round(conflict_count / len(bench) * 100, 1),
        "unique_drugs": len(drugs), "unique_lab_tests": len(labs),
        "lab_tests": sorted(labs),
    }
    all_data["benchmark"] = bench_stats

    pr(f"  Total examples:        {len(bench)}")
    pr(f"  Calibration split:     {len(cal)} (20%)")
    pr(f"  Test split:            {len(test)} (80%)")
    pr(f"  S0 (Safe):             {strata_counts[0]}")
    pr(f"  S1 (Categorical):      {strata_counts[1]}")
    pr(f"  S2 (Numerical):        {strata_counts[2]}")
    pr(f"  S3 (Multi-hop):        {strata_counts[3]}")
    pr(f"  Conflicts:             {conflict_count} ({bench_stats['positive_rate']}%)")
    pr(f"  Safe:                  {safe_count} ({100 - bench_stats['positive_rate']}%)")
    pr(f"  Unique drugs:          {len(drugs)}")
    pr(f"  Unique lab tests:      {len(labs)}: {', '.join(sorted(labs))}")
    pr(f"  Rules: 12 categorical + 16 numerical + 8 multi-hop = 36 total")
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 2: Performance Cliff (RQ1)
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 2: Per-Stratum Detection Rates — The Performance Cliff (RQ1)")
    pr("=" * 80)

    models = load_baseline_results()
    pr(f"\n  {'Model':<20} {'S0-Spec':>8} {'S1-Cat':>8} {'S2-Num':>8} {'S3-MH':>8} {'Cliff':>8} {'F1':>8}")
    pr(f"  {'-'*68}")

    model_metrics = {}
    for name in sorted(models.keys()):
        m = compute_model_metrics(models[name])
        model_metrics[name] = m
        s0 = f"{m.get('S0_specificity', 0):.3f}"
        s1 = f"{m.get('S1_detection', 0):.3f}"
        s2 = f"{m.get('S2_detection', 0):.3f}"
        s3 = f"{m.get('S3_detection', 0):.3f}"
        cliff = f"{m.get('cliff', 0):+.3f}"
        f1 = f"{m['f1']:.3f}"
        pr(f"  {name:<20} {s0:>8} {s1:>8} {s2:>8} {s3:>8} {cliff:>8} {f1:>8}")

    # Add GenMedAssist (Ours)
    abl_path = SYSTEM_DIR / "ablation" / "ablation_results.json"
    if abl_path.exists():
        abl = json.load(open(abl_path, encoding="utf-8"))
        full = abl.get("Full System", {})
        ps = full.get("per_stratum", {})
        s0_spec = ps.get("S0", {}).get("specificity", 1.0)
        s1_rec = ps.get("S1", {}).get("recall", 0)
        s2_rec = ps.get("S2", {}).get("recall", 0)
        s3_rec = ps.get("S3", {}).get("recall", 0)
        ours_f1 = full.get("macro_f1", 0)
        pr(f"  {'GenMedAssist (Ours)':<20} {s0_spec:>8.3f} {s1_rec:>8.3f} {s2_rec:>8.3f} {s3_rec:>8.3f} {'—':>8} {ours_f1:>8.3f}")

    all_data["baseline_results"] = model_metrics
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 3: Statistical Significance
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 3: Statistical Significance of the Performance Cliff")
    pr("=" * 80)

    pr(f"\n  {'Model':<20} {'S1':>6} {'S2':>6} {'Cliff':>8} {'z':>8} {'p-value':>12} {'Sig?':>8} {'95% CI':>16}")
    pr(f"  {'-'*84}")

    sig_data = {}
    for name in sorted(models.keys()):
        sig = compute_significance(models[name])
        if not sig:
            continue
        sig_data[name] = sig
        m = model_metrics[name]
        s1 = f"{m.get('S1_detection', 0):.3f}"
        s2 = f"{m.get('S2_detection', 0):.3f}"
        cliff = f"{sig['cliff']:+.3f}"
        z = f"{sig['z']:.3f}"
        p = f"{sig['p_value']:.6f}"
        s = "YES ***" if sig['p_value'] < 0.001 else "YES **" if sig['p_value'] < 0.01 else "YES *" if sig['p_value'] < 0.05 else "NO"
        ci = f"[{sig['ci_95'][0]:+.3f}, {sig['ci_95'][1]:+.3f}]"
        pr(f"  {name:<20} {s1:>6} {s2:>6} {cliff:>8} {z:>8} {p:>12} {s:>8} {ci:>16}")

    all_data["significance"] = sig_data
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 4: Ablation Study (RQ4)
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 4: Ablation Study (RQ4) — Slide 15")
    pr("=" * 80)

    if abl_path.exists():
        abl = json.load(open(abl_path, encoding="utf-8"))
        full_f1 = abl.get("Full System", {}).get("macro_f1", 0)

        pr(f"\n  {'Variant':<20} {'Macro-F1':>10} {'ΔF1':>10} {'S1-Cat':>10} {'S2-Num':>10} {'S3-MH':>10} {'S0-Spec':>10}")
        pr(f"  {'-'*80}")

        for variant in ["Full System", "No GraphRAG", "No NumDist", "No VectorRAG", "No Conformal", "LLM Only"]:
            v = abl.get(variant, {})
            ps = v.get("per_stratum", {})
            mf1 = v.get("macro_f1", 0)
            delta = f"{mf1 - full_f1:+.4f}" if variant != "Full System" else "—"
            s1 = ps.get("S1", {}).get("recall", 0)
            s2 = ps.get("S2", {}).get("recall", 0)
            s3 = ps.get("S3", {}).get("recall", 0)
            s0 = ps.get("S0", {}).get("specificity", 1.0)
            pr(f"  {variant:<20} {mf1:>10.4f} {delta:>10} {s1:>10.4f} {s2:>10.4f} {s3:>10.4f} {s0:>10.4f}")

        all_data["ablation"] = abl
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 5: Coverage-Accuracy Curves (RQ3)
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 5: Coverage-Accuracy Curves (RQ3)")
    pr("=" * 80)

    cov_path = SYSTEM_DIR / "coverage_accuracy_results.json"
    if cov_path.exists():
        cov = json.load(open(cov_path, encoding="utf-8"))
        pr(f"\n  {'Method':<20} {'α':>6} {'τ':>8} {'Coverage':>10} {'SelAcc':>10} {'Recall':>8} {'Missed':>8}")
        pr(f"  {'-'*70}")
        for entry in cov:
            method = entry.get("method", "")
            alpha = f"{entry.get('alpha', '')}" if entry.get("alpha") else "—"
            tau = f"{entry.get('tau', 0):.4f}" if entry.get("tau") else "—"
            cov_val = f"{entry.get('coverage', 0):.4f}"
            sel = f"{entry.get('sel_accuracy', entry.get('selective_accuracy', 0)):.4f}"
            rec = f"{entry.get('recall', 0):.4f}"
            missed = f"{entry.get('missed', 0)}"
            pr(f"  {method:<20} {alpha:>6} {tau:>8} {cov_val:>10} {sel:>10} {rec:>8} {missed:>8}")

        all_data["coverage_accuracy"] = cov
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 6: C-score Configuration
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 6: C-score Configuration")
    pr("=" * 80)
    pr(f"  Formulation: C(E,G) = w1 * PathViol(E,G) + w2 * NumDist(E)")
    pr(f"  w1 = 0.6075 (learned via logistic regression)")
    pr(f"  w2 = 0.3925 (learned via logistic regression)")
    pr(f"  tau = 0.0028 (split conformal prediction, alpha=0.05)")
    pr(f"  PathViol: fraction of KG paths with violation edges")
    pr(f"  NumDist: max normalized threshold violation magnitude")
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 7: Med-HALT Results
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 7: Med-HALT — Hallucination/Abstention Testing")
    pr("=" * 80)

    medhalt_path = ADDITIONAL_DIR / "medhalt_results.json"
    if medhalt_path.exists():
        mh = json.load(open(medhalt_path, encoding="utf-8"))
        cd = mh.get("clinical_domain", {})
        cats = cd.get("per_category", {})

        pr(f"\n  Med-HALT HF Dataset Info:")
        hf = mh.get("hf_dataset_info", mh.get("hf_dataset", {}))
        if hf:
            for k, v in hf.items():
                pr(f"    {k}: {v} examples")

        pr(f"\n  Clinical-Domain Abstention Tests:")
        pr(f"  {'Category':<12} {'Description':<45} {'Correct':>8} {'Total':>6} {'Acc':>8}")
        pr(f"  {'-'*80}")

        descs = {
            "FCT": "Should ABSTAIN (real conflicts, wrong claims)",
            "FQT": "Should SAFE (fake/nonexistent interactions)",
            "NOTA": "Should SAFE (safe values, misleading context)",
        }
        for cat in ["FCT", "FQT", "NOTA"]:
            if cat in cats:
                c = cats[cat]
                pr(f"  {cat:<12} {descs.get(cat, ''):<45} {c['correct']:>8} {c['total']:>6} {c['accuracy']:>8.1%}")

        pr(f"  {'OVERALL':<12} {'All clinical abstention tests':<45} {cd.get('correct', 0):>8} {cd.get('total', 0):>6} {cd.get('accuracy', 0):>8.1%}")

        all_data["medhalt"] = mh
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 8: MedQA Results
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 8: MedQA/USMLE Evaluation")
    pr("=" * 80)

    medqa_path = ADDITIONAL_DIR / "medqa_results.json"
    if medqa_path.exists():
        mq = json.load(open(medqa_path, encoding="utf-8"))
        pr(f"  Dataset: {mq.get('dataset', 'MedQA-USMLE-4-options')}")
        pr(f"  Total test questions: {mq.get('total', 0)}")
        pr(f"  Pharmacology-relevant: {mq.get('pharma', mq.get('pharma_relevant', 0))}")
        pr(f"  Other medical: {mq.get('other', 0)}")
        pr(f"  KG-evaluated (drug found in KG): {mq.get('kg_evaluated', 0)}")
        pr(f"  KG contributed to answer: {mq.get('kg_contributed', 0)}/{mq.get('kg_evaluated', 0)} = {mq.get('kg_accuracy', 0):.1%}")
        pr(f"  Note: GenMedAssist is a conflict detection system, not a general QA system.")
        pr(f"        KG accuracy measures overlap with pharmacology USMLE knowledge.")

        all_data["medqa"] = mq
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 9: Adversarial Robustness
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 9: MIMIC-IV Adversarial Robustness")
    pr("=" * 80)

    adv_path = ADDITIONAL_DIR / "adversarial_results.json"
    if adv_path.exists():
        adv = json.load(open(adv_path, encoding="utf-8"))
        pr(f"  Total adversarial cases: {adv.get('total', 0)}")
        pr(f"  Correctly detected despite adversarial prefix: {adv.get('detected', 0)}/{adv.get('total', 0)}")
        pr(f"  Robustness rate: {adv.get('robustness', 0):.1%}")
        pr(f"  Fooled: {adv.get('total', 0) - adv.get('detected', 0)}/{adv.get('total', 0)}")

        ps = adv.get("per_stratum", {})
        pr(f"\n  {'Stratum':<20} {'Detected':>10} {'Total':>8} {'Robustness':>12}")
        pr(f"  {'-'*50}")
        for s in sorted(ps.keys(), key=lambda x: int(x)):
            d = ps[s]
            name = {"1": "Categorical", "2": "Numerical", "3": "Multi-hop"}.get(s, f"S{s}")
            det = d.get("d", d.get("det", d.get("detected", 0)))
            tot = d.get("n", d.get("total", 0))
            rate = det / max(tot, 1)
            pr(f"  S{s} ({name}){'':<10} {det:>10} {tot:>8} {rate:>12.1%}")

        all_data["adversarial"] = adv
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 10: Computation Stats
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 10: Computation Statistics")
    pr("=" * 80)

    traj_path = SYSTEM_DIR / "trajectories" / "genmedassist_trajectories_v1.jsonl"
    if traj_path.exists():
        trajs = [json.loads(l) for l in open(traj_path, encoding="utf-8")]
        times = [t.get("processing_ms", 0) for t in trajs]
        pr(f"  GenMedAssist inference time per case:")
        pr(f"    Mean:   {np.mean(times):.1f} ms")
        pr(f"    Median: {np.median(times):.1f} ms")
        pr(f"    P95:    {np.percentile(times, 95):.1f} ms")
        pr(f"    Total for {len(trajs)} cases: {sum(times)/1000:.1f} sec")

        all_data["computation"] = {
            "mean_ms": round(np.mean(times), 1),
            "median_ms": round(np.median(times), 1),
            "p95_ms": round(np.percentile(times, 95), 1),
            "total_cases": len(trajs),
        }

    pr(f"  Hardware: Intel i7, 128GB RAM, RTX A5000 24GB")
    pr(f"  No GPU used for C-score (CPU-only, Neo4j + ChromaDB)")
    pr(f"  No training required — inference-time method")
    pr()

    # ══════════════════════════════════════════════════════════════════
    # TABLE 11: Model Versions
    # ══════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("TABLE 11: Model Versions (Reproducibility)")
    pr("=" * 80)

    doc_path = ADDITIONAL_DIR / "model_documentation.json"
    if doc_path.exists():
        docs = json.load(open(doc_path, encoding="utf-8"))
        mdls = docs.get("models", docs.get("evaluation_models", {}))
        pr(f"\n  {'Model':<20} {'Checkpoint ID':<45} {'Quantization':<15}")
        pr(f"  {'-'*80}")
        for name, info in mdls.items():
            mid = info.get("id", info.get("model_id", ""))
            quant = info.get("quant", info.get("quantization", "API"))
            pr(f"  {name:<20} {mid:<45} {quant:<15}")
    pr()

    # ══════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Text summary saved: {OUTPUT_PATH}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON data saved: {JSON_PATH}")

    pr("\n" + "=" * 80)
    pr("SEND THESE TWO FILES TO YOUR CO-AUTHOR:")
    pr(f"  1. {OUTPUT_PATH}")
    pr(f"  2. {JSON_PATH}")
    pr("=" * 80)


if __name__ == "__main__":
    main()
