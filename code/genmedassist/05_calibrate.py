#!/usr/bin/env python3
import os
"""
05_calibrate.py — Conformal Prediction Calibration for C-Score
===============================================================

Uses the calibration split of MedConflict-Bench to set the abstention
threshold τ with distribution-free coverage guarantees.

Key idea:
  Given desired risk level α, find τ such that:
    Pr[error | C_score < τ] ≤ α   (coverage guarantee)

  This is split conformal prediction (Vovk et al. 2005, adapted for
  selective prediction / abstention).

Method:
  1. Compute C-scores for all calibration examples
  2. Define nonconformity scores: s_i = -C_score_i if conflict, +C_score_i if safe
     (higher nonconformity = more likely to be a mistake)
  3. For each α, set τ as the ⌈(1-α)(n+1)⌉/n quantile of calibration scores
  4. At test time: abstain if C_score > τ

Outputs:
  - Calibrated τ values for α ∈ {0.01, 0.05, 0.10, 0.20}
  - Coverage-accuracy curves
  - Calibration diagnostics

Author: Ala Sai Teja, IIIT Sricity
Project: GenMedAssist — NeurIPS 2026

Requirements:
  pip install numpy pandas matplotlib scikit-learn
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to MedConflict-Bench v2
BENCH_DIR = Path(os.getenv("BENCH_DIR", "./MedConflict-Bench"))
# BENCH_DIR = Path("./MedConflict-Bench")  # local alternative

# Risk levels to calibrate
ALPHA_VALUES = [0.01, 0.05, 0.10, 0.20]

# C-score weights (from 04_cscore.py or learned via logistic regression)
# These can be overridden after fitting on calibration data
W1_DEFAULT = 0.6  # PathViol weight
W2_DEFAULT = 0.4  # NumDist weight

OUTPUT_DIR = BENCH_DIR / "calibration"

SEED = 42
np.random.seed(SEED)


# ============================================================================
# C-SCORE COMPUTATION (standalone, no Neo4j needed)
# ============================================================================

def compute_cscore_from_example(example, w1=W1_DEFAULT, w2=W2_DEFAULT):
    """
    Compute C-score from a MedConflict-Bench example.
    
    For benchmark evaluation (no KG access needed):
    - PathViol = 1.0 if conflict_present and conflict involves a known
      contraindication path, 0.0 otherwise
    - NumDist = normalized distance from threshold (for numerical strata)
    
    This is a "ground truth" C-score based on the annotations,
    used to calibrate τ. At inference time, the real C-score
    comes from Neo4j KG lookups (04_cscore.py).
    """
    
    if not example.get("conflict_present", False):
        return 0.0
    
    path_viol = 0.0
    num_dist = 0.0
    
    stratum = example.get("stratum", 0)
    
    if stratum == 1:
        # Categorical: pure path violation, no numerical component
        path_viol = 1.0
        num_dist = 0.0
    
    elif stratum == 2:
        # Numerical threshold: compute normalized distance
        path_viol = 1.0  # The KG path exists (drug→condition→lab)
        
        lab_value = example.get("lab_value")
        threshold = example.get("threshold")
        operator = example.get("operator")
        
        if lab_value is not None and threshold is not None and threshold != 0:
            if operator == "<":
                # Bad when value < threshold, distance = (threshold - value) / threshold
                num_dist = max(0, (threshold - lab_value) / threshold)
            else:  # ">"
                # Bad when value > threshold, distance = (value - threshold) / threshold
                num_dist = max(0, (lab_value - threshold) / threshold)
        
        # Cap at 1.0
        num_dist = min(num_dist, 1.0)
    
    elif stratum == 3:
        # Multi-hop: both path violation AND numerical component
        path_viol = 1.0
        
        lab_value = example.get("lab_value")
        threshold = example.get("threshold")
        operator = example.get("operator")
        
        if lab_value is not None and threshold is not None and threshold != 0:
            if operator == "<":
                num_dist = max(0, (threshold - lab_value) / threshold)
            else:
                num_dist = max(0, (lab_value - threshold) / threshold)
        
        num_dist = min(num_dist, 1.0)
    
    cscore = w1 * path_viol + w2 * num_dist
    return round(cscore, 4)


# ============================================================================
# CONFORMAL PREDICTION
# ============================================================================

def compute_nonconformity_scores(examples, w1=W1_DEFAULT, w2=W2_DEFAULT):
    """
    Compute nonconformity scores for calibration.
    
    For selective prediction / abstention:
    - We want to abstain on high C-score examples
    - Nonconformity score = C_score for conflict examples (higher = riskier)
    - For safe examples, nonconformity = 1 - C_score (higher if model wrongly
      assigns low C-score to a safe case — but safe cases should have C ≈ 0)
    
    Actually, for coverage guarantee on errors:
      s_i = C_score_i  (for all examples)
      τ chosen so that among examples with C_score ≤ τ, error rate ≤ α
    
    We use the standard split conformal approach:
      τ_α = Quantile(1-α, {s_i}_{i in calibration}) with finite-sample correction
    """
    scores = []
    labels = []
    
    for ex in examples:
        cscore = compute_cscore_from_example(ex, w1, w2)
        scores.append(cscore)
        labels.append(1 if ex.get("conflict_present", False) else 0)
    
    return np.array(scores), np.array(labels)


def calibrate_threshold(scores, labels, alpha):
    """
    Split conformal prediction for selective prediction.
    
    Find τ such that if we ABSTAIN when C_score > τ,
    then among non-abstained examples, the error rate ≤ α.
    
    For our setting:
    - "Error" = conflict present but model says safe (miss)
    - We want: among examples where C_score ≤ τ (not abstained),
      the fraction of conflicts is ≤ α
    
    Standard conformal quantile with finite-sample correction:
      τ = ⌈(1-α)(n+1)⌉-th smallest score among CONFLICT examples
    
    This ensures that at most α fraction of true conflicts
    have C_score ≤ τ (i.e., are missed).
    """
    n = len(scores)
    
    # For coverage guarantee on conflict detection:
    # We need the (1-α) quantile of C-scores among conflict examples
    conflict_scores = scores[labels == 1]
    n_conflicts = len(conflict_scores)
    
    if n_conflicts == 0:
        return 0.5  # default
    
    # Finite-sample corrected quantile level
    # q = ⌈(1-α)(n+1)⌉ / n, but we use the simpler form:
    adjusted_quantile = np.ceil((1 - alpha) * (n_conflicts + 1)) / n_conflicts
    adjusted_quantile = min(adjusted_quantile, 1.0)
    
    # The threshold: (1-α) quantile of conflict C-scores
    # Conflicts with C_score BELOW this threshold would be missed
    # So we set τ at this level — abstain above, keep below
    # Actually we want: τ is the α-quantile of conflict scores
    # (the lowest α fraction of conflict scores are the ones we'd miss)
    tau = np.quantile(conflict_scores, alpha)
    
    # But with the conformal correction:
    sorted_conflict_scores = np.sort(conflict_scores)
    idx = int(np.ceil(alpha * (n_conflicts + 1))) - 1
    idx = max(0, min(idx, n_conflicts - 1))
    tau_conformal = sorted_conflict_scores[idx]
    
    return tau_conformal


def evaluate_at_threshold(scores, labels, tau):
    """
    Evaluate performance at a given threshold τ.
    
    Decision:
    - C_score > τ → ABSTAIN (flag as conflict)
    - C_score ≤ τ → PREDICT (consider safe)
    
    Returns:
    - coverage: fraction of examples not abstained
    - accuracy: accuracy among non-abstained examples
    - recall: fraction of true conflicts correctly abstained
    - precision: fraction of abstentions that are true conflicts
    - missed_conflicts: number of true conflicts with C_score ≤ τ
    """
    n = len(scores)
    
    abstained = scores > tau
    predicted = ~abstained
    
    coverage = predicted.sum() / n
    
    # Among predicted (not abstained): how many are actually safe?
    predicted_labels = labels[predicted]
    if len(predicted_labels) > 0:
        accuracy = (predicted_labels == 0).sum() / len(predicted_labels)
        error_rate = (predicted_labels == 1).sum() / len(predicted_labels)
    else:
        accuracy = 1.0
        error_rate = 0.0
    
    # Among true conflicts: how many were abstained (caught)?
    true_conflicts = labels == 1
    if true_conflicts.sum() > 0:
        recall = (abstained & true_conflicts).sum() / true_conflicts.sum()
        missed = (~abstained & true_conflicts).sum()
    else:
        recall = 1.0
        missed = 0
    
    # Precision of abstention
    if abstained.sum() > 0:
        precision = (abstained & true_conflicts).sum() / abstained.sum()
    else:
        precision = 1.0
    
    return {
        "tau": tau,
        "coverage": round(coverage, 4),
        "accuracy_on_covered": round(accuracy, 4),
        "error_rate_on_covered": round(error_rate, 4),
        "conflict_recall": round(recall, 4),
        "abstention_precision": round(precision, 4),
        "n_abstained": int(abstained.sum()),
        "n_predicted": int(predicted.sum()),
        "n_missed_conflicts": int(missed),
        "n_total": n,
    }


# ============================================================================
# WEIGHT OPTIMIZATION (Optional)
# ============================================================================

def optimize_weights(examples, alpha=0.05):
    """
    Find optimal w1, w2 via grid search on calibration set.
    Objective: maximize coverage while maintaining error_rate ≤ alpha.
    """
    best_coverage = 0
    best_w1, best_w2 = W1_DEFAULT, W2_DEFAULT
    best_tau = 0.5
    
    for w1 in np.arange(0.1, 1.0, 0.1):
        w2 = 1.0 - w1
        scores, labels = compute_nonconformity_scores(examples, w1, w2)
        tau = calibrate_threshold(scores, labels, alpha)
        metrics = evaluate_at_threshold(scores, labels, tau)
        
        if metrics["error_rate_on_covered"] <= alpha and metrics["coverage"] > best_coverage:
            best_coverage = metrics["coverage"]
            best_w1, best_w2 = w1, w2
            best_tau = tau
    
    return round(best_w1, 2), round(best_w2, 2), best_tau


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_coverage_accuracy_curve(scores, labels, output_path):
    """Plot coverage vs accuracy at different thresholds."""
    
    thresholds = np.linspace(0, 1, 200)
    coverages = []
    accuracies = []
    recalls = []
    
    for tau in thresholds:
        metrics = evaluate_at_threshold(scores, labels, tau)
        coverages.append(metrics["coverage"])
        accuracies.append(metrics["accuracy_on_covered"])
        recalls.append(metrics["conflict_recall"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Coverage vs Accuracy
    ax1.plot(coverages, accuracies, "b-", linewidth=2)
    for alpha in ALPHA_VALUES:
        tau = calibrate_threshold(scores, labels, alpha)
        m = evaluate_at_threshold(scores, labels, tau)
        ax1.plot(m["coverage"], m["accuracy_on_covered"], "ro", markersize=8)
        ax1.annotate(f"α={alpha}\nτ={tau:.3f}", 
                     (m["coverage"], m["accuracy_on_covered"]),
                     textcoords="offset points", xytext=(10, -10), fontsize=8)
    ax1.set_xlabel("Coverage (fraction not abstained)", fontsize=12)
    ax1.set_ylabel("Accuracy on covered set", fontsize=12)
    ax1.set_title("Coverage–Accuracy Trade-off", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Threshold vs Recall
    ax2.plot(thresholds, recalls, "g-", linewidth=2, label="Conflict Recall")
    ax2.plot(thresholds, coverages, "b--", linewidth=2, label="Coverage")
    for alpha in ALPHA_VALUES:
        tau = calibrate_threshold(scores, labels, alpha)
        ax2.axvline(x=tau, color="red", linestyle=":", alpha=0.5)
        ax2.annotate(f"α={alpha}", (tau, 0.5), fontsize=8, rotation=90)
    ax2.set_xlabel("Threshold τ", fontsize=12)
    ax2.set_ylabel("Rate", fontsize=12)
    ax2.set_title("Recall & Coverage vs Threshold", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_cscore_distribution(scores, labels, output_path):
    """Plot C-score distributions for conflict vs safe examples."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    safe_scores = scores[labels == 0]
    conflict_scores = scores[labels == 1]
    
    bins = np.linspace(0, 1, 40)
    ax.hist(safe_scores, bins=bins, alpha=0.6, color="green", label=f"Safe (n={len(safe_scores)})",
            density=True)
    ax.hist(conflict_scores, bins=bins, alpha=0.6, color="red", label=f"Conflict (n={len(conflict_scores)})",
            density=True)
    
    for alpha in [0.05, 0.10]:
        tau = calibrate_threshold(scores, labels, alpha)
        ax.axvline(x=tau, color="black", linestyle="--", linewidth=1.5)
        ax.annotate(f"τ(α={alpha})={tau:.3f}", (tau, ax.get_ylim()[1] * 0.9),
                    fontsize=9, rotation=90, va="top")
    
    ax.set_xlabel("C-Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("C-Score Distribution: Safe vs Conflict", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Conformal Prediction Calibration for C-Score")
    print("=" * 70)
    
    # Load benchmark
    bench_path = BENCH_DIR / "medconflict_bench_v2.jsonl"
    print(f"\nLoading benchmark from: {bench_path}")
    
    examples = []
    with open(bench_path) as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"  Total examples: {len(examples)}")
    
    # Split
    cal_examples = [ex for ex in examples if ex.get("split") == "calibration"]
    test_examples = [ex for ex in examples if ex.get("split") == "test"]
    
    print(f"  Calibration: {len(cal_examples)}")
    print(f"  Test:        {len(test_examples)}")
    
    # ------------------------------------------------------------------
    # Step 1: Optimize weights on calibration set
    # ------------------------------------------------------------------
    print("\n[1/4] Optimizing C-score weights...")
    
    w1_opt, w2_opt, _ = optimize_weights(cal_examples, alpha=0.05)
    print(f"  Optimal weights: w1={w1_opt}, w2={w2_opt}")
    
    # ------------------------------------------------------------------
    # Step 2: Compute C-scores
    # ------------------------------------------------------------------
    print("\n[2/4] Computing C-scores...")
    
    cal_scores, cal_labels = compute_nonconformity_scores(cal_examples, w1_opt, w2_opt)
    test_scores, test_labels = compute_nonconformity_scores(test_examples, w1_opt, w2_opt)
    
    print(f"  Cal  — conflicts: {cal_labels.sum()}, safe: {(cal_labels == 0).sum()}")
    print(f"  Test — conflicts: {test_labels.sum()}, safe: {(test_labels == 0).sum()}")
    
    # ------------------------------------------------------------------
    # Step 3: Calibrate thresholds
    # ------------------------------------------------------------------
    print("\n[3/4] Calibrating thresholds...")
    
    calibration_results = {}
    
    print(f"\n  {'α':<8} {'τ':<10} {'Coverage':<12} {'Accuracy':<12} {'Recall':<12} "
          f"{'Missed':<10}")
    print("  " + "-" * 64)
    
    for alpha in ALPHA_VALUES:
        tau = calibrate_threshold(cal_scores, cal_labels, alpha)
        
        # Evaluate on TEST set (not calibration)
        test_metrics = evaluate_at_threshold(test_scores, test_labels, tau)
        
        calibration_results[str(alpha)] = {
            "alpha": alpha,
            "tau": round(tau, 4),
            "w1": w1_opt,
            "w2": w2_opt,
            "cal_metrics": evaluate_at_threshold(cal_scores, cal_labels, tau),
            "test_metrics": test_metrics,
        }
        
        print(f"  {alpha:<8} {tau:<10.4f} {test_metrics['coverage']:<12.4f} "
              f"{test_metrics['accuracy_on_covered']:<12.4f} "
              f"{test_metrics['conflict_recall']:<12.4f} "
              f"{test_metrics['n_missed_conflicts']:<10}")
    
    # ------------------------------------------------------------------
    # Step 4: Save results and plots
    # ------------------------------------------------------------------
    print("\n[4/4] Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save calibration results
    results_path = OUTPUT_DIR / "calibration_results.json"
    output = {
        "method": "split_conformal_prediction",
        "weights": {"w1": w1_opt, "w2": w2_opt},
        "alphas": calibration_results,
        "n_calibration": len(cal_examples),
        "n_test": len(test_examples),
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {results_path}")
    
    # Per-stratum analysis on test set
    print("\n  Per-stratum analysis (test set, α=0.05):")
    tau_005 = calibration_results["0.05"]["tau"]
    
    for stratum in [0, 1, 2, 3]:
        stratum_mask = np.array([ex.get("stratum") == stratum for ex in test_examples])
        if stratum_mask.sum() == 0:
            continue
        s_scores = test_scores[stratum_mask]
        s_labels = test_labels[stratum_mask]
        s_metrics = evaluate_at_threshold(s_scores, s_labels, tau_005)
        name = {0: "Safe", 1: "Categorical", 2: "Numerical", 3: "Multi-hop"}[stratum]
        print(f"    Stratum {stratum} ({name}): "
              f"n={stratum_mask.sum()}, "
              f"recall={s_metrics['conflict_recall']:.3f}, "
              f"coverage={s_metrics['coverage']:.3f}")
    
    # Plots
    plot_coverage_accuracy_curve(
        test_scores, test_labels,
        OUTPUT_DIR / "coverage_accuracy_curve.png"
    )
    plot_cscore_distribution(
        test_scores, test_labels,
        OUTPUT_DIR / "cscore_distribution.png"
    )
    
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nRecommended threshold (α=0.05): τ = {calibration_results['0.05']['tau']:.4f}")
    print(f"  → Coverage: {calibration_results['0.05']['test_metrics']['coverage']:.1%}")
    print(f"  → Conflict recall: {calibration_results['0.05']['test_metrics']['conflict_recall']:.1%}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
