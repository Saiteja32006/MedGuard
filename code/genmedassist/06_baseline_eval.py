#!/usr/bin/env python3
"""
06_baseline_eval.py — Baseline Evaluation on MedConflict-Bench
================================================================

Evaluates frontier LLMs on MedConflict-Bench v2 to measure:
  1. Per-stratum detection rates
  2. The "performance cliff" between categorical and numerical conflicts
  3. Overall conflict detection metrics

Models evaluated:
  - GPT-4o (via OpenAI API)
  - o4-mini (via OpenAI API)
  - (Optional) Open models via local inference

Usage (Google Colab):
  1. Set OPENAI_API_KEY environment variable
  2. Point BENCH_DIR to MedConflict-Bench v2
  3. Run: python 06_baseline_eval.py

Author: Ala Sai Teja, IIIT Sricity
Project: GenMedAssist — NeurIPS 2026

Requirements:
  pip install openai pandas numpy scikit-learn tqdm
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    raise

try:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )
except ImportError:
    print("Install scikit-learn: pip install scikit-learn")
    raise


# ============================================================================
# CONFIGURATION
# ============================================================================

# Lab machine:
BENCH_DIR = Path(os.getenv("GENMEDASSIST_DIR", "."))
# Colab:
# BENCH_DIR = Path(os.getenv("BENCH_DIR", "./MedConflict-Bench"))

OUTPUT_DIR = BENCH_DIR / "eval_results"

# Models to evaluate
MODELS = [
    {"name": "gpt-4o", "model_id": "gpt-4o", "provider": "openai"},
    {"name": "o4-mini", "model_id": "o4-mini", "provider": "openai"},
]

# Rate limiting
REQUESTS_PER_MINUTE = 30  # conservative
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE

# Max examples per stratum (for cost control during development)
MAX_PER_STRATUM = None  # Set to e.g. 50 for testing, None for full eval

SEED = 42


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

SYSTEM_PROMPT = """You are a clinical pharmacist reviewing a patient's medication order.
Your task is to determine if there is a clinically significant conflict between the
prescribed medication and the patient's clinical data.

A conflict can be:
1. A drug prescribed to a patient with a documented allergy or contraindicated condition
2. A drug prescribed when a lab value exceeds a safety threshold
3. A drug-drug interaction that becomes dangerous given the patient's lab values

Respond with EXACTLY this JSON format (no other text):
{
  "conflict_detected": true/false,
  "severity": "none" | "moderate" | "major",
  "conflict_type": "none" | "drug_allergy" | "drug_disease" | "numerical_threshold" | "drug_drug_interaction",
  "explanation": "Brief clinical reasoning (1-2 sentences)",
  "recommended_action": "none" | "contraindicated" | "dose_reduction" | "hold_dose" | "monitor_closely" | "discontinue"
}"""


def build_user_prompt(example):
    """
    Construct a clinical scenario prompt from a benchmark example.
    
    The prompt presents the information as a pharmacist would see it:
    prescription details + relevant lab values + any diagnoses.
    The model must determine if there's a conflict.
    """
    parts = []
    
    parts.append("=== MEDICATION ORDER REVIEW ===\n")
    
    # Drug information
    drug = example.get("drug", "Unknown")
    dose = example.get("dose_val")
    unit = example.get("dose_unit")
    route = example.get("route")
    
    drug_line = f"Prescribed Medication: {drug}"
    if dose and unit:
        drug_line += f" {dose} {unit}"
    if route:
        drug_line += f" ({route})"
    parts.append(drug_line)
    
    # Interacting drug (for multi-hop)
    if example.get("interacting_drug"):
        parts.append(f"Co-prescribed Medication: {example['interacting_drug']}")
    
    # Lab values
    lab_name = example.get("lab_name")
    lab_value = example.get("lab_value")
    lab_unit = example.get("lab_unit")
    
    if lab_name and lab_value is not None:
        parts.append(f"\nRelevant Lab Result:")
        lab_line = f"  {lab_name}: {lab_value}"
        if lab_unit:
            lab_line += f" {lab_unit}"
        parts.append(lab_line)
    
    # Diagnoses / conditions (for categorical conflicts)
    if example.get("stratum") == 1:
        condition = example.get("conflict_condition", "")
        if condition:
            parts.append(f"\nPatient History / Active Diagnoses:")
            parts.append(f"  - {condition}")
    
    parts.append("\n=== END OF ORDER ===")
    parts.append("\nIs there a clinically significant conflict with this medication order?")
    parts.append("Respond with the JSON format specified.")
    
    return "\n".join(parts)


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def call_openai(client, model_id, system_prompt, user_prompt, max_retries=3):
    """Call OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```" in raw:
                    raw = raw.split("```json")[-1].split("```")[0].strip()
                    parsed = json.loads(raw)
                else:
                    parsed = {"conflict_detected": False, "parse_error": True}
            
            return parsed, raw
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return {"conflict_detected": False, "api_error": str(e)}, ""


def evaluate_model(model_config, examples, output_dir):
    """Evaluate a single model on all examples with checkpoint/resume."""
    
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    
    # Check if already completed
    ckpt_path = output_dir / f"{model_name}_results.jsonl"
    if ckpt_path.exists():
        existing = [json.loads(l) for l in open(ckpt_path, encoding="utf-8")]
        if len(existing) >= len(examples):
            print(f"\n✅ {model_name} already completed ({len(existing)} results). Skipping.")
            return existing
        else:
            print(f"\n⏩ {model_name} resuming from {len(existing)}/{len(examples)}...")
            results = existing
            done_ids = set(r["example_id"] for r in existing)
            examples = [ex for ex in examples if ex["example_id"] not in done_ids]
    else:
        results = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} ({model_id}) — {len(examples)} remaining")
    print(f"{'='*60}")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    for i, example in enumerate(tqdm(examples, desc=f"  {model_name}")):
        user_prompt = build_user_prompt(example)
        
        prediction, raw_response = call_openai(client, model_id, SYSTEM_PROMPT, user_prompt)
        
        result = {
            "example_id": example["example_id"],
            "stratum": example["stratum"],
            "stratum_name": example.get("stratum_name", ""),
            "ground_truth_conflict": example.get("conflict_present", False),
            "predicted_conflict": prediction.get("conflict_detected", False),
            "predicted_severity": prediction.get("severity", "none"),
            "predicted_type": prediction.get("conflict_type", "none"),
            "predicted_action": prediction.get("recommended_action", "none"),
            "predicted_explanation": prediction.get("explanation", ""),
            "raw_response": raw_response,
            "rule_id": example.get("rule_id", ""),
        }
        results.append(result)
        
        # Save checkpoint every 50 examples
        if (i + 1) % 50 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"    💾 Checkpoint at {len(results)}")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Save final results
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Saved: {ckpt_path}")
    
    return results


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(results, model_name):
    """Compute comprehensive metrics from evaluation results."""
    
    y_true = [1 if r["ground_truth_conflict"] else 0 for r in results]
    y_pred = [1 if r["predicted_conflict"] else 0 for r in results]
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    overall = {
        "model": model_name,
        "n_examples": len(results),
        "accuracy": round((y_true == y_pred).mean(), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_conflict": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    overall["true_positives"] = int(tp)
    overall["false_positives"] = int(fp)
    overall["true_negatives"] = int(tn)
    overall["false_negatives"] = int(fn)
    
    # Per-stratum metrics
    stratum_metrics = {}
    stratum_names = {0: "safe", 1: "categorical", 2: "numerical_threshold", 3: "multi_hop_causal"}
    
    for stratum in [0, 1, 2, 3]:
        mask = np.array([r["stratum"] == stratum for r in results])
        if mask.sum() == 0:
            continue
        
        s_true = y_true[mask]
        s_pred = y_pred[mask]
        
        s_metrics = {
            "stratum": stratum,
            "stratum_name": stratum_names.get(stratum, f"stratum_{stratum}"),
            "n_examples": int(mask.sum()),
            "accuracy": round((s_true == s_pred).mean(), 4),
        }
        
        if stratum == 0:
            # Safe stratum: measure false positive rate
            s_metrics["false_positive_rate"] = round(s_pred.mean(), 4)
            s_metrics["specificity"] = round(1 - s_pred.mean(), 4)
        else:
            # Conflict strata: measure detection rate (recall)
            s_metrics["detection_rate"] = round(s_pred.mean(), 4)  # all should be 1
            s_metrics["recall"] = round(recall_score(s_true, s_pred, zero_division=0), 4)
            s_metrics["f1"] = round(f1_score(s_true, s_pred, zero_division=0), 4)
        
        stratum_metrics[stratum] = s_metrics
    
    # THE KEY FINDING: Performance cliff
    cliff = {}
    if 1 in stratum_metrics and 2 in stratum_metrics:
        s1_det = stratum_metrics[1].get("detection_rate", 0)
        s2_det = stratum_metrics[2].get("detection_rate", 0)
        cliff["categorical_detection"] = s1_det
        cliff["numerical_detection"] = s2_det
        cliff["absolute_drop"] = round(s1_det - s2_det, 4)
        cliff["relative_drop_pct"] = round(
            ((s1_det - s2_det) / s1_det * 100) if s1_det > 0 else 0, 2
        )
    if 3 in stratum_metrics:
        cliff["multihop_detection"] = stratum_metrics[3].get("detection_rate", 0)
    
    return {
        "overall": overall,
        "per_stratum": stratum_metrics,
        "performance_cliff": cliff,
    }


def print_metrics_table(all_metrics):
    """Print a formatted comparison table across models."""
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Overall
    print("\n--- Overall Metrics ---")
    header = f"{'Model':<15} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Recall':<8} {'Macro-F1':<8}"
    print(header)
    print("-" * len(header))
    for model, metrics in all_metrics.items():
        o = metrics["overall"]
        print(f"{model:<15} {o['accuracy']:<8} {o['f1_conflict']:<8} "
              f"{o['precision']:<8} {o['recall']:<8} {o['macro_f1']:<8}")
    
    # Per-stratum detection rates (THE KEY TABLE)
    print("\n--- Per-Stratum Detection Rates ---")
    print("(This is the 'performance cliff' finding)")
    header = f"{'Model':<15} {'S0-Safe':<12} {'S1-Cat':<12} {'S2-Num':<12} {'S3-MH':<12} {'Cliff(S1→S2)':<14}"
    print(header)
    print("-" * len(header))
    for model, metrics in all_metrics.items():
        ps = metrics["per_stratum"]
        cliff = metrics["performance_cliff"]
        
        s0 = ps.get(0, {}).get("specificity", "—")
        s1 = ps.get(1, {}).get("detection_rate", "—")
        s2 = ps.get(2, {}).get("detection_rate", "—")
        s3 = ps.get(3, {}).get("detection_rate", "—")
        drop = cliff.get("absolute_drop", "—")
        
        if isinstance(s0, float): s0 = f"{s0:.3f}"
        if isinstance(s1, float): s1 = f"{s1:.3f}"
        if isinstance(s2, float): s2 = f"{s2:.3f}"
        if isinstance(s3, float): s3 = f"{s3:.3f}"
        if isinstance(drop, float): drop = f"-{abs(drop):.3f}"
        
        print(f"{model:<15} {s0:<12} {s1:<12} {s2:<12} {s3:<12} {drop:<14}")
    
    # Performance cliff detail
    print("\n--- Performance Cliff Analysis ---")
    for model, metrics in all_metrics.items():
        cliff = metrics["performance_cliff"]
        if cliff:
            print(f"\n  {model}:")
            print(f"    Categorical detection:  {cliff.get('categorical_detection', '?'):.1%}")
            print(f"    Numerical detection:    {cliff.get('numerical_detection', '?'):.1%}")
            print(f"    Multi-hop detection:    {cliff.get('multihop_detection', '?'):.1%}")
            print(f"    Absolute cliff (S1→S2): {cliff.get('absolute_drop', 0):.1%}")
            print(f"    Relative cliff:         {cliff.get('relative_drop_pct', 0):.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MedConflict-Bench v2 — Baseline Evaluation")
    print("=" * 70)
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Set it with:")
        print("    os.environ['OPENAI_API_KEY'] = 'sk-...'")
        print("    or: export OPENAI_API_KEY='sk-...'")
        return
    
    # Load benchmark (test split only)
    bench_path = BENCH_DIR / "medconflict_bench_v2.jsonl"
    print(f"\nLoading benchmark: {bench_path}")
    
    all_examples = []
    with open(bench_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("split") == "test":
                all_examples.append(ex)
    
    print(f"  Test examples: {len(all_examples)}")
    
    # Subsample if MAX_PER_STRATUM is set
    if MAX_PER_STRATUM:
        sampled = []
        for stratum in [0, 1, 2, 3]:
            stratum_exs = [ex for ex in all_examples if ex["stratum"] == stratum]
            if len(stratum_exs) > MAX_PER_STRATUM:
                np.random.seed(SEED)
                stratum_exs = list(np.random.choice(stratum_exs, MAX_PER_STRATUM, replace=False))
            sampled.extend(stratum_exs)
        all_examples = sampled
        print(f"  Subsampled to: {len(all_examples)} (max {MAX_PER_STRATUM}/stratum)")
    
    # Stratum distribution
    for s in [0, 1, 2, 3]:
        count = sum(1 for ex in all_examples if ex["stratum"] == s)
        name = {0: "Safe", 1: "Categorical", 2: "Numerical", 3: "Multi-hop"}[s]
        print(f"    Stratum {s} ({name}): {count}")
    
    # Evaluate each model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    
    for model_config in MODELS:
        results = evaluate_model(model_config, all_examples, OUTPUT_DIR)
        metrics = compute_metrics(results, model_config["name"])
        all_metrics[model_config["name"]] = metrics
        
        # Save metrics
        metrics_path = OUTPUT_DIR / f"{model_config['name']}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved metrics: {metrics_path}")
    
    # Print comparison table
    print_metrics_table(all_metrics)
    
    # Save combined results
    combined_path = OUTPUT_DIR / "all_models_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nSaved combined results: {combined_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()