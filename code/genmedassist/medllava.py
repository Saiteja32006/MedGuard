#!/usr/bin/env python3
r"""
07_eval_text_models.py — Evaluate Text-Only Medical Models on MedConflict-Bench
==============================================================================

Phase 1: Pure Text Models
Runs Meditron-7B, BioMistral-7B, Llama-3-8B, and MedGemma-4B.

Prerequisites:
  pip install transformers torch accelerate bitsandbytes tqdm scikit-learn
  huggingface-cli login (Required for Llama-3 and MedGemma)
"""

import json
import argparse
import os
import gc
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================================
# CONFIG
# ============================================================================

BENCH_INPUT = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "medconflict_bench_v2.jsonl"
RESULTS_DIR = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1: Text-Only Model Registry
MODELS = {
    "meditron-7b": {
        "hf_id": "epfl-llm/meditron-7b",
        "type": "base", 
    },
    "biomistral-7b": {
        "hf_id": "BioMistral/BioMistral-7B",
        "type": "base", 
    },
    "llama-3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "type": "instruct",
    },
    "medgemma-4b": {
        "hf_id": "google/medgemma-4b-it",
        "type": "instruct", 
    },
}

# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a clinical pharmacist reviewing a patient's medication order.
Determine if there is a clinically significant conflict between the prescribed medication and the patient's clinical data.

A conflict can be:
1. A drug prescribed to a patient with a documented allergy or contraindicated condition
2. A drug prescribed when a lab value exceeds a safety threshold
3. A drug-drug interaction that becomes dangerous given the patient's lab values

Respond with EXACTLY this JSON format and nothing else:
{"conflict_detected": true/false, "severity": "none"|"moderate"|"major", "explanation": "brief reasoning"}"""

def build_prompt(example):
    parts = ["=== MEDICATION ORDER REVIEW ===\n"]

    drug = example.get("drug", "Unknown")
    dose = example.get("dose_val", "")
    unit = example.get("dose_unit", "")
    route = example.get("route", "")

    drug_line = f"Prescribed: {drug}"
    if dose and unit and dose != "nan" and unit != "nan":
        drug_line += f" {dose} {unit}"
    if route and route != "nan":
        drug_line += f" ({route})"
    parts.append(drug_line)

    if example.get("interacting_drug"):
        parts.append(f"Co-prescribed: {example['interacting_drug']}")

    lab_name = example.get("lab_name")
    lab_value = example.get("lab_value")
    lab_unit = example.get("lab_unit")

    if lab_name and lab_value is not None:
        lab_line = f"\nLab: {lab_name} = {lab_value}"
        if lab_unit and str(lab_unit) != "nan":
            lab_line += f" {lab_unit}"
        parts.append(lab_line)

    condition = example.get("conflict_condition", "")
    if example.get("stratum") == 1 and condition and condition != "none":
        parts.append(f"\nPatient history: {condition}")

    parts.append("\n=== END ===")
    parts.append("\nIs there a conflict? Respond with JSON only.")

    return "\n".join(parts)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_key):
    """Load text model and tokenizer cleanly."""
    config = MODELS[model_key]
    hf_id = config["hf_id"]

    print(f"\nLoading {model_key} from {hf_id}...")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16 # Protects against NaN math crashes
    )
    
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    
    return model, tokenizer, config["type"]

# ============================================================================
# GENERATION
# ============================================================================

def generate_response(model, tokenizer, model_type, prompt, max_tokens=300):
    """Generate a response using correct formatting per model."""
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    if model_type == "instruct":
        # Instruction-tuned models prefer chat templates
        messages = [{"role": "user", "content": full_prompt}]
        try:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_text = full_prompt
    else:
        # Base models (like Meditron/BioMistral) just get the raw string
        formatted_text = full_prompt

    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=0.1,
            do_sample=True, 
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()

# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_response(raw_response):
    """Extract conflict_detected from model response."""
    text = raw_response.lower().strip()

    json_match = re.search(r'\{[^}]+\}', raw_response)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            detected = parsed.get("conflict_detected", parsed.get("conflict", False))
            if isinstance(detected, str):
                detected = detected.lower() in ["true", "yes", "1"]
            return bool(detected)
        except json.JSONDecodeError:
            pass

    conflict_signals = ["conflict_detected\": true", "conflict detected", "yes, there is a conflict",
                        "significant conflict", "contraindicated", "should be held",
                        "should not be", "risk of", "dangerous", "\"true\""]
    safe_signals = ["conflict_detected\": false", "no conflict", "safe to proceed",
                    "no clinically significant", "\"false\"", "no significant"]

    for signal in conflict_signals:
        if signal in text: return True
    for signal in safe_signals:
        if signal in text: return False

    danger_words = ["contraindicated", "hold", "discontinue", "risk", "toxicity", "avoid"]
    return any(w in text for w in danger_words)

# ============================================================================
# METRICS & MAIN LOOP
# ============================================================================

def compute_metrics(results, model_name):
    y_true = np.array([1 if r["gt_conflict"] else 0 for r in results])
    y_pred = np.array([1 if r["pred_conflict"] else 0 for r in results])

    overall = {
        "model": model_name,
        "n": len(results),
        "accuracy": round((y_true == y_pred).mean(), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    per_stratum = {}
    for s in [0, 1, 2, 3]:
        mask = np.array([r["stratum"] == s for r in results])
        if mask.sum() == 0: continue
        s_true, s_pred = y_true[mask], y_pred[mask]
        
        m = {"n": int(mask.sum()), "accuracy": round((s_true == s_pred).mean(), 4)}
        if s == 0:
            m["specificity"] = round(1 - s_pred.mean(), 4)
        else:
            m["detection_rate"] = round(s_pred.mean(), 4)
        per_stratum[s] = m

    cliff = {}
    if 1 in per_stratum and 2 in per_stratum:
        s1, s2 = per_stratum[1].get("detection_rate", 0), per_stratum[2].get("detection_rate", 0)
        cliff["s1_categorical"] = s1
        cliff["s2_numerical"] = s2
        cliff["absolute_drop"] = round(s1 - s2, 4)
    if 3 in per_stratum:
        cliff["s3_multihop"] = per_stratum[3].get("detection_rate", 0)

    return {"overall": overall, "per_stratum": per_stratum, "cliff": cliff}

def evaluate_model(model_key):
    ckpt_path = RESULTS_DIR / f"{model_key}_results.jsonl"
    if ckpt_path.exists():
        print(f"\n✅ {model_key} already evaluated. Loading results...")
        results = [json.loads(l) for l in open(ckpt_path, encoding="utf-8")]
        return results, compute_metrics(results, model_key)

    examples = [json.loads(l) for l in open(BENCH_INPUT, encoding="utf-8")]
    
    # NOTE: I removed [:20] so it will process ALL 1,261 examples!
    test_examples = [e for e in examples if e.get("split") == "test"] 
    print(f"\nBenchmark: {len(test_examples)} test examples")

    model, tokenizer, model_type = load_model(model_key)
    
    results = []
    for i, ex in enumerate(tqdm(test_examples, desc=f"  {model_key}")):
        prompt = build_prompt(ex)
        try:
            raw = generate_response(model, tokenizer, model_type, prompt, max_tokens=300)
            pred = parse_response(raw)
        except Exception as e:
            raw, pred = f"ERROR: {e}", False

        results.append({
            "example_id": ex["example_id"],
            "stratum": ex["stratum"],
            "gt_conflict": ex.get("conflict_present", False),
            "pred_conflict": pred,
            "raw_response": raw[:500], 
        })

        if (i + 1) % 100 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r) + "\n")

    with open(ckpt_path, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results, compute_metrics(results, model_key)

def print_comparison(all_metrics):
    print("\n" + "=" * 80)
    print("RESULTS — Per-Stratum Detection Rates (THE PERFORMANCE CLIFF)")
    print("=" * 80)
    header = f"{'Model':<18} {'S0-Spec':<10} {'S1-Cat':<10} {'S2-Num':<10} {'S3-MH':<10} {'Cliff':<12} {'F1':<8}"
    print(header + "\n" + "-" * len(header))

    for model, metrics in all_metrics.items():
        ps, cliff, o = metrics["per_stratum"], metrics["cliff"], metrics["overall"]
        s0 = f"{ps.get(0, {}).get('specificity', '—'):.3f}" if 0 in ps else "—"
        s1 = f"{ps.get(1, {}).get('detection_rate', '—'):.3f}" if 1 in ps else "—"
        s2 = f"{ps.get(2, {}).get('detection_rate', '—'):.3f}" if 2 in ps else "—"
        s3 = f"{ps.get(3, {}).get('detection_rate', '—'):.3f}" if 3 in ps else "—"
        drop = f"{cliff.get('absolute_drop', 0):+.3f}" if cliff else "—"
        f1 = f"{o['f1']:.3f}"
        print(f"{model:<18} {s0:<10} {s1:<10} {s2:<10} {s3:<10} {drop:<12} {f1:<8}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(MODELS.keys()) + ["all"])
    args = parser.parse_args()

    if not BENCH_INPUT.exists():
        print(f"❌ Benchmark not found: {BENCH_INPUT}")
        return

    models_to_eval = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_metrics = {}

    for model_key in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key}")
        print(f"{'='*60}")

        try:
            results, metrics = evaluate_model(model_key)
            all_metrics[model_key] = metrics

            with open(RESULTS_DIR / f"{model_key}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"\n  {model_key} Results:")
            print(f"    Overall F1: {metrics['overall']['f1']}")
            print(f"    Cliff: {metrics['cliff']}")
        except Exception as e:
            print(f"\n  ❌ {model_key} failed: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        print_comparison(all_metrics)

        with open(RESULTS_DIR / "all_models_comparison.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR / 'all_models_comparison.json'}")

if __name__ == "__main__":
    main()