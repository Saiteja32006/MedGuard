#!/usr/bin/env python3
r"""
07_eval_medical_models.py — Evaluate Open Medical Models on MedConflict-Bench
==============================================================================

Runs QoQ-Med-7B, Med-R1-2B, MedGemma-4B, LLaVA-Med-7B on the benchmark.
Each model processes all test examples and outputs per-stratum detection rates.

RUN ON LAB MACHINE (RTX A5000 24GB):
  cd C:\Users\AIML\Sai_Teja _Honors\genmedassist
  python 07_eval_medical_models.py --model qoq-med-7b
  python 07_eval_medical_models.py --model med-r1
  python 07_eval_medical_models.py --model medgemma
  python 07_eval_medical_models.py --model llava-med

Or run all sequentially:
  python 07_eval_medical_models.py --model all

Prerequisites:
  pip install transformers torch accelerate bitsandbytes tqdm scikit-learn

Model downloads happen automatically on first run via HuggingFace.

Author: Ala Sai Teja, IIIT Sricity
"""

import json
import argparse
import time
import os
import gc
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ============================================================================
# CONFIG
# ============================================================================

BENCH_INPUT = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "medconflict_bench_v2.jsonl"
RESULTS_DIR = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model registry
MODELS = {
    "qoq-med-7b": {
        "hf_id": "ddvd233/QoQ-Med-VL-7B",
        "type": "vision2seq",  # AutoModelForVision2Seq
        "max_tokens": 512,
        "needs_image": False,  # our benchmark is text-only (no prescription images)
    },
    "med-r1": {
        "hf_id": "yuxianglai117/Med-R1", # <--- OFFICIAL REPO
        "type": "causal",  
        "max_tokens": 512,
        "needs_image": False,
    },
    "medgemma": {
        "hf_id": "google/medgemma-4b-it",
        "type": "causal",
        "max_tokens": 512,
        "needs_image": False,
    },
    "llava-med": {
        "hf_id": "microsoft/llava-med-v1.5-mistral-7b",
        "type": "vision2seq",
        "max_tokens": 512,
        "needs_image": False,
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
# MODEL LOADING — per-model handling
# ============================================================================

def load_model(model_key):
    import torch  # <--- ADD THIS HERE to fix the UnboundLocalError permanently
    """Load model and tokenizer/processor with correct classes per model."""
    config = MODELS[model_key]
    hf_id = config["hf_id"]

    print(f"\nLoading {model_key} from {hf_id}...")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
# ... rest of your function
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if model_key == "qoq-med-7b":
        # QoQ-Med-VL-7B is Qwen2-VL based
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            trust_remote_code=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        return model, processor, "qwen2vl"

    elif model_key == "med-r1":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from huggingface_hub import snapshot_download
        import shutil
        import os
        import re
        
        target_subfolder = "VQA_MRI"
        
        print(f"Downloading {target_subfolder} weights...")
        # 1. Download ONLY the specific subfolder (avoids downloading 100GB of other modalities)
        cache_dir = snapshot_download(
            repo_id=hf_id, 
            allow_patterns=[f"{target_subfolder}/*"]
        )
        source_model_path = os.path.join(cache_dir, target_subfolder)
        
        # 2. Copy to a local folder so we can safely edit the bad JSON files
        local_patched_dir = os.path.join(os.getcwd(), "patched_med_r1")
        if not os.path.exists(local_patched_dir):
            print("Creating a patched local copy of the model...")
            shutil.copytree(source_model_path, local_patched_dir, dirs_exist_ok=True)
            
            # 3. Physically fix the "None" typo in all config files!
            for filename in os.listdir(local_patched_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(local_patched_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # Regex search and replace the bad null value
                    if re.search(r'"use_cache"\s*:\s*null', text):
                        text = re.sub(r'"use_cache"\s*:\s*null', '"use_cache": true', text)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(text)
                            
        print("Config patched! Loading model into VRAM...")
        
        # 4. Load from our clean local folder
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        try:
            processor = AutoProcessor.from_pretrained(local_patched_dir, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
            
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_patched_dir, # Loading directly from our local folder
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        
        return model, processor, "qwen2vl"

    elif model_key == "medgemma":
        # MedGemma-4B-IT is Gemma based — needs chat template
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # FIX 1: Change compute dtype to bfloat16 to prevent NaN overflow
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16  # <--- Changed
        )
        
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch.bfloat16,                  # <--- Changed (also using 'dtype' instead of 'torch_dtype' to fix your warning!)
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        return model, tokenizer, "medgemma"

    elif model_key == "llava-med":
        from transformers import MistralForCausalLM, MistralConfig, PretrainedConfig, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        print("Extracting the internal Mistral LLM for text-only reasoning...")
        # 1. Grab the Mistral text dictionary
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=False)
        
        # 2. Intercept the raw config dictionary
        config_dict, _ = PretrainedConfig.get_config_dict(hf_id)
        
        # 3. Completely scrub the "llava" identity from the config
        config_dict["model_type"] = "mistral"
        config_dict["architectures"] = ["MistralForCausalLM"]
        
        # 4. Build a strict Mistral config from the patched dictionary
        config = MistralConfig.from_dict(config_dict)
        
        # 5. CRITICAL FIX: Use MistralForCausalLM directly! 
        # This bypasses the "Auto" verifier and forces the library to load the text brain.
        model = MistralForCausalLM.from_pretrained(
            hf_id,
            config=config,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True # Silently ignores the leftover camera weights
        )
        
        return model, tokenizer, "causal"

    else:
        # Generic fallback
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, device_map="auto",
            trust_remote_code=True, quantization_config=bnb_config,
        )
        return model, tokenizer, "causal"


def generate_response(model, tokenizer_or_processor, model_type, prompt, max_tokens=512):
    """Generate a response from the model."""
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    if model_type == "qwen2vl":
        # Qwen2-VL uses chat template with messages format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        text = tokenizer_or_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer_or_processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=True)

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer_or_processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    elif model_type == "llava":
        # LLaVA needs a dummy image for text-only input
        from PIL import Image
        dummy_img = Image.new("RGB", (224, 224), "white")
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt},
            ]},
        ]
        text = tokenizer_or_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = tokenizer_or_processor(images=dummy_img, text=text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=True)

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer_or_processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    elif model_type == "causal":
        inputs = tokenizer_or_processor(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.1,
                do_sample=True, top_p=0.9,
                pad_token_id=tokenizer_or_processor.pad_token_id or tokenizer_or_processor.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer_or_processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    elif model_type == "medgemma":
        # FIX: Gemma chat templates do not support the "system" role.
        # We must combine the system prompt and user prompt into a single user message.
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"},
        ]
        try:
            formatted = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = full_prompt

        inputs = tokenizer_or_processor(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer_or_processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return response.strip()


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_response(raw_response):
    """Extract conflict_detected from model response."""
    text = raw_response.lower().strip()

    # Try JSON parsing
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

    # Fallback: keyword detection
    conflict_signals = ["conflict_detected\": true", "conflict detected", "yes, there is a conflict",
                        "significant conflict", "contraindicated", "should be held",
                        "should not be", "risk of", "dangerous", "\"true\""]
    safe_signals = ["conflict_detected\": false", "no conflict", "safe to proceed",
                    "no clinically significant", "\"false\"", "no significant"]

    for signal in conflict_signals:
        if signal in text:
            return True
    for signal in safe_signals:
        if signal in text:
            return False

    # Default: if mentions any danger words, call it a conflict
    danger_words = ["contraindicated", "hold", "discontinue", "risk", "toxicity", "avoid"]
    return any(w in text for w in danger_words)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(results, model_name):
    y_true = np.array([1 if r["gt_conflict"] else 0 for r in results])
    y_pred = np.array([1 if r["pred_conflict"] else 0 for r in results])

    overall = {
        "model": model_name,
        "n": len(results),
        "accuracy": round((y_true == y_pred).mean(), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    overall.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})

    per_stratum = {}
    for s in [0, 1, 2, 3]:
        mask = np.array([r["stratum"] == s for r in results])
        if mask.sum() == 0:
            continue
        s_true = y_true[mask]
        s_pred = y_pred[mask]
        name = {0: "safe", 1: "categorical", 2: "numerical", 3: "multi_hop"}[s]
        m = {"n": int(mask.sum()), "accuracy": round((s_true == s_pred).mean(), 4)}
        if s == 0:
            m["false_positive_rate"] = round(s_pred.mean(), 4)
            m["specificity"] = round(1 - s_pred.mean(), 4)
        else:
            m["detection_rate"] = round(s_pred.mean(), 4)
            m["recall"] = round(recall_score(s_true, s_pred, zero_division=0), 4)
            m["f1"] = round(f1_score(s_true, s_pred, zero_division=0), 4)
        per_stratum[s] = m

    # Performance cliff
    cliff = {}
    if 1 in per_stratum and 2 in per_stratum:
        s1 = per_stratum[1].get("detection_rate", 0)
        s2 = per_stratum[2].get("detection_rate", 0)
        cliff["s1_categorical"] = s1
        cliff["s2_numerical"] = s2
        cliff["absolute_drop"] = round(s1 - s2, 4)
        cliff["relative_drop_pct"] = round((s1 - s2) / max(s1, 0.001) * 100, 1)
    if 3 in per_stratum:
        cliff["s3_multihop"] = per_stratum[3].get("detection_rate", 0)

    return {"overall": overall, "per_stratum": per_stratum, "cliff": cliff}


# ============================================================================
# MAIN
# ============================================================================

def evaluate_model(model_key):
    """Run full evaluation for one model."""
    ckpt_path = RESULTS_DIR / f"{model_key}_results.jsonl"

    # Check if already done
    if ckpt_path.exists():
        print(f"\n✅ {model_key} already evaluated. Loading results...")
        results = [json.loads(l) for l in open(ckpt_path, encoding="utf-8")]
        metrics = compute_metrics(results, model_key)
        return results, metrics

    # Load benchmark
    examples = [json.loads(l) for l in open(BENCH_INPUT, encoding="utf-8")]
    test_examples = [e for e in examples if e.get("split") == "test"]
    print(f"\nBenchmark: {len(test_examples)} test examples")

    # Load model
    model, tokenizer, model_type = load_model(model_key)
    print(f"  Model loaded: {model_key} ({model_type})")

    # Evaluate
    results = []
    for i, ex in enumerate(tqdm(test_examples, desc=f"  {model_key}")):
        prompt = build_prompt(ex)

        try:
            raw = generate_response(model, tokenizer, model_type, prompt, max_tokens=300)
            pred = parse_response(raw)
        except Exception as e:
            raw = f"ERROR: {e}"
            pred = False

        results.append({
            "example_id": ex["example_id"],
            "stratum": ex["stratum"],
            "gt_conflict": ex.get("conflict_present", False),
            "pred_conflict": pred,
            "raw_response": raw[:500],  # truncate for storage
            "rule_id": ex.get("rule_id", ""),
        })

        # Save checkpoint every 100 examples
        if (i + 1) % 100 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"    Checkpoint saved at {i+1}")

    # Save final results
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Free GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    metrics = compute_metrics(results, model_key)
    return results, metrics


def print_comparison(all_metrics):
    """Print the key results table."""
    print("\n" + "=" * 80)
    print("RESULTS — Per-Stratum Detection Rates (THE PERFORMANCE CLIFF)")
    print("=" * 80)

    header = f"{'Model':<18} {'S0-Spec':<10} {'S1-Cat':<10} {'S2-Num':<10} {'S3-MH':<10} {'Cliff':<12} {'F1':<8}"
    print(header)
    print("-" * len(header))

    for model, metrics in all_metrics.items():
        ps = metrics["per_stratum"]
        cliff = metrics["cliff"]
        o = metrics["overall"]

        s0 = f"{ps.get(0, {}).get('specificity', '—'):.3f}" if 0 in ps else "—"
        s1 = f"{ps.get(1, {}).get('detection_rate', '—'):.3f}" if 1 in ps else "—"
        s2 = f"{ps.get(2, {}).get('detection_rate', '—'):.3f}" if 2 in ps else "—"
        s3 = f"{ps.get(3, {}).get('detection_rate', '—'):.3f}" if 3 in ps else "—"
        drop = f"{cliff.get('absolute_drop', 0):+.3f}" if cliff else "—"
        f1 = f"{o['f1']:.3f}"

        print(f"{model:<18} {s0:<10} {s1:<10} {s2:<10} {s3:<10} {drop:<12} {f1:<8}")

    print("\n  S0=Safe(specificity), S1=Categorical, S2=Numerical, S3=Multi-hop")
    print("  Cliff = S1 detection rate - S2 detection rate (negative = cliff exists)")


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

            # Save metrics
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

        # Save combined
        with open(RESULTS_DIR / "all_models_comparison.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR / 'all_models_comparison.json'}")


if __name__ == "__main__":
    main()