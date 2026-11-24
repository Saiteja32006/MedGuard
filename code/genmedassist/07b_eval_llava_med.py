#!/usr/bin/env python3
r"""
07b_eval_llava_med.py — LLaVA-Med Evaluation on MedConflict-Bench
==================================================================
Uses the HF-compatible conversion (chaoyinshe/llava-med-v1.5-mistral-7b-hf)
which works with standard transformers LlavaForConditionalGeneration.

The original microsoft/llava-med-v1.5-mistral-7b requires their custom
LLaVA codebase; this conversion works out-of-the-box with transformers.

Run on LAB MACHINE:
    cd C:\Users\AIML\Sai_Teja _Honors\genmedassist
    python 07b_eval_llava_med.py

Prerequisites:
    pip install transformers>=4.46.0 accelerate bitsandbytes Pillow tqdm scikit-learn
"""

import json, os, sys, re, gc, time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ── Config ─────────────────────────────────────────────────────────────────
BENCH_PATH  = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "medconflict_bench_v2.jsonl"
RESULTS_DIR = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HF-compatible conversion — works with standard transformers
MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
MODEL_KEY = "llava-med"

SYSTEM_PROMPT = """You are a clinical pharmacist reviewing a patient's medication order.
Determine if there is a clinically significant conflict between the prescribed medication and the patient's clinical data.

A conflict can be:
1. A drug prescribed to a patient with a documented allergy or contraindicated condition
2. A drug prescribed when a lab value exceeds a safety threshold
3. A drug-drug interaction that becomes dangerous given the patient's lab values

Respond with EXACTLY this JSON format and nothing else:
{"conflict_detected": true/false, "severity": "none"|"moderate"|"major", "explanation": "brief reasoning"}"""


def build_prompt(example):
    """Build the clinical review prompt from benchmark example."""
    parts = ["=== MEDICATION ORDER REVIEW ===\n"]

    drug = example.get("drug", "Unknown")
    dose = example.get("dose_val", "")
    unit = example.get("dose_unit", "")
    route = example.get("route", "")

    drug_line = f"Prescribed: {drug}"
    if dose and unit and str(dose) != "nan" and str(unit) != "nan":
        drug_line += f" {dose} {unit}"
    if route and str(route) != "nan":
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
    safe_signals = ["no conflict", "safe to proceed", "no clinically significant",
                    "\"false\"", "no significant", "conflict_detected\": false"]
    for s in safe_signals:
        if s in text:
            return False

    conflict_signals = ["conflict_detected\": true", "conflict detected",
                        "yes, there is a conflict", "significant conflict",
                        "contraindicated", "should be held", "should not be",
                        "risk of", "dangerous", "\"true\""]
    for s in conflict_signals:
        if s in text:
            return True

    danger_words = ["contraindicated", "hold", "discontinue", "risk", "toxicity", "avoid"]
    return any(w in text for w in danger_words)


def load_llava_med():
    """Load LLaVA-Med using HF-compatible checkpoint."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    print(f"Loading LLaVA-Med from {MODEL_ID}...")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    print("  Model loaded successfully!")
    return model, processor


def generate_response(model, processor, prompt, max_tokens=300):
    """Generate response from LLaVA-Med.
    
    LLaVA always expects an image input. For text-only evaluation,
    we pass a minimal white dummy image.
    """
    from PIL import Image

    # LLaVA requires an image — use a tiny blank image
    dummy_image = Image.new("RGB", (64, 64), color=(255, 255, 255))

    # Build chat using the Mistral [INST] template
    full_prompt = f"[INST] <image>\n{SYSTEM_PROMPT}\n\n{prompt} [/INST]"

    inputs = processor(
        text=full_prompt,
        images=dummy_image,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device, torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


def compute_metrics(results):
    """Compute all metrics including per-stratum and performance cliff."""
    y_true = np.array([1 if r["gt_conflict"] else 0 for r in results])
    y_pred = np.array([1 if r["pred_conflict"] else 0 for r in results])

    overall = {
        "model": MODEL_KEY,
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
        m = {"name": name, "n": int(mask.sum()),
             "accuracy": round((s_true == s_pred).mean(), 4)}
        if s == 0:
            m["specificity"] = round(1 - s_pred.mean(), 4)
            m["false_positive_rate"] = round(s_pred.mean(), 4)
        else:
            m["detection_rate"] = round(s_pred.mean(), 4)
            m["recall"] = round(recall_score(s_true, s_pred, zero_division=0), 4)
            m["f1"] = round(f1_score(s_true, s_pred, zero_division=0), 4)
        per_stratum[s] = m

    cliff = {}
    if 1 in per_stratum and 2 in per_stratum:
        s1 = per_stratum[1].get("detection_rate", 0)
        s2 = per_stratum[2].get("detection_rate", 0)
        cliff = {
            "s1_categorical": s1,
            "s2_numerical": s2,
            "absolute_drop": round(s1 - s2, 4),
            "relative_drop_pct": round((s1 - s2) / max(s1, 0.001) * 100, 1),
        }
    if 3 in per_stratum:
        cliff["s3_multihop"] = per_stratum[3].get("detection_rate", 0)

    return {"overall": overall, "per_stratum": per_stratum, "cliff": cliff}


def main():
    print("=" * 60)
    print("Evaluating: LLaVA-Med-v1.5-Mistral-7B")
    print("=" * 60)

    ckpt_path = RESULTS_DIR / f"{MODEL_KEY}_results.jsonl"

    # Check if already done
    if ckpt_path.exists():
        existing = [json.loads(l) for l in open(ckpt_path, encoding="utf-8")]
        # Check if it's a full run (not partial)
        if len(existing) >= 1200:
            print(f"\n  Already evaluated ({len(existing)} results). Computing metrics...")
            metrics = compute_metrics(existing)
            print(f"\n  Results:")
            for s, data in sorted(metrics["per_stratum"].items()):
                rate_key = "detection_rate" if s > 0 else "specificity"
                print(f"    S{s} ({data['name']}): {rate_key}={data.get(rate_key, '?'):.3f} (n={data['n']})")
            if metrics["cliff"]:
                print(f"    Cliff (S1→S2): {metrics['cliff'].get('absolute_drop', 0):+.3f}")
            print(f"    Overall F1: {metrics['overall']['f1']:.4f}")
            with open(RESULTS_DIR / f"{MODEL_KEY}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            return

    # Load benchmark
    examples = [json.loads(l) for l in open(BENCH_PATH, encoding="utf-8")]
    test_examples = [e for e in examples if e.get("split") == "test"]
    print(f"  Benchmark: {len(test_examples)} test examples")

    # Resume from checkpoint
    done_ids = set()
    results = []
    if ckpt_path.exists():
        results = [json.loads(l) for l in open(ckpt_path, encoding="utf-8")]
        done_ids = {r["example_id"] for r in results}
        print(f"  Resuming: {len(done_ids)} already done")

    remaining = [e for e in test_examples if e.get("example_id") not in done_ids]
    print(f"  To process: {len(remaining)}")

    if not remaining:
        print("  All done!")
        metrics = compute_metrics(results)
        with open(RESULTS_DIR / f"{MODEL_KEY}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # Load model
    model, processor = load_llava_med()

    # Evaluate
    with open(ckpt_path, "a", encoding="utf-8") as fout:
        for i, ex in enumerate(tqdm(remaining, desc=f"  {MODEL_KEY}")):
            prompt = build_prompt(ex)

            try:
                raw = generate_response(model, processor, prompt)
                pred = parse_response(raw)
            except Exception as e:
                raw = f"ERROR: {str(e)[:200]}"
                pred = False

            result = {
                "example_id": ex["example_id"],
                "stratum": ex["stratum"],
                "gt_conflict": ex.get("conflict_present", False),
                "pred_conflict": pred,
                "raw_response": raw[:500],
                "rule_id": ex.get("rule_id", ""),
            }
            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % 100 == 0:
                # Print running metrics
                so_far = results
                for s in [1, 2, 3]:
                    s_res = [r for r in so_far if r["stratum"] == s]
                    if s_res:
                        det = sum(1 for r in s_res if r["pred_conflict"]) / len(s_res)
                        print(f"      S{s}: {det:.3f} ({len(s_res)} done)")

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    # Final metrics
    metrics = compute_metrics(results)

    print(f"\n{'=' * 60}")
    print(f"LLaVA-Med RESULTS")
    print(f"{'=' * 60}")
    for s, data in sorted(metrics["per_stratum"].items()):
        rate_key = "detection_rate" if s > 0 else "specificity"
        print(f"  S{s} ({data['name']}): {rate_key}={data.get(rate_key, '?'):.3f} (n={data['n']})")
    if metrics["cliff"]:
        print(f"  Cliff (S1→S2): {metrics['cliff'].get('absolute_drop', 0):+.3f}")
    print(f"  Overall F1: {metrics['overall']['f1']:.4f}")
    print(f"  Macro-F1:   {metrics['overall']['macro_f1']:.4f}")

    with open(RESULTS_DIR / f"{MODEL_KEY}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved: {ckpt_path}")
    print(f"  Saved: {RESULTS_DIR / f'{MODEL_KEY}_metrics.json'}")


if __name__ == "__main__":
    main()