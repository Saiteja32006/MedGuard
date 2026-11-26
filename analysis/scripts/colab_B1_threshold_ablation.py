#!/usr/bin/env python3
"""
colab_B1_threshold_ablation.py — RUN ON COLAB, DAY 1, FIRST THING.

THE most important experiment. Same 1,261 test cases, two prompt conditions:
  (a) THRESHOLD GIVEN    -> tests pure arithmetic
  (b) THRESHOLD WITHHELD -> tests knowledge recall + arithmetic

If the cliff only exists in (b), it is a RECALL gap, not a numeracy gap,
and MedGuard was handed the rule table the baselines had to remember.
You must know this before writing the rebuttal.

SETUP
  from google.colab import drive; drive.mount('/content/drive')
  !pip install openai anthropic -q
  Set OPENAI_API_KEY / ANTHROPIC_API_KEY below.

Checkpoints every 25 calls to Drive, so a disconnect costs almost nothing.
"""

import json
import os
import time

DRIVE = os.getenv("DATA_ROOT", ".")
BENCH = f"{DRIVE}/MedConflict-Bench/medconflict_bench_v2.jsonl"
OUT = f"{DRIVE}/Rebuttal_B1"
os.makedirs(OUT, exist_ok=True)

OPENAI_API_KEY = ""      # <-- fill in
MODELS = ["gpt-4o", "o4-mini"]      # add newer models for B2
N_PER_STRATUM = 120                 # 120 x 4 strata = 480 cases per condition


def prompt_given(ex):
    """(a) threshold stated in the prompt — pure arithmetic test."""
    return (
        f"A patient is prescribed {ex['drug']}.\n"
        f"Laboratory result: {ex['lab_name']} = {ex['lab_value']} {ex.get('lab_unit','')}\n"
        + (f"The patient is also taking {ex['interacting_drug']}.\n"
           if ex.get("interacting_drug") else "")
        + f"\nCLINICAL RULE: {ex['drug']} is contraindicated when "
          f"{ex['lab_name']} {ex.get('operator','<')} {ex.get('threshold')}.\n\n"
        "Given this rule and the patient's value, is there a conflict? "
        "Answer with exactly one word: YES or NO."
    )


def prompt_withheld(ex):
    """(b) threshold withheld — recall + arithmetic, as in the submission."""
    return (
        f"A patient is prescribed {ex['drug']}.\n"
        f"Laboratory result: {ex['lab_name']} = {ex['lab_value']} {ex.get('lab_unit','')}\n"
        + (f"The patient is also taking {ex['interacting_drug']}.\n"
           if ex.get("interacting_drug") else "")
        + "\nIs there a clinically significant conflict between this "
          "prescription and the laboratory result? "
          "Answer with exactly one word: YES or NO."
    )


def call_openai(model, prompt):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith("o"):
        kw["temperature"] = 0
    r = client.chat.completions.create(**kw)
    return r.choices[0].message.content.strip()


def sample_cases():
    rows = [json.loads(l) for l in open(BENCH)]
    test = [r for r in rows if r.get("split") == "test"]
    out = []
    for s in (0, 1, 2, 3):
        sub = [r for r in test if r.get("stratum") == s][:N_PER_STRATUM]
        out.extend(sub)
    print(f"sampled {len(out)} cases")
    return out


def run(model, condition, cases):
    fn = f"{OUT}/B1_{model}_{condition}.jsonl"
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)["example_id"] for l in open(fn)}
        print(f"  resuming: {len(done)} already done")
    pf = prompt_given if condition == "given" else prompt_withheld
    with open(fn, "a") as f:
        for i, ex in enumerate(cases):
            if ex["example_id"] in done:
                continue
            try:
                txt = call_openai(model, pf(ex))
            except Exception as e:
                print(f"    error {ex['example_id']}: {e}")
                time.sleep(5)
                continue
            pred = "yes" in txt.lower()[:12]
            f.write(json.dumps({
                "example_id": ex["example_id"], "stratum": ex["stratum"],
                "conflict_present": ex["conflict_present"],
                "prediction": pred, "raw": txt[:300],
            }) + "\n")
            if (i + 1) % 25 == 0:
                f.flush()
                print(f"    [{i+1}/{len(cases)}]")
    return fn


def score(fn, label):
    recs = [json.loads(l) for l in open(fn)]
    per = {}
    for s in (0, 1, 2, 3):
        sub = [r for r in recs if r["stratum"] == s]
        if not sub:
            continue
        if s == 0:
            per[s] = 1 - sum(r["prediction"] for r in sub) / len(sub)
        else:
            per[s] = sum(r["prediction"] for r in sub) / len(sub)
    cliff = per.get(1, 0) - per.get(2, 0)
    print(f"  {label:34s} " + "  ".join(f"S{s}={v:.3f}" for s, v in per.items())
          + f"   CLIFF={cliff:+.3f}")
    return {"per_stratum": per, "cliff": cliff}


def main():
    cases = sample_cases()
    summary = {}
    for model in MODELS:
        for cond in ("withheld", "given"):
            print(f"\n{model} / threshold {cond}")
            fn = run(model, cond, cases)
            summary[f"{model}_{cond}"] = score(fn, f"{model} [{cond}]")
    with open(f"{OUT}/B1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("  cliff(withheld) large, cliff(given) large  -> genuine numeracy")
    print("     failure. Your claim is validated and much stronger. Report both.")
    print("  cliff(withheld) large, cliff(given) ~0     -> the cliff is THRESHOLD")
    print("     RECALL, not arithmetic. The paper must be reframed. Better to")
    print("     learn this now than at camera-ready.")
    print("  both small -> current models have closed the gap; report honestly.")


if __name__ == "__main__":
    main()
