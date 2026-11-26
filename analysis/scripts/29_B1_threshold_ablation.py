#!/usr/bin/env python3
"""
29_B1_threshold_ablation.py — re-evaluate the cliff on current models
======================================================================

Answers Reviewer Sdyd W1(c): "The current models tested are pretty old and
newer models may have already solved the problem."

DESIGN
------
Two prompt conditions over the same cases:

  WITHHELD  the clinical picture only (drug, co-prescribed drug, labs,
            documented conditions) -- the model must supply the cutoff
  GIVEN     the same, plus the governing threshold stated explicitly

The contrast separates two hypotheses for the numerical cliff:
  * the model cannot RECALL the cutoff        -> GIVEN closes the gap
  * the model cannot APPLY the comparison     -> GIVEN does not close it

Table 4 in the submission measured only the WITHHELD condition, so it could
not distinguish these.

PROMPT UNIFORMITY -- THE BUG THIS FIXES
---------------------------------------
An earlier version of this ablation showed the SAME clinical context for every
stratum EXCEPT that categorical cases alone received a "Patient History" block.
That made stratum membership visible in the prompt and produced S1 = 0.000 for
every model. `clinical_context()` below emits one uniform template: every field
that exists is shown, for every stratum. Any field that is absent is simply not
printed.

KNOWN CONFOUND -- STATE IT WHEN REPORTING
-----------------------------------------
`conflict_condition` is populated for every conflict case and "none" for every
safe case in MedConflict-Bench, so a model that flags whenever a condition
appears scores perfectly. This affects the SAFE-vs-CONFLICT comparison. It does
NOT affect the WITHHELD-vs-GIVEN contrast, which is within-case, and that
contrast is the point of this experiment. Report the ablation, not the absolute
detection rate.

COST
----
default --max-per-stratum 100 -> ~400 cases x 2 conditions x N models.
Two models is ~1,600 calls. Raise it only once the numbers look sane.

USAGE
-----
    export OPENAI_API_KEY=sk-...
    python 29_B1_threshold_ablation.py --models gpt-4o,gpt-5.6 --max-per-stratum 100

    # smoke test first, always
    python 29_B1_threshold_ablation.py --models gpt-4o --max-per-stratum 5 --verbose
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

SYSTEM = (
    "You are a clinical pharmacist reviewing a medication order. "
    "Decide whether there is a clinically significant conflict between the "
    "prescribed medication and the patient's clinical data. "
    "Answer with exactly one word: YES or NO."
)


def clinical_context(ex, give_threshold, include_condition=False):
    """
    ONE uniform template for every stratum. Fields that do not exist are not
    printed; no field is gated on stratum. This is the fix for the earlier
    S1 = 0.000 artefact.
    """
    p = ["=== MEDICATION ORDER REVIEW ==="]

    drug = ex.get("drug")
    line = f"Prescribed medication: {drug}"
    if ex.get("dose_val") and ex.get("dose_unit"):
        line += f" {ex['dose_val']} {ex['dose_unit']}"
    if ex.get("route"):
        line += f" ({ex['route']})"
    p.append(line)

    if ex.get("interacting_drug"):
        p.append(f"Co-prescribed medication: {ex['interacting_drug']}")

    # conflict_condition is populated for every conflict and "none" for every
    # safe case, so printing it hands the model the label. OFF by default.
    if include_condition:
        cond = ex.get("conflict_condition")
        if cond and str(cond).lower() not in ("", "none"):
            p.append(f"Documented condition / allergy: {cond}")

    lab, val = ex.get("lab_name"), ex.get("lab_value")
    if lab and val is not None:
        unit = f" {ex['lab_unit']}" if ex.get("lab_unit") else ""
        p.append(f"Relevant lab result: {lab} = {val}{unit}")

    if give_threshold and ex.get("threshold") is not None and ex.get("operator"):
        p.append("")
        p.append(f"Clinical guideline: this medication is contraindicated or "
                 f"requires action when {lab} {ex['operator']} {ex['threshold']}.")

    p.append("=== END OF ORDER ===")
    p.append("Is there a clinically significant conflict? Answer YES or NO.")
    return "\n".join(p)


_NO_SAMPLING = {}


def parse_yesno(text):
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def call(client, model, prompt, retries=4):
    for a in range(retries):
        try:
            kw = dict(model=model,
                      messages=[{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": prompt}])
            if not _NO_SAMPLING.get(model):
                kw["temperature"] = 0.0
                kw["max_tokens"] = 16
            r = client.chat.completions.create(**kw)
            return (r.choices[0].message.content or "").strip(), None
        except Exception as e:
            msg = str(e)
            # Reasoning models reject temperature / max_tokens. Detect rather
            # than guess from the model name, then retry without them.
            if (not _NO_SAMPLING.get(model)
                    and any(k in msg.lower() for k in
                            ("temperature", "max_tokens", "unsupported",
                             "not supported"))):
                _NO_SAMPLING[model] = True
                continue
            if a == retries - 1:
                return "", msg
            time.sleep(2 ** a)
    return "", "unreachable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/raw/medconflict_bench_v3.jsonl")
    ap.add_argument("--out-dir", default="out/B1")
    ap.add_argument("--models", default="gpt-4o",
                    help="comma separated; include an old anchor and a current model")
    ap.add_argument("--max-per-stratum", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--include-condition", action="store_true",
                    help="print conflict_condition in the prompt. OFF by "
                         "default: it leaks the label (1261/1261).")
    ap.add_argument("--strata", default="0,2,3",
                    help="strata to evaluate. S1 excluded by default: a "
                         "drug-allergy case cannot be judged without the "
                         "allergy, which is the leaked field.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.bench, encoding="utf-8")]
    rows = [e for e in rows if e.get("split") == args.split]

    keep = {int(x) for x in args.strata.split(",") if x.strip()}
    rows = [e for e in rows if e.get("stratum") in keep]

    rng = random.Random(args.seed)
    by_s = defaultdict(list)
    for e in rows:
        by_s[e.get("stratum")].append(e)
    cases = []
    for s in sorted(by_s):
        pool = by_s[s]
        rng.shuffle(pool)
        cases.extend(pool[:args.max_per_stratum])

    print("=" * 74)
    print("B1 — THRESHOLD ABLATION ON CURRENT MODELS")
    print("=" * 74)
    print(f"  cases: {len(cases)}  "
          f"({ {s: min(len(by_s[s]), args.max_per_stratum) for s in sorted(by_s)} })")
    print(f"  models: {args.models}")
    print(f"  strata: {sorted(keep)}"
          f"{'   (S1 excluded: unanswerable without the leaked field)' if 1 not in keep else ''}")
    print(f"  conflict_condition in prompt: {args.include_condition}"
          f"{'   <-- LEAKS THE LABEL' if args.include_condition else ''}")
    print(f"  calls: {len(cases) * 2 * len(args.models.split(','))}")
    print()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=args.base_url) if args.base_url else \
        OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    summary = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for cond, give in (("withheld", False), ("given", True)):
            ck = out / f"B1_{model.replace('/', '_')}_{cond}.jsonl"
            done = {}
            if ck.exists():
                for l in open(ck, encoding="utf-8"):
                    r = json.loads(l)
                    done[r["example_id"]] = r
                print(f"  [resume] {ck.name}: {len(done)} done")

            todo = [e for e in cases if e["example_id"] not in done]
            print(f"  {model} / {cond}: {len(todo)} to run")

            with open(ck, "a", encoding="utf-8") as f:
                for i, e in enumerate(todo):
                    prompt = clinical_context(e, give,
                                              args.include_condition)
                    raw, err = call(client, model, prompt)
                    pred = parse_yesno(raw)
                    rec = {"example_id": e["example_id"],
                           "stratum": e.get("stratum"),
                           "gt": bool(e.get("conflict_present")),
                           "pred": pred, "raw": raw, "error": err}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[e["example_id"]] = rec
                    if args.verbose and (i < 2 or
                                         (i and e.get("stratum") !=
                                          todo[i-1].get("stratum"))):
                        print("-" * 60)
                        print(prompt)
                        print(f"--> {raw!r}  parsed={pred}  gt={rec['gt']}")
                    if (i + 1) % 50 == 0:
                        print(f"    ...{i+1}/{len(todo)}")

            recs = list(done.values())
            n_err = sum(1 for r in recs if r.get("error"))
            n_unp = sum(1 for r in recs if r.get("pred") is None
                        and not r.get("error"))
            per = {}
            for s in sorted({r["stratum"] for r in recs}):
                sub = [r for r in recs if r["stratum"] == s]
                if s == 0:
                    per[s] = round(sum(1 for r in sub if r["pred"] is False)
                                   / max(len(sub), 1), 4)
                else:
                    per[s] = round(sum(1 for r in sub if r["pred"] is True)
                                   / max(len(sub), 1), 4)
            cliff = (round(per.get(1, 0) - per.get(2, 0), 4)
                     if 1 in per and 2 in per else None)
            _ = cliff
            summary[f"{model}|{cond}"] = {
                "per_stratum": {str(k): v for k, v in per.items()},
                "cliff_S1_minus_S2": cliff,
                "n": len(recs), "errors": n_err, "unparsed": n_unp}
            print(f"    S0 {per.get(0)}  S1 {per.get(1)}  S2 {per.get(2)}  "
                  f"S3 {per.get(3)}  cliff {cliff}  err {n_err}  unparsed {n_unp}")
            print()

    with open(out / "B1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  {'model / condition':<30}{'S0 spec':>9}{'S2':>8}{'S3':>8}")
    for k, v in summary.items():
        p = v["per_stratum"]
        print(f"  {k:<30}{p.get('0','-'):>9}{p.get('2','-'):>8}{p.get('3','-'):>8}")
    print()
    print("  THE RESULT IS THE DELTA, PER MODEL:")
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        w = summary.get(f"{model}|withheld", {}).get("per_stratum", {})
        g = summary.get(f"{model}|given", {}).get("per_stratum", {})
        if "2" in w and "2" in g:
            print(f"    {model:<16} S2 withheld {w['2']:.3f} -> given "
                  f"{g['2']:.3f}   delta {g['2']-w['2']:+.3f}")
    print()
    print("   large positive delta = the model could not RECALL the cutoff")
    print("   delta near zero      = the model could not APPLY the comparison")
    print("                          (or already knew the cutoff)")
    print("   errors > 0           = failures counted as NO; fix before reporting")
    print()
    print("  S1 is excluded by default and the S1-vs-S2 cliff is NOT reported:")
    print("  a drug-allergy case cannot be judged without the allergy, and that")
    print("  field is exactly the one that leaks the label (1261/1261).")
    print(f"  [saved] {out / 'B1_summary.json'}")


if __name__ == "__main__":
    main()