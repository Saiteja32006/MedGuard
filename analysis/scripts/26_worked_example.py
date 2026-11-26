#!/usr/bin/env python3
"""
26_worked_example.py — render one case end to end
==================================================

Answers Reviewer Sdyd:
  Q1  "Show a detailed example of input, processing components, prompts,
       and LLM outputs for a representative example."
  W3  "Is it a fine tuning step or is it an in context learning step?"
  W5  "How is the input to the path violation and numerical threshold
       distance computation extracted from the data?"

All three are the same question asked three ways: what actually happens to
one case. This renders it from the recorded trajectory -- no reconstruction,
no hand-written illustration.

Writes a markdown file suitable for pasting into the response or appendix.

Usage:
    python scripts/26_worked_example.py --id c50a1ce39ba1
    python scripts/26_worked_example.py --id c50a1ce39ba1 --out out/worked_example.md
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "out"

W1, W2 = 0.6075, 0.3925
TAU_V1 = 0.0028


def fmt_step(step):
    return step.get("name", "?"), step


def render(eid, bench, traj):
    b = bench[eid]
    t = traj[eid]
    steps = {s.get("name"): s for s in t.get("steps", [])}

    L = []
    A = L.append

    A(f"# Worked example — `{eid}`")
    A("")
    A("Rendered directly from the released trajectory file; every field below "
      "appears verbatim in the JSONL record for this case.")
    A("")

    # ---------------------------------------------------------------- INPUT
    A("## 1. Input")
    A("")
    A("The benchmark case as constructed from MIMIC-IV:")
    A("")
    A("| Field | Source table | Value |")
    A("| --- | --- | --- |")
    A(f"| Prescribed drug | `prescriptions.drug` | {b.get('drug')} |")
    if b.get("interacting_drug"):
        A(f"| Co-prescribed drug | `prescriptions.drug` | {b.get('interacting_drug')} |")
    A(f"| Lab test | `d_labitems.label` | {b.get('lab_name')} |")
    A(f"| Lab value | `labevents.valuenum` | {b.get('lab_value')} |")
    if b.get("conflict_condition") and b.get("conflict_condition") != "none":
        A(f"| Condition | rule metadata | {b.get('conflict_condition')} |")
    A(f"| Stratum | — | {b.get('stratum')} |")
    A(f"| Rule | — | {b.get('rule_id')} ({b.get('operator')}{b.get('threshold')}) |")
    A(f"| Ground truth | — | conflict = {b.get('conflict_present')} |")
    A("")
    A("**Note on modality.** These fields arrive already structured from "
      "MIMIC-IV. The benchmark does not exercise perception: no image, no "
      "handwritten script, no free text is parsed. This is a real limitation "
      "of the benchmark and we state it explicitly rather than implying the "
      "extraction step is doing work it is not.")
    A("")
    A("**Note on `conflict_condition`.** In the originally released benchmark "
      "this field was populated only on conflict cases and set to `none` on "
      "safe ones, so a classifier reading it alone separated the two "
      "perfectly (1261/1261 on the test split). We found this during our "
      "audit and corrected it: safe cases now carry the condition their rule "
      "concerns, since a genuine negative is one where the rule applies and "
      "the threshold is not crossed. The field is now uninformative — the "
      "same one-line classifier drops to the base rate (838/1261 = 0.665). "
      "As the trace below shows, the field is read into the extraction "
      "output and the retrieval query; the two terms of the conflict score "
      "are computed from the drug, lab and threshold, and we verified "
      "separately that graph behaviour does not vary with it.")
    A("")

    # ----------------------------------------------------------- EXTRACTION
    A("## 2. Entity extraction")
    A("")
    s = steps.get("entity_extraction", {})
    o = s.get("output", {})
    A("```json")
    A(json.dumps(o, indent=2))
    A("```")
    A("")
    A("This step normalises drug strings and reads the lab name, value and "
      "unit. It is **not** a fine-tuning step and **not** in-context "
      "learning: no model weights are updated and no exemplars are placed in "
      "a prompt. It is a deterministic normalisation that produces the typed "
      "arguments consumed by the next two steps. (Answers W3 and W5.)")
    A("")

    # -------------------------------------------------------------- GRAPHRAG
    A("## 3. Knowledge-graph lookup")
    A("")
    kg = steps.get("graphrag", {})
    A(f"- Cypher queries issued: **{kg.get('queries_count')}**")
    A(f"- Paths checked: **{kg.get('paths_checked')}**")
    A(f"- Paths violated: **{kg.get('paths_violated')}**")
    A("")
    viols = kg.get("violations", [])
    if viols:
        A("Violations returned by the graph:")
        A("")
        A("```json")
        A(json.dumps(viols, indent=2))
        A("```")
    else:
        A("No violating paths returned.")
    A("")
    A("The graph is queried with the normalised drug names as parameters. "
      "The LLM does not author these queries and does not see the graph; the "
      "traversal is executed directly and returns structured records.")
    A("")

    # -------------------------------------------------------------- VECTOR
    A("## 4. Evidence retrieval")
    A("")
    vr = steps.get("vectorrag", {})
    A(f"- Query: `{vr.get('query')}`")
    A(f"- Passages retrieved: **{vr.get('evidence_count')}**")
    A(f"- Mean relevance: **{vr.get('avg_relevance')}**")
    A("")
    ev = vr.get("evidence", [])[:3]
    if ev:
        A("| PMID | Title | Relevance |")
        A("| --- | --- | --- |")
        for e in ev:
            title = str(e.get("title", ""))[:70]
            A(f"| {e.get('pmid')} | {title} | {e.get('relevance')} |")
        A("")
    A("Retrieved evidence supports the natural-language justification shown "
      "to the clinician. **It does not enter the conflict score**, which is "
      "computed in the next step from the graph and threshold results alone. "
      "Retrieval quality is uneven — passages are ranked by embedding "
      "similarity over PubMed abstracts and are sometimes only loosely "
      "on-topic — but because retrieval is downstream of the decision, this "
      "affects the explanation shown to the user rather than the decision "
      "itself.")
    A("")

    # -------------------------------------------------------------- CSCORE
    A("## 5. Conflict score")
    A("")
    cs = steps.get("cscore_computation", {})
    pv = cs.get("path_viol", 0.0)
    nd = cs.get("num_dist", 0.0)
    w1 = cs.get("w1", W1)
    w2 = cs.get("w2", W2)
    c = cs.get("cscore", 0.0)
    ndd = cs.get("num_dist_details", {})

    A(f"**PathViol** = paths_violated / paths_checked = "
      f"{kg.get('paths_violated')} / {kg.get('paths_checked')} = **{pv}**")
    A("")
    if ndd:
        A(f"**NumDist** = ReLU((τ − v) / τ) with τ = {ndd.get('threshold')}, "
          f"v = {ndd.get('actual')} → **{nd}**")
        A(f"  matched rule: `{ndd.get('matched_rule')}` on {ndd.get('lab')}")
    else:
        A(f"**NumDist** = **{nd}**")
    A("")
    A(f"**C** = w₁·PathViol + w₂·NumDist = "
      f"{w1}·{pv} + {w2}·{nd} = **{c}**")
    A("")
    A(f"Decision: **C = {c} > τ̂ = {TAU_V1} → "
      f"{t.get('decision')}**")
    A("")
    A("Both terms are computed by deterministic functions over the extracted "
      "entities. No generative model participates in the score or the "
      "decision; the pipeline\'s five recorded steps are entity extraction, "
      "graph lookup, embedding-based retrieval, score computation and "
      "thresholding, and none of them invokes a language model. (Answers W2: "
      "the abstention decision is not made by an LLM and does not depend on "
      "one. MLLMs appear in this work as the systems we measure, not as "
      "components of the method.)")
    A("")

    dec = steps.get("decision")
    if dec:
        A("The recorded decision step:")
        A("")
        A("```json")
        A(json.dumps(dec, indent=2))
        A("```")
        A("")

    # -------------------------------------------------------------- OUTCOME
    A("## 6. Outcome")
    A("")
    A("| | |")
    A("| --- | --- |")
    A(f"| C-score | {t.get('cscore')} |")
    A(f"| Decision | {t.get('decision')} |")
    A(f"| Ground truth | conflict = {(t.get('ground_truth') or {}).get('conflict_present')} |")
    A(f"| Correct? | {'yes' if (t.get('decision') == 'ABSTAIN') == bool((t.get('ground_truth') or {}).get('conflict_present')) else 'no'} |")
    A(f"| Wall time | {t.get('processing_ms')} ms |")
    A("")
    A("Under our scoring convention ABSTAIN counts as a positive detection: "
      "the system correctly refers this order for human review. (Answers W7.)")
    A("")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--bench", default=str(DATA / "medconflict_bench_v3.jsonl"))
    ap.add_argument("--traj", default=str(DATA / "genmedassist_trajectories_v2.jsonl"))
    ap.add_argument("--out", default=str(OUT / "worked_example.md"))
    args = ap.parse_args()

    bench = {json.loads(l)["example_id"]: json.loads(l)
             for l in open(args.bench, encoding="utf-8")}
    traj = {json.loads(l)["example_id"]: json.loads(l)
            for l in open(args.traj, encoding="utf-8")}

    if args.id not in bench:
        raise SystemExit(f"{args.id} not in benchmark")
    if args.id not in traj:
        raise SystemExit(f"{args.id} not in trajectories")

    md = render(args.id, bench, traj)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print()
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()