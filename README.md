# MedGuard: Deterministic Verification and Threshold Checking for Clinical LLMs

[![ICON 2026](https://img.shields.io/badge/ICON-2026-blue)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sai Teja Ala, Parinitha Naga Surya Doddi, Sreeja S R**
IIIT Sri City, Andhra Pradesh, India

Accepted at **ICON 2026**, the 22nd International Conference on Natural Language Processing (Gauhati University, Guwahati, 20–23 December 2026), as a poster with a short oral presentation in the Poster Booster session.

- **Paper:** the ACL Anthology link will be added when the ICON 2026 proceedings are published.
- **Data:** MedConflict-Bench and the per-case decision trajectories are distributed through PhysioNet under credentialed access (link will be added once the project is approved).

---

## Overview

Clinical language models are increasingly proposed for prescription review, where the deciding step is often a single comparison: a laboratory value against the cutoff that governs a drug. Metformin is contraindicated below an eGFR of 30 mL/min/1.73m² and needs dose reduction below 45; digoxin toxicity risk rises below a serum potassium of 3.5 mEq/L. The check has an exact deterministic answer. We ask whether models that score well on general medical benchmarks perform it reliably, and how their failures are distributed.

**Finding 1. Benchmark standing does not predict threshold competence.** On numerical-threshold conflicts, frontier generalists sit at or near ceiling (GPT-4o 0.997, Claude 3.5 Sonnet 0.957), while several domain-specialised medical models sit far below (QoQ-Med 0.752, Med-R1 0.348, LLaVA-Med 0.074). That 92-point gap is not anticipated by the benchmark scores these models report.

**Finding 2. At ceiling, the residual failure is specificity.** With the governing cutoff withheld, models that detect conflicts reliably still raise false alarms on 15% to 33% of safe orders, and 83% to 97% of each model's false positives fall on digoxin, spironolactone and lisinopril, all governed by potassium thresholds. Supplying the cutoff raises specificity by 12.2 to 31.7 points: the models substitute a drug-level risk association for a patient-level threshold comparison.

**MedGuard** is the reference behaviour for the encoded part of the task: a deterministic verifier with no language model in the decision path. It reaches macro-F1 0.991 on the test split, reported with its scope: an encoded rule applies to only 20.2% of the 20.3M prescription events in MIMIC-IV.

## What is released, and where

| Artifact | Location | Access |
|---|---|---|
| MedGuard verifier | this repo, `medguard/` | public (MIT) |
| MedConflict-Bench builder | this repo, `bench/` | public; needs your own MIMIC-IV v3.1 access |
| Rule set: 36 templates with FDA-label cutoffs, plus the derivation script | this repo, `rules/` | public |
| Prompts for every evaluated model | this repo, `prompts/` | public |
| Evaluation and analysis scripts | this repo, `eval/`, `analysis/` | public |
| MedConflict-Bench (`medconflict_bench_v2.jsonl`, `splits.json`) | PhysioNet (link coming) | credentialed |
| Per-case decision trajectories | PhysioNet (link coming) | credentialed |
| Knowledge graph (Neo4j) | built locally from your own UMLS release | UMLS license |

**Why the data is not in this repository.** MedConflict-Bench and the trajectories are derived from MIMIC-IV, and the PhysioNet Credentialed Health Data License does not allow sharing that data outside PhysioNet's credentialed access. The knowledge graph contains UMLS Metathesaurus content, which is covered by the NLM UMLS license. This repository therefore holds code, rules and prompts only. The builder regenerates the benchmark from your own MIMIC-IV copy with the same deterministic rules.

## MedConflict-Bench

1,576 prescription–laboratory cases from MIMIC-IV v3.1 (`prescriptions`, `labevents`, `patients`, `diagnoses_icd`). Labels come from deterministic application of the encoded rule set, so the benchmark is exactly reproducible and its scope is exactly statable.

| Stratum | Type | Example | Cases |
|---|---|---|---:|
| S0 | Safe | a rule applies, its threshold is not crossed | 530 |
| S1 | Categorical | penicillin allergy + amoxicillin | 332 |
| S2 | Numerical threshold | metformin + eGFR 28 (28 < 30) | 416 |
| S3 | Multi-hop causal | fluconazole + warfarin + INR 4.2 | 298 |
| | | **Total** | **1,576** |

- **Splits.** Stratified 20/80 into calibration (315 cases, 107 safe) and test (1,261 cases: 838 conflicts, 423 safe).
- **Rules.** 36 templates (12 categorical, 16 numerical, 8 multi-hop) that expand to 27 ingredients. 33 are instantiated; CAT-002, CAT-006 and MH-006 are encoded but no MIMIC-IV case met their preconditions. A rule is included only when an FDA label gives an unambiguous numerical cutoff on a single laboratory value that MIMIC-IV records.
- **Laboratory variables.** eGFR (computed from serum creatinine with the race-free CKD-EPI 2021 equation), creatinine, potassium, INR, ALT, glucose, white cell count, platelets.
- **Pairing.** A drug order and a laboratory value are paired when the measurement falls within 48 hours of the order.
- **Negatives.** A safe case is one where a rule applies and its threshold is not crossed, sampled with at least a 20% relative margin from the threshold.
- **Input format.** Every case is built from structured fields, so the task isolates threshold reasoning from perception.

## How MedGuard works

No language model participates in the decision: no prompts, no fine-tuning, no in-context learning.

1. **Extract.** Deterministic normalisation of structured fields; each drug is mapped to its ingredient (UMLS CUI), and eGFR is computed from creatinine.
2. **Graph lookup.** Parameterised Cypher queries against a Neo4j 5.12 graph (740K UMLS nodes, 1.26M edges) whose `CONTRAINDICATED_WITH`, `INTERACTS_WITH` and `REQUIRES_MONITORING` edges carry their numeric thresholds as properties. The full co-prescribed drug set is passed to the interaction query.
3. **Score.** Conflict Score `C = w1 · PathViol + w2 · NumDist` with `w1 = 0.6075` and `w2 = 0.3925`. PathViol is the fraction of queried paths that are violation edges. NumDist is the largest normalised distance past any applicable cutoff, re-checked against a local threshold table that works independently of the graph.
4. **Decide.** Split conformal calibration on the 107 safe calibration cases gives `τ̂ = 0` at every risk level α, so the decision is `SAFE` when `C = 0` and `ABSTAIN` (refer for human review) otherwise. A matched rule whose governing laboratory value is missing from the 48-hour window also abstains.
5. **Explain.** Retrieval over PubMed abstracts (ChromaDB) produces the clinician-facing explanation only. It does not enter `C`.

MedGuard needs no GPU and no training. It runs on CPU at a mean of 245.6 ms per case on an Intel i7.

## Repository layout

```text
MedGuard/
├── medguard/     # verifier: extraction, graph lookup, NumDist, PathViol, Conflict Score, conformal threshold
├── bench/        # MedConflict-Bench builder (reads your local MIMIC-IV v3.1)
├── rules/        # 36 rule templates with FDA-label cutoffs, plus the derivation script
├── kg/           # builds the Neo4j graph from your UMLS release
├── retrieval/    # ChromaDB index over PubMed abstracts (explanations only)
├── prompts/      # exact prompts for every evaluated model
├── eval/         # model runners and scoring
├── analysis/     # coverage, eICU-CRD transfer, measurement-noise simulation
├── data/         # empty; your credentialed files go here (git-ignored)
├── requirements.txt
└── LICENSE
```

## Setup

You need:

- Credentialed PhysioNet access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/), and to [eICU-CRD](https://physionet.org/content/eicu-crd/) for the external evaluation
- A [UMLS license](https://uts.nlm.nih.gov/uts/signup-login) and a local Metathesaurus release, to build the graph
- [Neo4j](https://neo4j.com/) 5.12
- Python 3 with the packages in `requirements.txt`; a GPU only if you run the open-weight baselines

```bash
git clone https://github.com/Saiteja32006/MedGuard.git
cd MedGuard
pip install -r requirements.txt
```

## Usage

<!-- TODO: make these commands match the actual script names before release -->

```bash
# 1. Build the knowledge graph (Neo4j 5.12 running locally)
python kg/build_graph.py --umls-dir /path/to/umls --neo4j-uri bolt://localhost:7687

# 2. Build MedConflict-Bench from your own MIMIC-IV v3.1 copy
python bench/build_benchmark.py --mimic-dir /path/to/mimic-iv/3.1 --out data/

# 3. Run MedGuard on the test split
python -m medguard.run --bench data/medconflict_bench_v2.jsonl --splits data/splits.json --split test

# 4. Score: per-stratum detection, S0 specificity, macro-F1
python eval/score.py --pred outputs/medguard_test.jsonl
```

**Scoring convention.** Abstention counts as a correct detection when a conflict is present. We report per-stratum detection, specificity on S0, and macro-F1 averaging F1(conflict) and F1(safe).

### Evaluating language models on MIMIC-derived cases

Do not send benchmark cases to an LLM service unless its settings guarantee zero data retention, no use of the data for training and no human review. PhysioNet strongly recommends locally deployed models. See PhysioNet's guidance on [using MIMIC data with LLMs and online services](https://physionet.org/news/post/llm-responsible-use/).

| Paradigm | Models (identifiers) |
|---|---|
| Frontier generalists | GPT-4o (`gpt-4o-2024-05-13`), Claude 3.5 Sonnet (`claude-3-5-sonnet-20240620`) |
| Reasoning-optimised | o4-mini (`o4-mini-2024-07-18`), Med-R1-2B (`yuxianglai117/Med-R1`) |
| Specialised clinical MLLMs | QoQ-Med-7B (`ddvd233/QoQ-Med-VL-7B`), MedGemma-4B (`google/medgemma-4b-it`), LLaVA-Med-7B (`microsoft/llava-med-v1.5-mistral-7b`) |

API models run at temperature 0; open-weight models use 4-bit NF4 quantisation. Prompts are identical across strata, so the only quantity that varies between S1 and S2 is the arithmetic content of the case. The cutoff experiment (governing cutoff withheld vs. supplied, 400 cases) uses GPT-4o, GPT-5.4-mini, GPT-5.6 and o4-mini.

## Results

Per-stratum detection and S0 specificity on the test split. Abstention counts as detection.

| Model | S0 specificity | S1 categorical | S2 numerical | S3 multi-hop | Gap (S1 − S2) |
|---|---:|---:|---:|---:|---:|
| GPT-4o | 0.917 | 0.993 | 0.997 | 1.000 | −0.005 |
| Claude 3.5 Sonnet\* | 0.965 | 0.902 | 0.957 | 1.000 | −0.056 |
| MedGemma-4B | 0.210 | 1.000 | 0.950 | 1.000 | +0.050 |
| o4-mini | 0.983 | 0.944 | 0.847 | 0.996 | +0.097 |
| QoQ-Med-7B | 0.955 | 0.873 | 0.752 | 0.996 | +0.120 |
| Med-R1-2B | 0.655 | 0.509 | 0.348 | 0.672 | +0.161 |
| LLaVA-Med-7B | 0.969 | 0.993 | 0.074 | 0.224 | +0.919 |
| **MedGuard** | **1.000** | **0.996** | **0.973** | **1.000** | +0.023 |

\* Evaluated on a subset of the test split.

MedGuard clears all 423 safe test cases, abstains on 828 of the 838 conflicts and misses 10 (macro-F1 0.991). MedGemma's 0.950 numerical detection comes with 0.210 specificity because it flags nearly every order, which is why specificity is reported alongside detection throughout.

## Reproducing the paper

| Result | Where |
|---|---|
| Table 2 and Figure 2 (model evaluation; cutoff withheld vs. supplied) | `eval/` |
| Table 3 (rule-only checker vs. MedGuard) and Table A1 (normalisation ablation) | `analysis/` |
| Coverage over 20.3M MIMIC-IV prescription events (Section 6.5, Table A4) | `analysis/` |
| eICU-CRD external evaluation (Section 6.6, Appendix F) | `analysis/` |
| Measurement-noise simulation, 2,000 draws (Section 6.7, Table A3) | `analysis/` |

## Scope and intended use

MedGuard is a measurement instrument and a reference implementation, **not a clinical decision support system**. It has not been validated prospectively or against pharmacist judgment, and it must not be used to clear or withhold a prescription for a real patient.

- **`SAFE` does not mean "verified safe".** It means no encoded constraint was violated. An encoded rule applies to only 20.2% of MIMIC-IV prescription events, and only about 17.1% are actually evaluated against a constraint. Any downstream use should show whether an order was checked or simply had no applicable rule.
- **The specificity of 1.000 is largely a property of the benchmark.** No safe case sits near a decision boundary, and there are no categorical negatives, so S1 specificity is untested.
- **The conformal guarantee is narrow.** It bounds the false-alarm rate for a fixed rule set, not missed conflicts, and the threshold is degenerate (`τ̂ = 0`).
- **Silent failure.** In the rare case where neither the graph nor the local threshold table resolves a rule, a conflict can be cleared without abstention (Appendix B).
- **Measurement noise** moves conflicts to safe roughly 190 times more often than the reverse, so it degrades recall, not specificity.
- **Transfer.** The rule set derives from FDA labelling and US critical-care recording conventions; transfer to other settings is not established.

## Citation

```bibtex
@inproceedings{ala-etal-2026-medguard,
    title     = "{M}ed{G}uard: Deterministic Verification and Threshold Checking for Clinical {LLM}s",
    author    = "Ala, Sai Teja and Doddi, Parinitha Naga Surya and S R, Sreeja",
    booktitle = "Proceedings of the 22nd International Conference on Natural Language Processing (ICON)",
    month     = dec,
    year      = "2026",
    address   = "Guwahati, India",
    note      = "To appear"
}
```

If you use the benchmark, please also cite MIMIC-IV (Johnson et al., 2023), and eICU-CRD (Pollard et al., 2018) for the external evaluation.

## License

The code in this repository is released under the [MIT License](LICENSE). The license does not cover MIMIC-IV, eICU-CRD or anything derived from them, which are governed by the PhysioNet Credentialed Health Data License, or UMLS content, which is governed by the NLM UMLS license.

## Contact

Sai Teja Ala, saiteja.a23@iiits.in. Questions and bug reports are welcome as GitHub issues.
