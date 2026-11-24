#!/usr/bin/env python3
"""
20_coverage_denominator.py — MedGuard rule coverage over all of MIMIC-IV
=========================================================================

Answers Reviewer oJaU's requested Experiment 1.

IMPORTANT FRAMING
-----------------
MIMIC-IV contains NO prescribing-error labels. "What % of prescribing errors
would MedGuard catch" is therefore NOT computable and must not be claimed.
What IS computable, and what the reviewer is actually asking for, is the
COVERAGE DENOMINATOR:

  A  APPLICABILITY   % of prescription events whose drug matches an encoded rule
  B  EVALUABLE       of A, % where the required lab exists within the window
  C  SILENT PASS     % of ALL prescriptions MedGuard clears having checked nothing
  D  HAZARD PROXY    coverage among prescriptions co-occurring with an
                     abnormal-flagged lab within the window

C is the headline number. It quantifies "the abstention is also its alibi."

The rule set is DERIVED from your own artifacts (benchmark + threshold script),
not hardcoded, so this script also settles the 19-vs-36 rule-count discrepancy
empirically. Read the STAGE 0 output for that.

WHERE TO RUN
------------
Colab (MIMIC-IV lives in Drive). Needs no Neo4j, no GPU, no API keys.

    from google.colab import drive; drive.mount('/content/drive')
    !pip install pandas pyarrow -q
    !python 20_coverage_denominator.py --stage all --quick

Run --quick first (samples hadm_ids, ~5-10 min, preliminary number).
Then run without --quick for the reportable number (~45-90 min, mostly stage 2).

Every stage checkpoints to Drive. Re-running skips completed stages, so a
runtime disconnect costs you only the stage that was in flight.

Author: Ala Sai Teja, IIIT Sricity — NeurIPS 2026 rebuttal
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================

DRIVE = Path(os.getenv("DATA_ROOT", "."))
MIMIC = DRIVE / "Dataset" / "mimic-iv-hosp"
BENCH = DRIVE / "MedConflict-Bench"
OUT = DRIVE / "Rebuttal_Coverage"

LAB_WINDOW_HOURS = 48          # from splits_v2.json — your stated pairing rule
QUICK_N_HADM = 20000           # hadm_ids sampled in --quick mode
CHUNK = 500_000
SEED = 42

# MIMIC-IV lab itemids for the labs the rule set references.
# Source: handoff doc section 5 + d_labitems.
# Values may be an int or a list of ints. VERIFIED against d_labitems +
# observed volume (22_coverage_diagnostics.py check 1).
LAB_ITEMIDS = {
    "creatinine": 50912,       # eGFR is COMPUTED from this (CKD-EPI 2021)
    "potassium": 50971,
    "inr": 51237,
    "alt": 50861,
    "glucose": 50931,
    "platelet count": 51265,
    # 51301 "White Blood Cells" (Blood/Hematology) carries ~146x the volume of
    # 51300 "WBC Count". NOT 51516 -- that is URINE WBC and is unrelated to
    # myelosuppression.
    "wbc": [51301, 51300],
    "tsh": 50993,
}


def _itemids(v):
    return v if isinstance(v, (list, tuple, set)) else [v]

# Labs referenced by rules that MIMIC-IV hosp CANNOT supply at all.
# QTc comes from ECG waveforms, not labevents. This is a real, reportable gap.
LAB_UNAVAILABLE = {"qtc interval", "qtc"}

# Map a rule's lab_name onto the underlying MIMIC lab that supplies it.
LAB_ALIAS = {
    "egfr": "creatinine",      # computed, not measured
    "k+": "potassium",
    "potassium": "potassium",
    "inr": "inr",
    "inr(pt)": "inr",
    "alt": "alt",
    "creatinine": "creatinine",
    "glucose": "glucose",
    "platelet count": "platelet count",
    "platelets": "platelet count",
    "wbc": "wbc",
    "tsh": "tsh",
}

# ============================================================================
# DRUG NORMALIZATION
# ============================================================================

_FORM_TOKENS = r"""
tablet tablets tab tabs capsule capsules cap caps injection inj solution soln
oral iv ivpb po sc subcutaneous intravenous syringe premix bag vial ampule amp
suspension syrup elixir er xr sr la cr dr odt chewable delayed extended release
neb nebulizer inhaler patch cream ointment drops ophthalmic topical
in dextrose in nacl sodium chloride dextrose water sterile
mg mcg ug gm gram grams ml unit units meq mmol percent
immediate release preservative free human regular novolin humulin
concentrate concentrated pediatric adult bolus infusion drip
flush lock dwell prime priming
""".split()

_SALTS = """
hydrochloride hcl sodium potassium calcium magnesium sulfate sulphate succinate
tartrate bitartrate besylate maleate mesylate citrate acetate phosphate fumarate
gluconate lactate carbonate bromide chloride nitrate oxalate pamoate
""".split()

# Brand -> generic, restricted to drugs the rule set actually references.
BRAND2GENERIC = {
    "glucophage": "metformin", "fortamet": "metformin", "riomet": "metformin",
    "glumetza": "metformin",
    "lanoxin": "digoxin", "digitek": "digoxin",
    "coumadin": "warfarin", "jantoven": "warfarin",
    "aldactone": "spironolactone", "carospir": "spironolactone",
    "diflucan": "fluconazole",
    "neurontin": "gabapentin", "gralise": "gabapentin", "horizant": "gabapentin",
    "pradaxa": "dabigatran",
    "xarelto": "rivaroxaban",
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine",
    "unithroid": "levothyroxine", "euthyrox": "levothyroxine",
    "cordarone": "amiodarone", "pacerone": "amiodarone", "nexterone": "amiodarone",
    "tylenol": "acetaminophen", "ofirmev": "acetaminophen",
    "trexall": "methotrexate", "rheumatrex": "methotrexate", "otrexup": "methotrexate",
    "lovenox": "enoxaparin",
    "lipitor": "atorvastatin", "zocor": "simvastatin", "crestor": "rosuvastatin",
    "pravachol": "pravastatin", "mevacor": "lovastatin", "lescol": "fluvastatin",
    "livalo": "pitavastatin",
    "prinivil": "lisinopril", "zestril": "lisinopril", "vasotec": "enalapril",
    "altace": "ramipril", "capoten": "captopril", "lotensin": "benazepril",
    "accupril": "quinapril", "monopril": "fosinopril", "aceon": "perindopril",
    "glucotrol": "glipizide", "diabeta": "glyburide", "micronase": "glyburide",
    "glynase": "glyburide", "amaryl": "glimepiride", "diamicron": "gliclazide",
}

# Class rules ("Statins", "ACE Inhibitor", ...) expand to member ingredients.
DRUG_CLASSES = {
    "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
               "lovastatin", "fluvastatin", "pitavastatin"],
    "statins": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
                "lovastatin", "fluvastatin", "pitavastatin"],
    "ace inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril",
                      "benazepril", "quinapril", "fosinopril", "perindopril",
                      "moexipril", "trandolapril"],
    "ace inhibitors": ["lisinopril", "enalapril", "ramipril", "captopril",
                       "benazepril", "quinapril", "fosinopril", "perindopril",
                       "moexipril", "trandolapril"],
    "sulfonylurea": ["glipizide", "glyburide", "glimepiride", "gliclazide",
                     "chlorpropamide", "tolbutamide", "tolazamide"],
    "sulfonylureas": ["glipizide", "glyburide", "glimepiride", "gliclazide",
                      "chlorpropamide", "tolbutamide", "tolazamide"],
    "insulin": ["insulin"],
    "heparin": ["heparin", "enoxaparin", "dalteparin", "fondaparinux"],
}

# Route/administration entries that are not systemic therapeutic doses:
# catheter flushes, dwell/lock solutions, and circuit priming. A 10-unit
# heparin flush carries no thrombocytopenia risk, so counting it as in-scope
# inflates applicability.
NONTHERAPEUTIC = re.compile(
    r"flush|dwell|\block\b|prim(e|ing)|machine priming", re.I)

_punct = re.compile(r"[^a-z0-9 ]+")
_ws = re.compile(r"\s+")
_num = re.compile(r"\b\d+(\.\d+)?\b")
_paren = re.compile(r"\([^)]*\)")


def canon_lab(name):
    """
    'eGFR (CKD-EPI 2021)' -> 'egfr' -> 'creatinine'.
    Strips parentheticals and punctuation before looking up LAB_ALIAS, so
    benchmark lab_name variants do not silently count as uncovered.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    x = _paren.sub(" ", name.lower())
    x = _punct.sub(" ", x)
    x = _ws.sub(" ", x).strip()
    if x in LAB_ALIAS:
        return LAB_ALIAS[x]
    if x in LAB_UNAVAILABLE:
        return "__UNAVAILABLE__"
    for key in LAB_ALIAS:
        if x.startswith(key) or key in x.split():
            return LAB_ALIAS[key]
    for key in LAB_UNAVAILABLE:
        if key in x:
            return "__UNAVAILABLE__"
    return None


def normalize_drug(s):
    """MIMIC `drug` free text -> best-guess ingredient token(s)."""
    if not isinstance(s, str) or not s.strip():
        return ""
    x = s.lower()
    x = _punct.sub(" ", x)
    x = re.sub(r"(\d)([a-z])", r"\1 \2", x)   # 500mg -> 500 mg
    x = re.sub(r"([a-z])(\d)", r"\1 \2", x)   # b12 -> b 12
    x = _num.sub(" ", x)
    toks = [t for t in _ws.sub(" ", x).split()
            if t and t not in _FORM_TOKENS and t not in _SALTS]
    toks = [BRAND2GENERIC.get(t, t) for t in toks]
    return " ".join(toks).strip()


def drug_tokens(norm):
    return set(norm.split())


# ============================================================================
# STAGE 0 — derive the operational rule set from your own artifacts
# ============================================================================

def stage0_rules(out_dir, verbose=True):
    """
    Derive the rule set MedGuard actually operates on.

    Primary source: distinct (rule_id, drug, lab_name, operator, threshold)
    tuples present in medconflict_bench_v2.jsonl. That is the operational
    definition -- whatever the paper text says, these are the rules the system
    was evaluated against.
    """
    # Prefer the newest benchmark version present.
    bench_path = None
    for cand in ("medconflict_bench_v4.jsonl", "medconflict_bench_v3.jsonl",
                 "medconflict_bench_v2.jsonl"):
        if (BENCH / cand).exists():
            bench_path = BENCH / cand
            break
    if bench_path is None:
        sys.exit(f"FATAL: no medconflict_bench_v*.jsonl in {BENCH}")
    if verbose:
        print(f"  using benchmark: {bench_path.name}")

    rules = {}
    strata_of_rule = defaultdict(set)
    n = 0
    with open(bench_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            n += 1
            rid = ex.get("rule_id")
            if not rid or rid in ("", "none", None):
                continue
            key = rid
            if key not in rules:
                rules[key] = {
                    "rule_id": rid,
                    "drug": (ex.get("drug") or "").strip(),
                    "interacting_drug": (ex.get("interacting_drug") or "").strip(),
                    "lab_name": (ex.get("lab_name") or "").strip(),
                    "operator": ex.get("operator"),
                    "threshold": ex.get("threshold"),
                    "conflict_condition": (ex.get("conflict_condition") or "").strip(),
                }
            strata_of_rule[rid].add(ex.get("stratum"))

    for rid, r in rules.items():
        r["strata"] = sorted(s for s in strata_of_rule[rid] if s is not None)

    by_stratum = Counter()
    for r in rules.values():
        for s in r["strata"]:
            by_stratum[s] += 1

    if verbose:
        print("=" * 74)
        print("STAGE 0 — OPERATIONAL RULE SET (derived, not hardcoded)")
        print("=" * 74)
        print(f"  benchmark examples scanned : {n}")
        print(f"  DISTINCT rule_id values    : {len(rules)}   <-- the real rule count")
        print(f"  by stratum                 : {dict(sorted(by_stratum.items()))}")
        print()
        print("  ** Use this number in the rebuttal. Do not assert 19 or 36 **")
        print("  ** until you have read it here. **")
        print()
        unavailable = [r["rule_id"] for r in rules.values()
                       if canon_lab(r["lab_name"]) == "__UNAVAILABLE__"]
        unmapped = sorted({r["lab_name"] for r in rules.values()
                           if r["lab_name"] and canon_lab(r["lab_name"]) is None})
        print(f"  rules whose lab is UNAVAILABLE in MIMIC-IV hosp: {len(unavailable)}")
        for rid in unavailable:
            print(f"      {rid}  (lab={rules[rid]['lab_name']})")
        if unmapped:
            print(f"  !! lab_name values with no MIMIC itemid mapping: {unmapped}")
            print("     Add them to LAB_ALIAS / LAB_ITEMIDS or they count as uncovered.")
        print()

    # Expand each rule's drug into SINGLE-TOKEN ingredients.
    # Stage 1 matches token sets, so a multi-word key such as
    # "dabigatran etexilate" or "metformin metformin" could never match.
    expanded = []
    audit = []
    for r in rules.values():
        raw = (r["drug"] or "").lower().strip()
        d = normalize_drug(r["drug"]) or raw
        members = DRUG_CLASSES.get(d) or DRUG_CLASSES.get(raw)
        if members is None:
            members = [t for t in dict.fromkeys(d.split()) if len(t) > 2]
            if not members:
                members = [t for t in dict.fromkeys(raw.split()) if len(t) > 2] or [d]
        for m in members:
            expanded.append({**r, "ingredient": m})
        audit.append((r["rule_id"], r["drug"], sorted(set(members))))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rule_set_derived.json", "w") as f:
        json.dump({"n_rules": len(rules),
                   "by_stratum": dict(by_stratum),
                   "rules": list(rules.values()),
                   "expanded_ingredients": sorted({e["ingredient"] for e in expanded})},
                  f, indent=2, default=str)

    with open(out_dir / "rule_drug_ingredient_audit.csv", "w", encoding="utf-8") as f:
        f.write("rule_id,rule_drug,matched_ingredients\n")
        for rid, drug, mem in sorted(audit):
            f.write(f'"{rid}","{drug}","{" ".join(mem)}"\n')

    if verbose:
        ings = sorted({e["ingredient"] for e in expanded})
        print(f"  expanded to {len(ings)} distinct SINGLE-TOKEN ingredients:")
        print(f"    {', '.join(ings)}")
        print()
        multi = [a for a in audit if any(" " in m for m in a[2])]
        if multi:
            print("  !! still multi-word (these cannot match, fix them):")
            for rid, drug, mem in multi:
                print(f"     {rid}: {drug!r} -> {mem}")
            print()
        print("  AUDIT — rule drug -> ingredients it will match on:")
        for rid, drug, mem in sorted(audit)[:60]:
            print(f"    {rid:<28} {drug[:34]:<34} -> {', '.join(mem)}")
        print(f"  (full list in rule_drug_ingredient_audit.csv)")
        print()

    return list(rules.values()), expanded


# ============================================================================
# STAGE 1 — scan every prescription, tag in-scope vs out-of-scope
# ============================================================================

def stage1_prescriptions(out_dir, expanded, quick=False):
    ck = out_dir / ("presc_scoped_quick.parquet" if quick else "presc_scoped.parquet")
    if ck.exists():
        print(f"STAGE 1 — cached, loading {ck.name}")
        return pd.read_parquet(ck)

    print("=" * 74)
    print("STAGE 1 — PRESCRIPTION SCAN")
    print("=" * 74)

    ing2rules = defaultdict(list)
    for e in expanded:
        ing2rules[e["ingredient"]].append(e["rule_id"])
    ingredients = set(ing2rules)

    path = MIMIC / "prescriptions.csv.gz"
    usecols = ["subject_id", "hadm_id", "drug", "starttime"]

    hadm_keep = None
    if quick:
        rng = np.random.default_rng(SEED)
        seen = set()
        for ch in pd.read_csv(path, usecols=["hadm_id"], chunksize=CHUNK,
                              compression="gzip"):
            seen.update(ch["hadm_id"].dropna().astype("int64").unique().tolist())
            if len(seen) > 400_000:
                break
        seen = np.array(sorted(seen))
        hadm_keep = set(rng.choice(seen, size=min(QUICK_N_HADM, len(seen)),
                                   replace=False).tolist())
        print(f"  QUICK MODE: sampled {len(hadm_keep)} hadm_ids")

    rows = []
    total = 0
    norm_cache = {}
    unmatched = Counter()

    for i, ch in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK,
                                       compression="gzip",
                                       parse_dates=["starttime"])):
        if hadm_keep is not None:
            ch = ch[ch["hadm_id"].isin(hadm_keep)]
            if ch.empty:
                continue
        total += len(ch)

        norms = []
        for d in ch["drug"].values:
            if d not in norm_cache:
                norm_cache[d] = normalize_drug(d)
            norms.append(norm_cache[d])
        ch = ch.assign(drug_norm=norms)

        matched_rule, matched_ing = [], []
        for nm in ch["drug_norm"].values:
            toks = drug_tokens(nm)
            hit = toks & ingredients
            if hit:
                ing = sorted(hit)[0]
                matched_ing.append(ing)
                matched_rule.append("|".join(sorted(set(ing2rules[ing]))))
            else:
                matched_ing.append("")
                matched_rule.append("")
        ch = ch.assign(ingredient=matched_ing, rule_ids=matched_rule)

        for nm, ing in zip(ch["drug_norm"].values, ch["ingredient"].values):
            if not ing and nm:
                unmatched[nm] += 1

        ch["nontherapeutic"] = [bool(NONTHERAPEUTIC.search(str(d)))
                                for d in ch["drug"].values]

        # Memory: 17M rows x 4 string columns will not fit in a Colab session.
        # drug/drug_norm are only needed for the unmatched audit above, which
        # is already accumulated in the Counter, so drop them here.
        ch = ch.drop(columns=["drug", "drug_norm"])
        ch["ingredient"] = ch["ingredient"].astype("category")
        ch["rule_ids"] = ch["rule_ids"].astype("category")

        rows.append(ch)
        if (i + 1) % 5 == 0:
            print(f"    ...{total:,} prescription rows scanned")

    df = pd.concat(rows, ignore_index=True)
    in_scope = (df["ingredient"] != "").sum()

    print(f"  total prescription events   : {len(df):,}")
    print(f"  IN SCOPE (drug matches rule): {in_scope:,}  "
          f"({in_scope / max(len(df),1):.2%})   <-- METRIC A")
    print()
    print("  Top 25 UNMATCHED normalized drug strings (sanity-check these by eye —")
    print("  if a rule drug appears here, your normalizer is dropping it):")
    for nm, c in unmatched.most_common(25):
        print(f"    {c:>9,}  {nm}")
    print()

    with open(out_dir / "unmatched_drugs.csv", "w", encoding="utf-8") as f:
        f.write("drug_norm,count\n")
        for nm, c in unmatched.most_common(3000):
            f.write(f'"{nm}",{c}\n')

    df.to_parquet(ck, index=False)
    print(f"  checkpoint -> {ck.name}")
    return df


# ============================================================================
# STAGE 2 — extract only the labs we need (the expensive stage)
# ============================================================================

def stage2_labs(out_dir, hadm_filter=None, quick=False):
    ck = out_dir / ("labs_quick.parquet" if quick else "labs.parquet")
    if ck.exists():
        print(f"STAGE 2 — cached, loading {ck.name}")
        return pd.read_parquet(ck)

    print("=" * 74)
    print("STAGE 2 — LAB EXTRACTION  (slow: labevents is ~158M rows)")
    print("=" * 74)

    want = {i for v in LAB_ITEMIDS.values() for i in _itemids(v)}
    id2name = {i: k for k, v in LAB_ITEMIDS.items() for i in _itemids(v)}
    path = MIMIC / "labevents.csv.gz"
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "flag"]

    keep, scanned = [], 0
    for i, ch in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK,
                                       compression="gzip",
                                       parse_dates=["charttime"],
                                       low_memory=False)):
        scanned += len(ch)
        ch = ch[ch["itemid"].isin(want)]
        if hadm_filter is not None and not ch.empty:
            ch = ch[ch["hadm_id"].isin(hadm_filter)]
        if not ch.empty:
            keep.append(ch)
        if (i + 1) % 20 == 0:
            got = sum(len(k) for k in keep)
            print(f"    ...{scanned:,} lab rows scanned, {got:,} kept")

    labs = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=usecols)
    labs["lab"] = labs["itemid"].map(id2name)
    labs = labs.dropna(subset=["valuenum", "charttime"])
    print(f"  kept {len(labs):,} relevant lab results")
    labs.to_parquet(ck, index=False)
    print(f"  checkpoint -> {ck.name}")
    return labs


# ============================================================================
# eGFR — CKD-EPI 2021 (race-free)
# ============================================================================

def add_egfr(labs, out_dir):
    pts = pd.read_csv(MIMIC / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"],
                      compression="gzip")
    cr = labs[labs["lab"] == "creatinine"].merge(pts, on="subject_id", how="left")
    cr = cr.dropna(subset=["gender", "anchor_age", "valuenum"])
    cr = cr[cr["valuenum"] > 0]

    female = cr["gender"].eq("F")
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    scr = cr["valuenum"].to_numpy(dtype=float)
    age = cr["anchor_age"].to_numpy(dtype=float)

    egfr = (142.0
            * np.power(np.minimum(scr / kappa, 1.0), alpha)
            * np.power(np.maximum(scr / kappa, 1.0), -1.200)
            * np.power(0.9938, age)
            * np.where(female, 1.012, 1.0))

    out = cr.assign(lab="egfr", valuenum=egfr)[
        ["subject_id", "hadm_id", "charttime", "valuenum", "flag", "lab"]]
    print(f"  computed eGFR for {len(out):,} creatinine results (CKD-EPI 2021)")
    return pd.concat([labs, out], ignore_index=True)


# ============================================================================
# STAGE 3 — pair prescriptions to labs, compute A/B/C/D
# ============================================================================

def stage3_join(out_dir, presc, labs, rules, quick=False):
    """
    Pair prescriptions to labs within +/- LAB_WINDOW_HOURS and compute A/B/C/D.

    Uses pd.merge_asof rather than a Python loop: with ~17M prescriptions a
    per-row scan does not finish in a Colab session.
    """
    print("=" * 74)
    print("STAGE 3 — PAIRING + COVERAGE METRICS")
    print("=" * 74)

    rule_by_id = {r["rule_id"]: r for r in rules}
    win = pd.Timedelta(hours=LAB_WINDOW_HOURS)

    def required_lab(rule_ids):
        for rid in (rule_ids or "").split("|"):
            r = rule_by_id.get(rid)
            if not r:
                continue
            raw = (r.get("lab_name") or "").strip()
            if not raw:
                return "__CATEGORICAL__"
            stripped = _ws.sub(" ", _punct.sub(" ", _paren.sub(" ", raw.lower()))).strip()
            if stripped.startswith("egfr"):
                return "egfr"
            c = canon_lab(raw)
            if c:
                return c
        return None

    presc = presc.dropna(subset=["hadm_id", "starttime"]).copy()
    presc["hadm_id"] = presc["hadm_id"].astype("int64")

    labs = labs.dropna(subset=["hadm_id", "charttime", "valuenum"]).copy()
    labs["hadm_id"] = labs["hadm_id"].astype("int64")

    scoped = presc[presc["ingredient"].astype(str) != ""].copy()
    scoped["req_lab"] = (scoped["rule_ids"].astype(str)
                         .map(required_lab).astype("object"))

    def asof_hit(left, right, by_cols, ltime, rtime):
        """True where a right-row exists within +/- win of each left-row."""
        if left.empty or right.empty:
            return np.zeros(len(left), dtype=bool)
        L = left.sort_values(ltime, kind="mergesort").copy()
        R = right.sort_values(rtime, kind="mergesort").copy()
        # merge_asof requires identical dtypes on every `by` key; pandas may
        # infer StringDtype on one side and object on the other.
        for c in by_cols:
            if L[c].dtype != R[c].dtype:
                L[c] = L[c].astype(str)
                R[c] = R[c].astype(str)
        hit = np.zeros(len(L), dtype=bool)
        for direction in ("backward", "forward"):
            m = pd.merge_asof(L, R, left_on=ltime, right_on=rtime,
                              by=by_cols, direction=direction,
                              tolerance=win)
            hit |= m[rtime].notna().to_numpy()
        return pd.Series(hit, index=L.index).reindex(left.index).to_numpy()

    # ---- METRIC B: is the governing lab present in the window? -------------
    cat_mask = scoped["req_lab"].eq("__CATEGORICAL__").to_numpy()
    need_mask = scoped["req_lab"].notna().to_numpy() & ~cat_mask
    evaluable = cat_mask.copy()

    if need_mask.any():
        L = scoped.loc[need_mask, ["hadm_id", "starttime", "req_lab"]].copy()
        R = labs[["hadm_id", "lab", "charttime"]].rename(columns={"lab": "req_lab"})
        evaluable[need_mask] = asof_hit(L, R, ["hadm_id", "req_lab"],
                                        "starttime", "charttime")
    scoped["evaluable"] = evaluable

    # ---- METRIC D: any abnormal-flagged lab near the order -----------------
    ab = labs[labs["flag"].astype(str).str.strip().ne("")
              & labs["flag"].notna()][["hadm_id", "charttime"]]
    presc["abnormal_lab_near"] = asof_hit(
        presc[["hadm_id", "starttime"]], ab, ["hadm_id"], "starttime", "charttime")
    scoped["abnormal_lab_near"] = presc["abnormal_lab_near"].reindex(scoped.index)

    n_all = len(presc)
    n_scope = len(scoped)
    n_eval = int(scoped["evaluable"].sum())

    if "nontherapeutic" in scoped.columns:
        nt = scoped["nontherapeutic"].fillna(False).to_numpy().astype(bool)
        n_nt = int(nt.sum())
        n_scope_t = n_scope - n_nt
        n_eval_t = int((scoped["evaluable"].to_numpy() & ~nt).sum())
        A_t = n_scope_t / max(n_all, 1)
        B_t = n_eval_t / max(n_scope_t, 1)
        C_t = 1.0 - n_eval_t / max(n_all, 1)
    else:
        n_nt = 0
        A_t = B_t = C_t = None

    A = n_scope / max(n_all, 1)
    B = n_eval / max(n_scope, 1)
    C = 1.0 - (n_eval / max(n_all, 1))

    n_haz = int(presc["abnormal_lab_near"].sum())
    n_haz_cov = int((scoped["evaluable"] & scoped["abnormal_lab_near"]).sum())
    D = n_haz_cov / max(n_haz, 1)

    res = {
        "mode": "quick" if quick else "full",
        "lab_window_hours": LAB_WINDOW_HOURS,
        "n_prescription_events": int(n_all),
        "n_in_scope": int(n_scope),
        "n_evaluable": int(n_eval),
        "n_hazard_proxy": int(n_haz),
        "n_hazard_proxy_covered": int(n_haz_cov),
        "A_applicability": round(A, 6),
        "B_evaluable_given_scope": round(B, 6),
        "C_silent_passthrough": round(C, 6),
        "D_hazard_proxy_coverage": round(D, 6),
    }

    print()
    print(f"  Prescription events analysed        : {n_all:,}")
    print(f"  A  APPLICABILITY  drug in rule set  : {A:>8.2%}  ({n_scope:,})")
    print(f"  B  EVALUABLE      lab within {LAB_WINDOW_HOURS}h    : {B:>8.2%}  ({n_eval:,} of {n_scope:,})")
    print(f"  C  SILENT PASS-THROUGH              : {C:>8.2%}   <-- HEADLINE")
    print(f"       ({n_all - n_eval:,} orders MedGuard clears having checked nothing)")
    print(f"  D  HAZARD-PROXY COVERAGE            : {D:>8.2%}  ({n_haz_cov:,} of {n_haz:,})")
    if A_t is not None:
        print()
        print(f"  EXCLUDING NON-THERAPEUTIC ENTRIES (flush / dwell / lock / priming)")
        print(f"    excluded from in-scope            : {n_nt:,}")
        print(f"    A  applicability                  : {A_t:>8.2%}")
        print(f"    B  evaluable | in scope           : {B_t:>8.2%}")
        print(f"    C  silent pass-through            : {C_t:>8.2%}")
        res_extra = {"n_nontherapeutic_excluded": n_nt,
                     "A_applicability_therapeutic": round(A_t, 6),
                     "B_evaluable_therapeutic": round(B_t, 6),
                     "C_silent_passthrough_therapeutic": round(C_t, 6)}
        res.update(res_extra)
    print()
    print("  Interpretation for the rebuttal:")
    print("   A  = how often the rule set applies at all")
    print("   B  = of those, how often it can actually be evaluated")
    print("   C  = the alibi number oJaU is pointing at. Report it plainly.")
    print("   D  = coverage where a clinician would plausibly look (abnormal lab")
    print("        drawn near the order). Proxy, NOT an error label. Say so.")
    print()

    in_scope_no_lab = int((~scoped["evaluable"] & ~pd.Series(cat_mask, index=scoped.index)).sum())
    print(f"  DESIGN DEFECT TO REPORT: {in_scope_no_lab:,} orders match a rule but")
    print("  have no governing lab in the window. The system passes these through")
    print("  rather than abstaining -- a coverage gap AND a design error.")
    print()

    by_ing = (scoped.groupby("ingredient", observed=True)
              .agg(n=("ingredient", "size"), evaluable=("evaluable", "sum"))
              .sort_values("n", ascending=False))
    by_ing.to_csv(out_dir / "coverage_by_ingredient.csv")

    with open(out_dir / "coverage_results.json", "w") as f:
        json.dump(res, f, indent=2)

    lines = [
        "MEDGUARD COVERAGE DENOMINATOR — MIMIC-IV (Reviewer oJaU, Experiment 1)",
        "=" * 74,
        f"mode={res['mode']}  lab_window={LAB_WINDOW_HOURS}h",
        "",
        "NOTE: MIMIC-IV carries no prescribing-error labels. These are coverage",
        "measurements, not error-detection rates. D is a proxy denominator.",
        "",
        f"prescription events        {n_all:,}",
        f"A applicability            {A:.2%}",
        f"B evaluable | in scope     {B:.2%}",
        f"C silent pass-through      {C:.2%}",
        f"D hazard-proxy coverage    {D:.2%}",
        f"in-scope but no lab        {in_scope_no_lab:,}",
    ]
    (out_dir / "coverage_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote coverage_results.json / coverage_report.txt / coverage_by_ingredient.csv")
    return res


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "0", "1", "2", "3"])
    ap.add_argument("--quick", action="store_true",
                    help=f"sample {QUICK_N_HADM} hadm_ids for a fast preliminary number")
    ap.add_argument("--bench-dir", default=None,
                    help="folder containing medconflict_bench_v2.jsonl")
    ap.add_argument("--mimic-dir", default=None,
                    help="folder containing prescriptions.csv.gz, labevents.csv.gz, patients.csv.gz")
    ap.add_argument("--out-dir", default=None,
                    help="where to write results")
    args = ap.parse_args()

    global BENCH, MIMIC, OUT
    if args.bench_dir:
        BENCH = Path(args.bench_dir)
    if args.mimic_dir:
        MIMIC = Path(args.mimic_dir)
    if args.out_dir:
        OUT = Path(args.out_dir)

    print(f"  bench dir : {BENCH}")
    if args.stage in ("all", "1", "2", "3"):
        print(f"  mimic dir : {MIMIC}")
    print(f"  out dir   : {OUT}")
    print()

    out_dir = OUT / ("quick" if args.quick else "full")
    out_dir.mkdir(parents=True, exist_ok=True)

    rules, expanded = stage0_rules(out_dir)
    if args.stage == "0":
        return

    presc = stage1_prescriptions(out_dir, expanded, quick=args.quick)
    if args.stage == "1":
        return

    hadm_filter = None
    if args.quick:
        hadm_filter = set(presc["hadm_id"].dropna().astype("int64").unique().tolist())
    labs = stage2_labs(out_dir, hadm_filter=hadm_filter, quick=args.quick)
    if args.stage == "2":
        return

    labs = add_egfr(labs, out_dir)
    stage3_join(out_dir, presc, labs, rules, quick=args.quick)


if __name__ == "__main__":
    main()
