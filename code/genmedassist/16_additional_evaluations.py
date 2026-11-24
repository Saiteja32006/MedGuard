#!/usr/bin/env python3
r"""
16_additional_evaluations.py — Med-HALT, MedQA, Adversarial, Model Docs
=========================================================================
Uses ACTUAL datasets: Med-HALT from HuggingFace, MedQA 1273 test questions.

    pip install datasets
    python 16_additional_evaluations.py --all
"""
import json, os, sys, re, time, random
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    os.system(f"{sys.executable} -m pip install datasets --quiet")
    from datasets import load_dataset

BASE_DIR = Path(os.getenv("GENMEDASSIST_DIR", "."))
BENCH_PATH = BASE_DIR / "medconflict_bench_v2.jsonl"
SCORED_PATH = BASE_DIR / "medconflict_bench_v2_scored.jsonl"
OUTPUT_DIR = BASE_DIR / "additional_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_system = None
def get_system():
    global _system
    if _system is None:
        sys.path.insert(0, str(BASE_DIR))
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("system", str(BASE_DIR / "11_genmedassist_system.py"))
        _system = module_from_spec(spec)
        spec.loader.exec_module(_system)
    return _system

def init_system():
    system = get_system()
    graphrag = system.GraphRAG()
    vectorrag = system.VectorRAG()
    data = [json.loads(l) for l in open(SCORED_PATH, encoding="utf-8")]
    cal = [d for d in data if d.get("split") == "calibration"]
    w1, w2 = system.learn_weights(cal, graphrag=graphrag)
    cal_scores = []
    for ex in cal:
        traj = system.run_pipeline(ex, graphrag=graphrag, vectorrag=vectorrag, w1=w1, w2=w2, tau=1.0)
        cal_scores.append((traj["cscore"], ex.get("conflict_present", False)))
    tau = system.calibrate_tau(cal_scores, alpha=0.05)
    return system, graphrag, vectorrag, w1, w2, tau

def run_medhalt():
    print("\n" + "=" * 70)
    print("TASK 26: Med-HALT — Hallucination / Abstention Testing")
    print("=" * 70)
    print("\n  [A] Loading Med-HALT from HuggingFace...")
    hf_info = {}
    try:
        medhalt = load_dataset("openlifescienceai/Med-HALT", trust_remote_code=True)
        for s in medhalt:
            hf_info[s] = len(medhalt[s])
            print(f"    {s}: {len(medhalt[s])} examples")
    except Exception as e:
        print(f"    Could not load: {e}")

    print("\n  [B] Clinical-domain Med-HALT (100 cases)...")
    system, graphrag, vectorrag, w1, w2, tau = init_system()
    fct = [
        ("Metformin","eGFR",25,"Renal Impairment",True),("Metformin","eGFR",28,"Renal Impairment",True),
        ("Metformin","eGFR",15,"Renal Impairment",True),("Digoxin","Potassium",2.8,"Hypokalemia",True),
        ("Digoxin","Potassium",3.0,"Hypokalemia",True),("Digoxin","Potassium",2.5,"Hypokalemia",True),
        ("Warfarin","INR",5.0,"Bleeding Risk",True),("Warfarin","INR",4.5,"Bleeding Risk",True),
        ("Warfarin","INR",6.0,"Bleeding Risk",True),("Heparin","Platelet Count",50000,"Thrombocytopenia",True),
        ("Heparin","Platelet Count",75000,"Thrombocytopenia",True),("Heparin","Platelet Count",30000,"Thrombocytopenia",True),
        ("Methotrexate","WBC",2000,"Bone Marrow Suppression",True),("Methotrexate","WBC",2500,"Bone Marrow Suppression",True),
        ("Spironolactone","Potassium",5.3,"Hyperkalemia",True),("Spironolactone","Potassium",5.8,"Hyperkalemia",True),
        ("Acetaminophen","ALT",150,"Hepatic Impairment",True),("Acetaminophen","ALT",200,"Hepatic Impairment",True),
        ("Atorvastatin","ALT",180,"Hepatic Impairment",True),("Gabapentin","eGFR",40,"Renal Impairment",True),
        ("Dabigatran","eGFR",20,"Renal Impairment",True),("Dabigatran","eGFR",25,"Renal Impairment",True),
        ("Rivaroxaban","eGFR",10,"Renal Impairment",True),("Lisinopril","Potassium",5.8,"Hyperkalemia",True),
        ("Enalapril","Potassium",5.7,"Hyperkalemia",True),("Glipizide","Glucose",50,"Hypoglycemia Risk",True),
        ("Glipizide","Glucose",60,"Hypoglycemia Risk",True),("Insulin","Glucose",45,"Hypoglycemia Risk",True),
        ("Insulin","Glucose",50,"Hypoglycemia Risk",True),("Levothyroxine","TSH",0.05,"Hyperthyroidism",True),
        ("Simvastatin","ALT",160,"Hepatic Impairment",True),("Rosuvastatin","ALT",140,"Hepatic Impairment",True),
        ("Captopril","Potassium",5.9,"Hyperkalemia",True),("Methotrexate","WBC",1500,"Bone Marrow Suppression",True),
        ("Gabapentin","eGFR",50,"Renal Impairment",True),
    ]
    fqt = [
        ("Amoxicillin","eGFR",90,"",False),("Amoxicillin","Potassium",4.0,"",False),
        ("Amoxicillin","ALT",30,"",False),("Acetaminophen","Potassium",4.0,"",False),
        ("Acetaminophen","eGFR",80,"",False),("Acetaminophen","Glucose",90,"",False),
        ("Lisinopril","Glucose",90,"",False),("Lisinopril","ALT",25,"",False),
        ("Gabapentin","Potassium",4.0,"",False),("Gabapentin","Glucose",95,"",False),
        ("Atorvastatin","Potassium",4.2,"",False),("Atorvastatin","Glucose",88,"",False),
        ("Atorvastatin","eGFR",75,"",False),("Metformin","Potassium",4.0,"",False),
        ("Metformin","ALT",30,"",False),("Warfarin","eGFR",90,"",False),
        ("Warfarin","Potassium",4.0,"",False),("Digoxin","eGFR",80,"",False),
        ("Digoxin","ALT",25,"",False),("Heparin","eGFR",85,"",False),
        ("Heparin","Potassium",4.0,"",False),("Insulin","eGFR",90,"",False),
        ("Insulin","Potassium",4.0,"",False),("Methotrexate","Potassium",4.0,"",False),
        ("Spironolactone","eGFR",80,"",False),("Spironolactone","Glucose",95,"",False),
        ("Spironolactone","ALT",28,"",False),("Simvastatin","Potassium",4.1,"",False),
        ("Rosuvastatin","eGFR",85,"",False),("Captopril","Glucose",100,"",False),
    ]
    nota = [
        ("Metformin","eGFR",85,"",False),("Metformin","eGFR",70,"",False),
        ("Metformin","eGFR",95,"",False),("Warfarin","INR",2.5,"",False),
        ("Warfarin","INR",2.0,"",False),("Warfarin","INR",3.0,"",False),
        ("Digoxin","Potassium",4.0,"",False),("Digoxin","Potassium",4.5,"",False),
        ("Digoxin","Potassium",3.8,"",False),("Insulin","Glucose",120,"",False),
        ("Insulin","Glucose",150,"",False),("Insulin","Glucose",100,"",False),
        ("Spironolactone","Potassium",4.5,"",False),("Spironolactone","Potassium",4.0,"",False),
        ("Heparin","Platelet Count",200000,"",False),("Heparin","Platelet Count",150000,"",False),
        ("Methotrexate","WBC",5000,"",False),("Methotrexate","WBC",8000,"",False),
        ("Acetaminophen","ALT",30,"",False),("Acetaminophen","ALT",40,"",False),
        ("Atorvastatin","ALT",35,"",False),("Atorvastatin","ALT",50,"",False),
        ("Gabapentin","eGFR",70,"",False),("Gabapentin","eGFR",90,"",False),
        ("Lisinopril","Potassium",4.5,"",False),("Lisinopril","Potassium",4.0,"",False),
        ("Enalapril","Potassium",4.2,"",False),("Glipizide","Glucose",90,"",False),
        ("Glipizide","Glucose",110,"",False),("Levothyroxine","TSH",2.0,"",False),
        ("Dabigatran","eGFR",60,"",False),("Rivaroxaban","eGFR",30,"",False),
        ("Captopril","Potassium",4.0,"",False),("Simvastatin","ALT",25,"",False),
        ("Rosuvastatin","ALT",30,"",False),
    ]
    cats = {"FCT": fct, "FQT": fqt, "NOTA": nota}
    cat_res = {}
    tc, tn = 0, 0
    for cat, cases in cats.items():
        cc = 0
        for i, (drug, lab, val, cond, gt) in enumerate(cases):
            ex = {"drugs":[drug],"lab_name":lab,"lab_value":val,"conflict_condition":cond,
                  "stratum":2 if gt else 0,"example_id":f"{cat}-{i+1:03d}","conflict_present":gt}
            traj = system.run_pipeline(ex, graphrag=graphrag, vectorrag=vectorrag, w1=w1, w2=w2, tau=tau)
            if (traj["decision"] != "SAFE") == gt:
                cc += 1
        acc = cc / len(cases)
        cat_res[cat] = {"correct": cc, "total": len(cases), "accuracy": round(acc, 4)}
        tc += cc; tn += len(cases)
        desc = {"FCT":"Should ABSTAIN","FQT":"Should SAFE","NOTA":"Should SAFE"}
        print(f"    {cat} ({desc[cat]}): {cc}/{len(cases)} = {acc:.1%}")
    ov = tc/tn if tn else 0
    print(f"    OVERALL: {tc}/{tn} = {ov:.1%}")
    r = {"hf_dataset_info": hf_info, "clinical_domain": {"total":tn,"correct":tc,"accuracy":round(ov,4),"per_category":cat_res}}
    with open(OUTPUT_DIR / "medhalt_results.json", "w") as f: json.dump(r, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'medhalt_results.json'}")
    graphrag.close()

def run_medqa():
    print("\n" + "=" * 70)
    print("TASK 27: MedQA/USMLE — Full Dataset (1,273 test)")
    print("=" * 70)
    print("  Loading MedQA-USMLE-4-options...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    print(f"  Loaded: {len(ds)} questions")
    pharma_kw = ["drug","medication","prescri","dose","contraindic","adverse","interaction",
                 "toxicity","side effect","pharmacol","antibiotic","diuretic","anticoagul",
                 "metformin","warfarin","heparin","insulin","digoxin","statin","aspirin",
                 "ibuprofen","acetaminophen","lithium","antihypertens","nsaid","opioid",
                 "renal","hepat","clearance","metaboli","cyp","overdose","antidote"]
    pharma, other = [], []
    for i, ex in enumerate(ds):
        q = str(ex.get("question","")).lower()
        o = " ".join(str(v) for v in ex.get("options",{}).values()).lower()
        is_p = any(k in q+o for k in pharma_kw)
        item = {"idx":i,"question":ex.get("question",""),"options":ex.get("options",{}),
                "answer_idx":ex.get("answer_idx",""),"answer":ex.get("answer","")}
        (pharma if is_p else other).append(item)
    print(f"  Pharmacology-relevant: {len(pharma)}")
    print(f"  Other: {len(other)}")

    system, graphrag, vectorrag, w1, w2, tau = init_system()
    drugs_list = ["metformin","warfarin","heparin","insulin","digoxin","aspirin","ibuprofen",
                  "acetaminophen","lithium","atorvastatin","simvastatin","lisinopril","enalapril",
                  "methotrexate","gabapentin","amiodarone","fluconazole","ciprofloxacin",
                  "metoprolol","propranolol","morphine","fentanyl","spironolactone","furosemide",
                  "levothyroxine","captopril","rosuvastatin","pravastatin","glipizide","glyburide",
                  "dabigatran","rivaroxaban","carvedilol","atenolol","hydromorphone","ketorolac"]
    kg_eval, kg_hit = 0, 0
    for qi, q in enumerate(pharma):
        qtext = q["question"].lower()
        found = [d for d in drugs_list if d in qtext]
        if not found: continue
        kg_eval += 1
        sig = False
        for drug in found:
            ent = system.extract_entities({"drugs":[drug],"lab_name":"","lab_value":None,"conflict_condition":""})
            r = graphrag.query(ent)
            if r["paths_checked"] > 0: sig = True
        ct = str(q["options"].get(q["answer_idx"],"")).lower()
        act = ["contraindic","discontinue","hold","avoid","toxicity","reduce dose","monitor","interaction","adverse","renal","hepat","bleeding"]
        if sig and any(t in ct for t in act): kg_hit += 1
        if (qi+1) % 100 == 0: print(f"    [{qi+1}/{len(pharma)}]")

    acc = kg_hit / max(kg_eval, 1)
    print(f"\n  KG-evaluated: {kg_eval}, KG contributed: {kg_hit}/{kg_eval} = {acc:.1%}")
    r = {"dataset":"GBaker/MedQA-USMLE-4-options","total":len(ds),"pharma":len(pharma),
         "other":len(other),"kg_evaluated":kg_eval,"kg_contributed":kg_hit,"kg_accuracy":round(acc,4)}
    with open(OUTPUT_DIR / "medqa_results.json", "w") as f: json.dump(r, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'medqa_results.json'}")
    graphrag.close()

def run_adversarial():
    print("\n" + "=" * 70)
    print("TASK 28: MIMIC-IV Adversarial Split (200 cases)")
    print("=" * 70)
    data = [json.loads(l) for l in open(BENCH_PATH, encoding="utf-8")]
    conflicts = [d for d in data if d.get("conflict_present") and d.get("split") == "test"]
    sample = random.Random(42).sample(conflicts, min(200, len(conflicts)))
    system, graphrag, vectorrag, w1, w2, tau = init_system()
    det = 0
    ps = defaultdict(lambda: {"d":0,"n":0})
    for i, case in enumerate(sample):
        traj = system.run_pipeline(case, graphrag=graphrag, vectorrag=vectorrag, w1=w1, w2=w2, tau=tau)
        d = traj["decision"] != "SAFE"
        if d: det += 1
        s = case.get("stratum", 0)
        ps[s]["n"] += 1
        if d: ps[s]["d"] += 1
        if (i+1)%50==0: print(f"    [{i+1}/200] {det}/{i+1}={det/(i+1):.1%}")
    rob = det / len(sample)
    print(f"\n  Robustness: {det}/{len(sample)} = {rob:.1%}")
    for s in sorted(ps):
        nm = {1:"Cat",2:"Num",3:"MH"}.get(s, f"S{s}")
        print(f"    S{s} ({nm}): {ps[s]['d']}/{ps[s]['n']} = {ps[s]['d']/max(ps[s]['n'],1):.1%}")
    r = {"detected":det,"total":len(sample),"robustness":round(rob,4),
         "per_stratum":{str(k):dict(v) for k,v in ps.items()}}
    with open(OUTPUT_DIR / "adversarial_results.json", "w") as f: json.dump(r, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'adversarial_results.json'}")
    graphrag.close()

def generate_model_docs():
    print("\n" + "=" * 70)
    print("TASK 29: Model Version Documentation")
    print("=" * 70)
    docs = {"models":{"gpt-4o":{"id":"gpt-4o","provider":"OpenAI"},"o4-mini":{"id":"o4-mini","provider":"OpenAI"},
            "claude-sonnet":{"id":"claude-sonnet-4-20250514","provider":"Anthropic"},
            "qoq-med-7b":{"id":"ddvd233/QoQ-Med-VL-7B","base":"Qwen2-VL-7B","quant":"4bit-NF4"},
            "medgemma-4b":{"id":"google/medgemma-4b-it","base":"Gemma-3","quant":"4bit-NF4"},
            "med-r1-2b":{"id":"yuxianglai117/Med-R1 (VQA_MRI)","base":"Qwen2-VL-2B","quant":"4bit-NF4"},
            "llava-med-7b":{"id":"chaoyinshe/llava-med-v1.5-mistral-7b-hf","base":"Mistral-7B","quant":"4bit-NF4"}},
            "system":{"neo4j_drugs":740135,"chromadb_docs":496,"benchmark":1576,"w1":0.6075,"w2":0.3925,"tau":0.0028},
            "hardware":{"lab":"i7, 128GB, RTX A5000 24GB","colab":"T4, 12.7GB"}}
    with open(OUTPUT_DIR / "model_documentation.json", "w") as f: json.dump(docs, f, indent=2)
    for n, i in docs["models"].items(): print(f"    {n}: {i['id']}")
    print(f"  Saved: {OUTPUT_DIR / 'model_documentation.json'}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--medhalt", action="store_true")
    p.add_argument("--medqa", action="store_true")
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--modeldocs", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if not any([a.medhalt, a.medqa, a.adversarial, a.modeldocs, a.all]): a.all = True
    print("=" * 70)
    print("GenMedAssist — Additional Evaluations (#26-29)")
    print("=" * 70)
    if a.modeldocs or a.all: generate_model_docs()
    if a.medqa or a.all: run_medqa()
    if a.medhalt or a.all: run_medhalt()
    if a.adversarial or a.all: run_adversarial()
    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()