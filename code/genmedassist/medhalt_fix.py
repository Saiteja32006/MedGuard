from datasets import load_dataset
import json
from pathlib import Path

OUT = Path("additional_eval")
OUT.mkdir(exist_ok=True)
results = {}

for cfg in ["reasoning_FCT", "reasoning_nota", "reasoning_fake"]:
    print(f"Loading {cfg}...")
    ds = load_dataset("openlifescienceai/Med-HALT", cfg)
    for split in ds:
        n = len(ds[split])
        results[f"{cfg}_{split}"] = n
        print(f"  {cfg}/{split}: {n} examples")
        if n > 0:
            print(f"  Sample keys: {list(ds[split][0].keys())}")

json.dump(results, open(OUT / "medhalt_hf_info.json", "w"), indent=2)
print(f"\nDone! Saved: {OUT / 'medhalt_hf_info.json'}")