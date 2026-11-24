import json

# Load the dataset
data = [json.loads(l) for l in open('medconflict_bench_v2.jsonl', encoding='utf-8')]

drugs = set()
labs = set()

# Extract unique drugs and labs
for ex in data:
    d = ex.get('drug', ex.get('drugs', ''))
    if isinstance(d, list):
        for x in d: 
            drugs.add(x.lower().strip())
    elif d: 
        drugs.add(d.lower().strip())
        
    lab = ex.get('lab_name', '')
    if lab: 
        labs.add(lab)

# Sort the lists alphabetically
sorted_drugs = sorted(list(drugs))
sorted_labs = sorted(list(labs))

# Print highly readable formatted results
print("\n" + "="*50)
print(" MEDICATION & LAB SUMMARY")
print("="*50)
print(f"Total Unique Drugs:      {len(drugs)}")
print(f"Total Unique Lab Tests:  {len(labs)}")
print(f"Rules Breakdown:         12 categorical + 16 numerical + 8 multi-hop = 36 total")

print("\n--- Lab Tests ---")
for lab in sorted_labs:
    print(f"  • {lab}")

print("\n--- Drug Examples (First 20) ---")
for drug in sorted_drugs[:20]:
    print(f"  • {drug.title()}") # .title() capitalizes the first letters for better readability
print("="*50 + "\n")
