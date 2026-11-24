import json
from pathlib import Path

eval_dir = Path('eval_results')

print("\n" + "="*40)
print(" EVALUATION RESULTS COUNT")
print("="*40)

if not eval_dir.exists():
    print(f"Directory '{eval_dir}' not found.")
else:
    for f in eval_dir.glob('*_results.jsonl'):
        # Open and count lines efficiently
        with open(f, encoding='utf-8') as file:
            line_count = sum(1 for line in file)
        
        # Clean up the name for a nicer display
        name = f.stem.replace('_results', '')
        print(f"  • {name}: {line_count} results")

print("="*40 + "\n")