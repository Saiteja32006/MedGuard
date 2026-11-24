import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score

eval_dir = Path('eval_results')
models = {}

# Load all results
for f in eval_dir.glob('*_results.jsonl'):
    name = f.stem.replace('_results', '')
    preds = [json.loads(l) for l in open(f, encoding='utf-8')]
    if not preds or 'stratum' not in preds[0]:
        continue
    models[name] = preds
    print(f'{name}: {len(preds)} results')

print('\n' + '='*70)
print('PERFORMANCE CLIFF — Statistical Significance')
print('='*70)

for name, preds in sorted(models.items()):
    s1 = [p for p in preds if p.get('stratum') == 1]
    s2 = [p for p in preds if p.get('stratum') == 2]
    if not s1 or not s2:
        continue
    s1_det = sum(1 for p in s1 if p.get('pred_conflict')) / len(s1)
    s2_det = sum(1 for p in s2 if p.get('pred_conflict')) / len(s2)
    cliff = s1_det - s2_det

    # Two-proportion z-test
    s1_n, s2_n = len(s1), len(s2)
    s1_k = sum(1 for p in s1 if p.get('pred_conflict'))
    s2_k = sum(1 for p in s2 if p.get('pred_conflict'))
    p_pool = (s1_k + s2_k) / (s1_n + s2_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/s1_n + 1/s2_n)) if p_pool > 0 else 1
    z = cliff / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boots = []
    s1_arr = np.array([1 if p.get('pred_conflict') else 0 for p in s1])
    s2_arr = np.array([1 if p.get('pred_conflict') else 0 for p in s2])
    for _ in range(10000):
        b1 = rng.choice(s1_arr, size=len(s1_arr), replace=True).mean()
        b2 = rng.choice(s2_arr, size=len(s2_arr), replace=True).mean()
        boots.append(b1 - b2)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    sig = 'YES ***' if p_val < 0.001 else 'YES **' if p_val < 0.01 else 'YES *' if p_val < 0.05 else 'NO'
    print(f'\n  {name}:')
    print(f'    S1={s1_det:.3f}, S2={s2_det:.3f}, Cliff={cliff:+.3f}')
    print(f'    z={z:.3f}, p={p_val:.6f}, Significant: {sig}')
    print(f'    95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]')

# Also print the full comparison table
print('\n' + '='*70)
print('FULL RESULTS TABLE')
print('='*70)
print(f"{'Model':<20} {'S0-Spec':<10} {'S1-Cat':<10} {'S2-Num':<10} {'S3-MH':<10} {'Cliff':<10} {'F1':<8}")
print('-'*68)
for name, preds in sorted(models.items()):
    s0 = [p for p in preds if p.get('stratum')==0]
    s1 = [p for p in preds if p.get('stratum')==1]
    s2 = [p for p in preds if p.get('stratum')==2]
    s3 = [p for p in preds if p.get('stratum')==3]

    spec = f'{1-sum(1 for p in s0 if p.get("pred_conflict"))/max(len(s0),1):.3f}' if s0 else '-'
    d1 = f'{sum(1 for p in s1 if p.get("pred_conflict"))/max(len(s1),1):.3f}' if s1 else '-'
    d2 = f'{sum(1 for p in s2 if p.get("pred_conflict"))/max(len(s2),1):.3f}' if s2 else '-'
    d3 = f'{sum(1 for p in s3 if p.get("pred_conflict"))/max(len(s3),1):.3f}' if s3 else '-'

    if s1 and s2:
        r1 = sum(1 for p in s1 if p.get('pred_conflict'))/len(s1)
        r2 = sum(1 for p in s2 if p.get('pred_conflict'))/len(s2)
        cliff_str = f'{r1-r2:+.3f}'
    else:
        cliff_str = '-'

    yt = [1 if p.get('gt_conflict') else 0 for p in preds]
    yp = [1 if p.get('pred_conflict') else 0 for p in preds]
    f1 = f'{f1_score(yt, yp, zero_division=0):.3f}'

    print(f'{name:<20} {spec:<10} {d1:<10} {d2:<10} {d3:<10} {cliff_str:<10} {f1:<8}')