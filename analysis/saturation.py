import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('csweeps', nargs='+')
p.add_argument('--out')
args = p.parse_args()

table = {}
for path in args.csweeps:
    rows = json.load(open(path, encoding='utf-8')).get('ood_bound_check', [])
    rows = [r for r in rows if 'sigma_g_max' in r]
    if not rows:
        continue
    bound = max(r['sigma_g_max'] for r in rows)
    by_c = defaultdict(list)
    for r in rows:
        by_c[float(r['c'])].append(r['residual_linf'])
    mean_linf = {c: sum(v) / len(v) for c, v in sorted(by_c.items())}
    table[Path(path).stem] = {
        'sigma_g_max': bound,
        'mean_linf_vs_c': mean_linf,
        'ratio': max(mean_linf.values()) / bound,
        'max_linf': max(r['residual_linf'] for r in rows),
        'within_per_dim_bound': all(r['within_per_dim_bound'] for r in rows),
    }

print(f'{"run":48s} {"sigma_g_max":>11s} {"ratio":>7s} {"max_linf":>9s}  within')
for name, t in table.items():
    print(f'{name[:48]:48s} {t["sigma_g_max"]:11.4f} {t["ratio"]:7.3f} {t["max_linf"]:9.4f}  {t["within_per_dim_bound"]}')

if args.out:
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(table, f, indent=2)
