import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('csweeps', nargs='+')
p.add_argument('--T', type=float, help='temperature, for csweeps that do not store it')
p.add_argument('--out')
args = p.parse_args()

summary = {}
for path in args.csweeps:
    d = json.load(open(path, encoding='utf-8'))
    rows = [r for r in d.get('ood_bound_check', []) if 'sigma_g_max' in r]
    if not rows:
        continue
    T = d.get('T', args.T)
    if T is None:
        raise SystemExit(f'{path} does not store T, pass --T')
    slope_bound = max(r['sigma_g_max'] for r in rows) / T

    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r['prompt']].append(r)

    measure = 'step_linf' if any('step_linf' in r for r in rows) else 'linf_difference'
    slopes = []
    for prompt_rows in by_prompt.values():
        prompt_rows.sort(key=lambda r: r['c'])
        for a, b in zip(prompt_rows, prompt_rows[1:]):
            gap = b['step_linf'] if measure == 'step_linf' else abs(b['residual_linf'] - a['residual_linf'])
            slopes.append(gap / (b['c'] - a['c']))

    summary[Path(path).stem] = {
        'measure': measure,
        'slope_bound': slope_bound,
        'max_slope': max(slopes),
        'pairs': len(slopes),
        'violations': sum(s > slope_bound + 1e-6 for s in slopes),
    }

print(f'{"run":48s} {"measure":16s} {"bound":>7s} {"max":>7s}  violations')
for name, s in summary.items():
    print(f'{name[:48]:48s} {s["measure"]:16s} {s["slope_bound"]:7.4f} {s["max_slope"]:7.4f}  {s["violations"]}/{s["pairs"]}')
total = sum(s['pairs'] for s in summary.values())
print(f'\n{total} adjacent pairs, {sum(s["violations"] for s in summary.values())} violations')

if args.out:
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
