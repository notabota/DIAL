import argparse
import json

import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument('judged')
p.add_argument('--dial', nargs='+', required=True)
p.add_argument('--group', action='append', default=[], metavar='NAME=ARM,ARM')
p.add_argument('--bench', default='harmbench')
p.add_argument('--judge', default='semantic', choices=['semantic', 'strict'])
p.add_argument('--lo', default='-5.0')
p.add_argument('--hi', default='5.0')
args = p.parse_args()

arms = json.load(open(args.judged, encoding='utf-8'))['arms'][args.judge]


def delta(arm, bench):
    r = arms[arm][bench]
    return r[args.hi]['refusal_rate'] - r[args.lo]['refusal_rate']


for arm, benches in arms.items():
    print(f'{arm:20s} ' + '  '.join(f'{b} {delta(arm, b):+.3f}' for b in benches))

dial = np.array([delta(a, args.bench) for a in args.dial])
print(f'\n{args.bench}, {args.judge} judge: DIAL mean {dial.mean():+.3f} (n={len(dial)})')
print(f'{"baseline":14s} {"n":>3s} {"mean":>7s} {"diff":>7s} {"d":>6s} {"p":>7s}')
for spec in args.group:
    name, members = spec.split('=', 1)
    base = np.array([delta(a, args.bench) for a in members.split(',')])
    diff = dial.mean() - base.mean()
    if len(base) > 1:
        t = stats.ttest_ind(dial, base, equal_var=False)
        sd = np.sqrt((dial.var(ddof=1) + base.var(ddof=1)) / 2)
        print(f'{name:14s} {len(base):3d} {base.mean():+7.3f} {diff:+7.3f} {diff / sd:6.2f} {t.pvalue:7.4f}')
    else:
        print(f'{name:14s} {len(base):3d} {base.mean():+7.3f} {diff:+7.3f}      -       -')
