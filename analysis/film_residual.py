import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from dial.heads import FiLMHead

p = argparse.ArgumentParser()
p.add_argument('heads', nargs='+', help='FiLM head.pt files')
p.add_argument('--h-norm', type=float, required=True, help='L2 norm of the probe hidden state')
p.add_argument('--c', default='0,1,2,5,10,100,1000,10000')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--dial-csweep', help='DIAL csweep to compare against')
p.add_argument('--out')
args = p.parse_args()

c_values = [float(c) for c in args.c.split(',')]
table = {}

for path in args.heads:
    state = torch.load(path, map_location='cpu')
    d = state['gamma_head.weight'].shape[0]
    head = FiLMHead(d)
    head.load_state_dict(state)
    gen = torch.Generator().manual_seed(args.seed)
    h = torch.randn(1, 1, d, generator=gen)
    h = h * args.h_norm / h.norm()
    with torch.no_grad():
        table[path] = {c: (head(h, torch.tensor([c])) - h).abs().max().item() for c in c_values}

if args.dial_csweep:
    rows = json.load(open(args.dial_csweep, encoding='utf-8'))['ood_bound_check']
    by_c = defaultdict(list)
    for r in rows:
        by_c[float(r['c'])].append(r['residual_linf'])
    table[args.dial_csweep] = {c: sum(v) / len(v) for c, v in by_c.items() if c in c_values}

names = [Path(k).stem if k.endswith('.json') else Path(k).parent.name for k in table]
print(f'{"c":>8s}  ' + '  '.join(f'{n[:22]:>22s}' for n in names))
for c in c_values:
    print(f'{c:8g}  ' + '  '.join(f'{col.get(c, float("nan")):22.4f}' for col in table.values()))

if args.out:
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({k: {str(c): v for c, v in col.items()} for k, col in table.items()}, f, indent=2)
