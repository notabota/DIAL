import argparse
import json
import os
import time

from datasets import load_dataset

from dial.model import ARMS, DEFAULT_MODEL, chat, load_steered

SAFETY = {
    'harmbench': ('walledai/HarmBench', 'standard', 'train'),
    'advbench': ('walledai/AdvBench', None, 'train'),
    'xstest': ('walledai/XSTest', None, 'test'),
}

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True, choices=ARMS)
p.add_argument('--adapter')
p.add_argument('--head')
p.add_argument('--caa', help='caa_direction.pt, for --arm caa')
p.add_argument('--caa-layer', type=int, default=16)
p.add_argument('--model', default=DEFAULT_MODEL)
p.add_argument('--out', required=True)
p.add_argument('--benches', default='harmbench,advbench,xstest')
p.add_argument('--c', default='-5,0,5', help='control values for the safety benchmarks')
p.add_argument('--n', type=int, default=100)
p.add_argument('--mmlu-n', type=int, default=200)
p.add_argument('--max-new', type=int, default=100)
p.add_argument('--T', type=float, default=5.0)
p.add_argument('--clamp', type=float, default=0.5)
p.add_argument('--seed', type=int, default=42, help='dataset shuffle seed')
args = p.parse_args()

benches = args.benches.split(',')
unknown = set(benches) - set(SAFETY) - {'mmlu'}
if unknown:
    p.error(f'unknown benchmarks: {sorted(unknown)}')
c_values = [float(c) for c in args.c.split(',')]

if args.arm == 'caa':
    if not args.caa:
        p.error('--arm caa needs --caa')
    lm = load_steered('caa', args.model, caa_path=args.caa, caa_layer=args.caa_layer)
else:
    if not args.adapter:
        p.error(f'--arm {args.arm} needs --adapter')
    lm = load_steered(args.arm, args.model, adapter=args.adapter, head_path=args.head, T=args.T, cap=args.clamp)

os.makedirs(args.out, exist_ok=True)


def run_safety(name):
    path, config, split = SAFETY[name]
    ds = load_dataset(path, config, split=split).shuffle(seed=args.seed).select(range(args.n))
    rows = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        per_c = {str(c): {'gen': lm.generate(chat(ex['prompt']), c, args.max_new)[:400]} for c in c_values}
        row = {'prompt': ex['prompt'][:200], 'per_c': per_c}
        for field in ('category', 'type', 'label'):
            if field in ex:
                row[field] = ex[field]
        rows.append(row)
        if i % 10 == 0:
            print(f'{name} {i + 1}/{len(ds)}  {time.time() - t0:.0f}s', flush=True)
    return {'results': rows}


def run_mmlu():
    ds = load_dataset('cais/mmlu', 'all', split='test').shuffle(seed=args.seed).select(range(args.mmlu_n))
    rows = []
    for ex in ds:
        prompt = f"Question: {ex['question']}\n"
        prompt += ''.join(f'{"ABCD"[j]}. {choice}\n' for j, choice in enumerate(ex['choices']))
        prompt += 'Answer:'
        gen = lm.generate(prompt, 0.0, 5).strip().upper()
        pred = next((x for x in 'ABCD' if x in gen[:3]), None)
        gold = 'ABCD'[ex['answer']]
        rows.append({'question': ex['question'][:100], 'pred': pred, 'gold': gold, 'correct': pred == gold})
    return {'accuracy': sum(r['correct'] for r in rows) / len(rows), 'n': len(rows), 'per_question': rows}


for name in benches:
    out = os.path.join(args.out, f'{name}.json')
    if os.path.exists(out):
        print(f'{out} exists, skipping')
        continue
    result = run_mmlu() if name == 'mmlu' else run_safety(name)
    with open(out, 'w') as f:
        json.dump(result, f)
    print(f'wrote {out}')
