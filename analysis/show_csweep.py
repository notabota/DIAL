import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('csweep')
p.add_argument('--prompt', type=int, default=0)
p.add_argument('--kind', default='harmful', choices=['harmful', 'helpful'])
args = p.parse_args()

row = json.load(open(args.csweep, encoding='utf-8'))[args.kind][args.prompt]
print(row['prompt'], '\n')
for c, out in row['per_c'].items():
    print(f'c={c:>4}  {out["gen"][:110].replace(chr(10), " ")}')
