import argparse
import json
import os

import torch
from datasets import load_dataset

from dial.model import DEFAULT_MODEL, load_base

p = argparse.ArgumentParser()
p.add_argument('--model', default=DEFAULT_MODEL)
p.add_argument('--out', required=True)
p.add_argument('--layer', type=int, default=16)
p.add_argument('--pairs', type=int, default=1000)
p.add_argument('--max-len', type=int, default=384)
args = p.parse_args()

tok, model = load_base(args.model)
model.eval()
device = model.lm_head.weight.device
ds = load_dataset('Anthropic/hh-rlhf', split='train').select(range(args.pairs))


@torch.no_grad()
def last_token_state(text):
    ids = tok(text, return_tensors='pt', truncation=True, max_length=args.max_len).input_ids.to(device)
    states = model(input_ids=ids, output_hidden_states=True).hidden_states
    return states[args.layer + 1][0, -1].float().cpu()


diffs = []
for i, ex in enumerate(ds):
    diffs.append(last_token_state(ex['chosen']) - last_token_state(ex['rejected']))
    if i % 100 == 0:
        print(f'pair {i}/{args.pairs}', flush=True)

direction = torch.stack(diffs).mean(0)
os.makedirs(args.out, exist_ok=True)
torch.save({'direction': direction, 'direction_unit': direction / direction.norm(), 'layer': args.layer},
           os.path.join(args.out, 'caa_direction.pt'))
with open(os.path.join(args.out, 'caa_report.json'), 'w') as f:
    json.dump({**vars(args), 'direction_norm': direction.norm().item()}, f, indent=2)
