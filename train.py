import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from dial.heads import make_head
from dial.model import ARMS, DEFAULT_MODEL, HEAD_ARMS, SteeredLM, load_base

LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True, choices=[a for a in ARMS if a != 'caa'])
p.add_argument('--model', default=DEFAULT_MODEL)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=5000)
p.add_argument('--samples', type=int, default=16000)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--max-len', type=int, default=384)
p.add_argument('--lora-r', type=int, default=8)
p.add_argument('--T', type=float, default=5.0)
p.add_argument('--clamp', type=float, default=0.5, help='cap for film_clamp')
p.add_argument('--gate-cap', type=float, help='upper bound on sigmoid(g) for the dial arm')
p.add_argument('--freeze-gate', action='store_true')
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

if (args.gate_cap or args.freeze_gate) and args.arm != 'dial':
    p.error('--gate-cap and --freeze-gate are for --arm dial')

os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)

tok, base = load_base(args.model)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
model = get_peft_model(base, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
    target_modules=LORA_TARGETS, lora_dropout=0.0, bias='none',
))
model.print_trainable_parameters()

head = None
if args.arm in HEAD_ARMS:
    head = make_head(args.arm, base.config.hidden_size, args.T, args.clamp)
    head.to(base.lm_head.weight.device)
    if args.freeze_gate:
        head.intensity_gain.requires_grad_(False)

lm = SteeredLM(model, tok, args.arm, head, args.T)
params = [q for q in model.parameters() if q.requires_grad]
if head is not None:
    params += [q for q in head.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr)
gate_logit_cap = None if args.gate_cap is None else math.log(args.gate_cap / (1 - args.gate_cap))

ds = load_dataset('Anthropic/hh-rlhf', split='train').select(range(args.samples))


def lm_loss(text, c):
    enc = tok(text, max_length=args.max_len, truncation=True, padding='max_length', return_tensors='pt')
    ids = enc.input_ids.to(lm.device)
    mask = enc.attention_mask.to(lm.device)
    logits, _ = lm.logits(ids, torch.tensor([c], device=lm.device), attention_mask=mask)
    labels = ids.masked_fill(mask == 0, -100)
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)


model.train()
losses = []
t0 = time.time()
for step in range(args.steps):
    ex = ds[step % args.samples]
    if args.arm == 'lora':
        loss = lm_loss(ex['chosen'], 0.0)
    else:
        loss = 0.5 * (lm_loss(ex['chosen'], 1.0) + lm_loss(ex['rejected'], -1.0))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if gate_logit_cap is not None:
        with torch.no_grad():
            head.intensity_gain.clamp_(max=gate_logit_cap)

    losses.append(loss.item())
    if step % 25 == 0:
        print(f'step {step:5d}  loss {loss.item():.3f}  {time.time() - t0:.0f}s', flush=True)

model.save_pretrained(os.path.join(args.out, 'adapter'))
report = {**vars(args), 'loss_first': losses[0], 'loss_last': losses[-1],
          'loss_curve': losses[::max(1, len(losses) // 50)]}
if head is not None:
    torch.save(head.state_dict(), os.path.join(args.out, 'head.pt'))
if args.arm == 'dial':
    g = head.gate().detach().cpu()
    report['sigma_g'] = {'min': g.min().item(), 'max': g.max().item(), 'mean': g.mean().item()}
with open(os.path.join(args.out, 'train_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
