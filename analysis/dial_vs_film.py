import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument('dial', help='trained DIAL head checkpoint')
p.add_argument('film', help='trained FiLM head checkpoint')
p.add_argument('--T', type=float, default=5.0)
p.add_argument('--h-scale', type=float, default=0.5)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

dial = torch.load(args.dial, map_location='cpu')
film = torch.load(args.film, map_location='cpu')

gate = torch.sigmoid(dial['intensity_gain'])
W, b = dial['direction_head.weight'], dial['direction_head.bias']


def dial_delta(h, c):
    return gate * torch.tanh(torch.tensor(c / args.T)) * torch.tanh(W @ h + b)


def film_delta(h, c):
    gamma = film['gamma_head.weight'][:, 0] * c + film['gamma_head.bias']
    beta = film['beta_head.weight'][:, 0] * c + film['beta_head.bias']
    return gamma * h + beta - h


torch.manual_seed(args.seed)
h = torch.randn(W.shape[1]) * args.h_scale

print(f'DIAL bound read from weights: sigmoid(g_max) = {gate.max():.4f}\n')
print(f'{"c":>12}  {"DIAL max|delta|":>16}  {"FiLM max|delta|":>16}')
for c in [0, 1, 5, 10, 100, 1e4, 1e8]:
    print(f'{c:>12g}  {dial_delta(h, c).abs().max():>16.4f}  {film_delta(h, c).abs().max():>16.4f}')
