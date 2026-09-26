import torch
import torch.nn as nn


class DIALHead(nn.Module):
    def __init__(self, d, T=5.0):
        super().__init__()
        self.intensity_gain = nn.Parameter(torch.zeros(d))
        self.direction_head = nn.Linear(d, d)
        nn.init.normal_(self.direction_head.weight, std=0.02)
        nn.init.zeros_(self.direction_head.bias)
        self.T = T

    def gate(self):
        return torch.sigmoid(self.intensity_gain)

    def forward(self, h, c):
        strength = torch.tanh(c / self.T).view(-1, 1, 1)
        return self.gate() * strength * torch.tanh(self.direction_head(h))


class _FiLMBase(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gamma_head = nn.Linear(1, d)
        self.beta_head = nn.Linear(1, d)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def modulate(self, h, c):
        c = c.view(-1, 1)
        return self.gamma_head(c).unsqueeze(1) * h + self.beta_head(c).unsqueeze(1)


class FiLMHead(_FiLMBase):
    def forward(self, h, c):
        return self.modulate(h, c)


class FiLMTanhHead(_FiLMBase):
    def forward(self, h, c):
        return torch.tanh(self.modulate(h, c) - h)


class FiLMClampHead(_FiLMBase):
    def __init__(self, d, cap):
        super().__init__(d)
        self.cap = cap

    def forward(self, h, c):
        return torch.clamp(self.modulate(h, c) - h, -self.cap, self.cap)


def make_head(arm, d, T=5.0, cap=0.5):
    if arm == 'dial':
        return DIALHead(d, T)
    if arm == 'film':
        return FiLMHead(d)
    if arm == 'film_tanh':
        return FiLMTanhHead(d)
    if arm == 'film_clamp':
        return FiLMClampHead(d, cap)
    raise ValueError(arm)
