"""Neural network blocks: critics and simplex policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Dirichlet

Tensor = torch.Tensor


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DirichletPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def _concentration(self, state: Tensor, beta: Tensor) -> Tensor:
        if beta.dim() == 1:
            beta = beta.unsqueeze(-1)
        payload = torch.cat([state, beta], dim=-1)
        logits = self.net(payload)
        return torch.nn.functional.softplus(logits) + 1e-3

    def sample(self, state: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
        conc = self._concentration(state, beta)
        dist = Dirichlet(conc)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob.unsqueeze(-1)

    def deterministic(self, state: Tensor, beta: Tensor) -> Tensor:
        conc = self._concentration(state, beta)
        return conc / conc.sum(dim=-1, keepdim=True)
