"""Risk encoder, Hebbian trace, and metaplasticity head."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig

Tensor = torch.Tensor


class HebbianRiskEncoder(nn.Module):
    def __init__(self, feature_dim: int, hebb_dim: int, latent_dim: int = 128):
        super().__init__()
        branch_dim = latent_dim // 2
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, branch_dim),
            nn.GELU(),
        )
        self.hebb_branch = nn.Sequential(
            nn.Linear(hebb_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )
        self.merge = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, features: Tensor, hebb: Tensor) -> Tensor:
        feat_latent = self.feature_branch(features)
        hebb_latent = self.hebb_branch(hebb)
        merged = torch.cat([feat_latent, hebb_latent], dim=-1)
        return self.merge(merged)


class PlasticityHead(nn.Module):
    def __init__(self, latent_dim: int, hebb_dim: int, beta_min: float, beta_max: float):
        super().__init__()
        self.beta_min = beta_min
        self.beta_range = beta_max - beta_min
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hebb_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2),
        )

    def forward(self, latent: Tensor, hebb: Tensor) -> Tuple[Tensor, Tensor]:
        payload = torch.cat([latent, hebb], dim=-1)
        raw = self.net(payload)
        beta_scalar = torch.sigmoid(raw[:, :1])
        beta_vector = torch.sigmoid(raw[:, 1:])
        beta_val = self.beta_min + self.beta_range * beta_scalar
        return beta_val, beta_vector


class RiskModule(nn.Module):
    def __init__(self, feature_dim: int, hebb_dim: int, cfg: TrainingConfig):
        super().__init__()
        self.encoder = HebbianRiskEncoder(feature_dim, hebb_dim)
        self.head = PlasticityHead(128, hebb_dim, cfg.beta_min, cfg.beta_max)
        self.cfg = cfg
        self.hebb_dim = hebb_dim
        self.register_buffer("hebb_trace", torch.zeros(hebb_dim))
        self.register_buffer("beta_center", torch.tensor(cfg.homeostat_target))
        self.decay = 0.9
        self.learning_rate = 0.1

    def observe(self, features: Tensor, signal: Tensor) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray]:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        signal = signal.view(1, 1)
        hebb_snapshot = self.hebb_trace.detach().clone()
        latent = self.encoder(features, hebb_snapshot.unsqueeze(0))
        beta_val, beta_vec = self.head(latent, hebb_snapshot.unsqueeze(0))
        update = self.decay * hebb_snapshot + self.learning_rate * (features.squeeze(0) * signal.squeeze(0))
        self.hebb_trace.copy_(update.detach().clamp(-5.0, 5.0))
        next_hebb = self.hebb_trace.detach().clone()
        self.update_homeostat(beta_val.detach())
        return beta_val.squeeze(0), beta_vec.squeeze(0), hebb_snapshot.cpu().numpy(), next_hebb.cpu().numpy()

    def encode_batch(self, features: Tensor, hebb: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder(features, hebb)
        beta_val, beta_vec = self.head(latent, hebb)
        return beta_val, beta_vec

    def update_homeostat(self, beta_val: Tensor) -> None:
        new_center = (1 - self.cfg.homeostat_tau) * self.beta_center + self.cfg.homeostat_tau * beta_val.mean()
        self.beta_center.copy_(new_center.detach())
