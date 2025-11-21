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
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.hebb_norm = nn.LayerNorm(hebb_dim)
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
        self._log_feature_norm = 0
        self._log_trace = 0

    def forward(self, features: Tensor, hebb: Tensor) -> Tensor:
        feat_normed = self.feature_norm(features)
        if self._log_feature_norm < 5:
            print(f"[risk] LayerNorm active | feat_mean={float(feat_normed.mean().item()):.6f}", flush=True)
            self._log_feature_norm += 1
        hebb_normed = self.hebb_norm(hebb)
        feat_latent = self.feature_branch(feat_normed)
        hebb_latent = self.hebb_branch(hebb_normed)
        merged = torch.cat([feat_latent, hebb_latent], dim=-1)
        return self.merge(merged)


class PlasticityHead(nn.Module):
    def __init__(self, latent_dim: int, hebb_dim: int, beta_min: float, beta_max: float):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hebb_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2),
        )
        self.beta_scale = 0.02

    def forward(self, latent: Tensor, hebb: Tensor) -> Tuple[Tensor, Tensor]:
        payload = torch.cat([latent, hebb], dim=-1)
        raw = self.net(payload)
        beta_offset = torch.tanh(raw[:, :1]) * self.beta_scale
        beta_vector = torch.sigmoid(raw[:, 1:])
        return beta_offset, beta_vector


class RiskModule(nn.Module):
    def __init__(self, feature_dim: int, hebb_dim: int, cfg: TrainingConfig):
        super().__init__()
        self.encoder = HebbianRiskEncoder(feature_dim, hebb_dim)
        self.head = PlasticityHead(128, hebb_dim, cfg.beta_min, cfg.beta_max)
        self.trace_encoder = nn.Sequential(
            nn.Linear(feature_dim, hebb_dim),
            nn.Tanh(),
        )
        self.cfg = cfg
        self.hebb_dim = hebb_dim
        self.register_buffer("hebb_trace", torch.zeros(hebb_dim))
        self.register_buffer("beta_center", torch.tensor(cfg.homeostat_target))
        self.decay = 0.7
        self.learning_rate = 0.25
        self.homeostat_tau = 0.1
        self.last_beta_raw: Tensor | None = None

    def observe(self, features: Tensor, signal: Tensor) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray]:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        signal = signal.view(1, 1)
        hebb_snapshot = self.hebb_trace.detach().clone()
        latent = self.encoder(features, hebb_snapshot.unsqueeze(0))
        beta_raw, beta_vec = self.head(latent, hebb_snapshot.unsqueeze(0))
        beta_val = self._combine_beta(beta_raw)
        self.last_beta_raw = beta_raw.detach()
        feature_proj = self.trace_encoder(features).squeeze(0)
        if self.encoder._log_trace < 5:
            print(
                f"[risk] trace_encoder active | trace_mean={float(feature_proj.mean().item()):.6f}",
                flush=True,
            )
            self.encoder._log_trace += 1
        update = self.decay * hebb_snapshot + self.learning_rate * (feature_proj * signal.squeeze(0))
        self.hebb_trace.copy_(update.detach().clamp(-5.0, 5.0))
        next_hebb = self.hebb_trace.detach().clone()
        self.update_homeostat(beta_val.detach())
        return beta_val.squeeze(0), beta_vec.squeeze(0), hebb_snapshot.cpu().numpy(), next_hebb.cpu().numpy()

    def encode_batch(self, features: Tensor, hebb: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder(features, hebb)
        beta_raw, beta_vec = self.head(latent, hebb)
        beta_val = self._combine_beta(beta_raw)
        self.last_beta_raw = beta_raw.detach()
        return beta_val, beta_vec

    def _combine_beta(self, offset: Tensor) -> Tensor:
        center = self.beta_center.view(1, 1)
        beta = center + offset
        beta = torch.clamp(beta, self.cfg.beta_min, self.cfg.beta_max)
        return beta

    def update_homeostat(self, beta_val: Tensor) -> None:
        before = float(self.beta_center.item())
        beta_mean = float(beta_val.mean().item())
        print(
            f"[risk] update_homeostat | beta_raw={beta_mean:.6f} homeostat_before={before:.6f}",
            flush=True,
        )
        delta = beta_val.mean() - self.beta_center
        new_center = self.beta_center + self.homeostat_tau * delta
        new_center = torch.clamp(new_center, self.cfg.beta_min, self.cfg.beta_max)
        self.beta_center.copy_(new_center.detach())
        after = float(self.beta_center.item())
        print(f"[risk] homeostat_after={after:.6f}", flush=True)
