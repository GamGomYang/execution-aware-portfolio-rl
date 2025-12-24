from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch as th


@dataclass
class PRLConfig:
    alpha0: float
    beta: float
    lambdav: float
    bias: float
    alpha_min: float
    alpha_max: float
    vol_mean: float
    vol_std: float
    window_size: int
    num_assets: int


class PRLAlphaScheduler:
    """Volatility-only policy regularization schedule."""

    def __init__(self, cfg: PRLConfig):
        self.cfg = cfg
        start = cfg.window_size * cfg.num_assets
        self.vol_slice = slice(start, start + cfg.num_assets)
        self.vol_mean = th.tensor(cfg.vol_mean, dtype=th.float32)
        self.vol_std = th.tensor(cfg.vol_std, dtype=th.float32)

    def _prepare_obs(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs

    def prl_probability(self, obs: th.Tensor) -> th.Tensor:
        obs = self._prepare_obs(obs)
        vol_vector = obs[:, self.vol_slice]
        V = th.mean(vol_vector, dim=1, keepdim=True)
        vol_mean = self.vol_mean.to(obs.device)
        vol_std = self.vol_std.to(obs.device)
        Vz = (V - vol_mean) / (vol_std + 1e-8)
        return th.sigmoid(self.cfg.lambdav * Vz + self.cfg.bias)

    def alpha_from_obs(self, obs: th.Tensor) -> th.Tensor:
        P = self.prl_probability(obs)
        alpha = self.cfg.alpha0 * (1 + self.cfg.beta * P)
        return th.clamp(alpha, min=self.cfg.alpha_min, max=self.cfg.alpha_max)
