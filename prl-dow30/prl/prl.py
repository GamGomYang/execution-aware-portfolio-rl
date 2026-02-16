from __future__ import annotations

from dataclasses import dataclass
import logging
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
    center_prob: bool = True
    emergency_mode: str = "clamp"
    emergency_vz_threshold: float = 2.0
    mid_plasticity_multiplier: float = 1.0
    # Optional risk penalty knobs (currently logged only)
    var_penalty_beta: float | None = None
    cvar_penalty_gamma: float | None = None
    penalty_clip_ratio: float = 0.2


@dataclass
class AlphaDiagnostics:
    prl_prob: th.Tensor
    vz: th.Tensor
    alpha_raw: th.Tensor
    alpha_clamped: th.Tensor
    emergency: th.Tensor
    beta_effective: th.Tensor


class PRLAlphaScheduler:
    """Volatility-only policy regularization schedule."""

    def __init__(self, cfg: PRLConfig):
        self.cfg = cfg
        start = cfg.window_size * cfg.num_assets
        self.vol_slice = slice(start, start + cfg.num_assets)
        self.vol_mean = th.tensor(cfg.vol_mean, dtype=th.float32)
        self.vol_std = th.tensor(cfg.vol_std, dtype=th.float32)
        self.mid_multiplier = float(cfg.mid_plasticity_multiplier)
        self._logged_mid_multiplier = False

    def _prepare_obs(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs

    def _compute_vz(self, obs: th.Tensor) -> th.Tensor:
        obs = self._prepare_obs(obs)
        vol_vector = obs[:, self.vol_slice]
        V = th.mean(vol_vector, dim=1, keepdim=True)
        vol_mean = self.vol_mean.to(obs.device)
        vol_std = self.vol_std.to(obs.device)
        return (V - vol_mean) / (vol_std + 1e-8)

    def prl_probability(self, obs: th.Tensor) -> th.Tensor:
        Vz = self._compute_vz(obs)
        return th.sigmoid(self.cfg.lambdav * Vz + self.cfg.bias)

    def alpha_from_obs(self, obs: th.Tensor, return_diagnostics: bool = False):
        eps = 1e-8
        Vz = self._compute_vz(obs)
        P = th.sigmoid(self.cfg.lambdav * Vz + self.cfg.bias)
        if self.cfg.center_prob:
            p0 = th.sigmoid(th.tensor(self.cfg.bias, dtype=P.dtype, device=P.device))
            P_centered = P - p0
        else:
            P_centered = P
        # Apply mid-plasticity multiplier when Vz is near the mid regime (|Vz| <= 0.5 approximates mid vol).
        mid_mask = th.abs(Vz) <= 0.5
        multiplier = th.where(mid_mask, th.tensor(self.mid_multiplier, dtype=P.dtype, device=P.device), th.tensor(1.0, dtype=P.dtype, device=P.device))
        if not self._logged_mid_multiplier and mid_mask.any():
            logging.getLogger(__name__).info(
                "PRL mid_plasticity_multiplier applied: multiplier=%.3f mid_fraction=%.3f",
                self.mid_multiplier,
                float(mid_mask.float().mean().item()),
            )
            self._logged_mid_multiplier = True

        alpha_raw = self.cfg.alpha0 * (1 + self.cfg.beta * multiplier * P_centered)
        alpha_clamped = th.clamp(alpha_raw, min=self.cfg.alpha_min, max=self.cfg.alpha_max)

        if self.cfg.emergency_mode == "vz":
            emergency = Vz >= self.cfg.emergency_vz_threshold
        elif self.cfg.emergency_mode == "clamp":
            emergency = (alpha_raw != alpha_clamped) | (alpha_raw < self.cfg.alpha_min) | (alpha_raw > self.cfg.alpha_max)
        else:
            raise ValueError(f"Unknown emergency_mode: {self.cfg.emergency_mode}")

        beta_effective = th.zeros_like(P)
        mask = th.abs(P_centered) >= eps
        if mask.any():
            beta_effective[mask] = (alpha_clamped[mask] / self.cfg.alpha0 - 1.0) / P_centered[mask]

        if return_diagnostics:
            diagnostics = AlphaDiagnostics(
                prl_prob=P,
                vz=Vz,
                alpha_raw=alpha_raw,
                alpha_clamped=alpha_clamped,
                emergency=emergency,
                beta_effective=beta_effective,
            )
            return alpha_clamped, diagnostics
        return alpha_clamped
