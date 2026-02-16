from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from .metrics import turnover_l1


def stable_softmax(logits: np.ndarray, scale: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64) * float(scale)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(logits, 1.0 / logits.size)
    weights = exps / denom
    return weights.astype(np.float32)


@dataclass
class EnvConfig:
    returns: pd.DataFrame
    volatility: pd.DataFrame
    window_size: int
    transaction_cost: float
    log_clip: float = 1e-8
    logit_scale: float = 10.0
    random_reset: bool = False
    risk_lambda: float = 0.0
    risk_penalty_type: str = "r2"
    rebalance_eta: Optional[float] = None


class Dow30PortfolioEnv(Env):
    """Gymnasium environment for Dow30 PRL experiments."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.returns = cfg.returns.astype(np.float32)
        self.volatility = cfg.volatility.astype(np.float32)
        if not self.returns.index.equals(self.volatility.index):
            raise ValueError("Returns and volatility indices must match exactly.")

        self.num_assets = self.returns.shape[1]
        self.window_size = cfg.window_size

        obs_dim = self.window_size * self.num_assets + 2 * self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)

        self.current_step = self.window_size
        self.prev_weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
        self.reset_count = 0

        assert self.observation_space.shape[0] == obs_dim, "Observation dimension mismatch"
        assert self.window_size > 0, "window_size must be > 0"
        assert cfg.transaction_cost >= 0, "transaction cost must be non-negative"
        assert cfg.logit_scale is not None, "logit_scale must be set"
        if cfg.risk_penalty_type != "r2":
            raise ValueError(f"Unsupported risk_penalty_type: {cfg.risk_penalty_type}")
        if cfg.rebalance_eta is not None:
            eta = float(cfg.rebalance_eta)
            if not np.isfinite(eta) or eta <= 0.0 or eta > 1.0:
                raise ValueError(f"rebalance_eta must satisfy 0 < eta <= 1, got: {cfg.rebalance_eta}")

    def seed(self, seed: Optional[int] = None) -> None:  # pragma: no cover - gymnasium compatibility
        np.random.seed(seed)

    def _get_returns_window(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.returns.iloc[start:end].to_numpy(copy=True)
        return window.reshape(-1)

    def _get_vol_vector(self) -> np.ndarray:
        idx = self.current_step - 1
        return self.volatility.iloc[idx].to_numpy(copy=True)

    def _get_observation(self) -> np.ndarray:
        returns_flat = self._get_returns_window()
        vol_vector = self._get_vol_vector()
        obs = np.concatenate([returns_flat, vol_vector, self.prev_weights], dtype=np.float32)
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        if self.cfg.random_reset:
            max_start = len(self.returns) - 1
            if max_start < self.window_size:
                raise ValueError("Not enough data to random reset the environment.")
            self.current_step = int(np.random.randint(self.window_size, max_start + 1))
        else:
            self.current_step = self.window_size
        self.prev_weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
        self.reset_count += 1
        obs = self._get_observation()
        info: Dict[str, Any] = {"reset_count": self.reset_count, "start_step": self.current_step}
        return obs, info

    def _safe_normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return self.prev_weights.astype(np.float64)
        normalized = weights / total
        return normalized.astype(np.float64)

    def step(self, action: np.ndarray):
        z = np.clip(action, self.action_space.low, self.action_space.high)
        w_target = stable_softmax(z, scale=self.cfg.logit_scale).astype(np.float64)

        if self.current_step >= len(self.returns):
            raise RuntimeError("Environment step beyond data length.")

        returns_t = self.returns.iloc[self.current_step].to_numpy(copy=False)
        step_date = self.returns.index[self.current_step]
        arithmetic_returns = np.expm1(returns_t)
        prev_weights = self.prev_weights.astype(np.float64)
        assert self.prev_weights.shape == arithmetic_returns.shape == w_target.shape == (self.num_assets,)

        eta = self.cfg.rebalance_eta
        if eta is None:
            w_exec = w_target
        else:
            eta_f = float(eta)
            w_exec = (1.0 - eta_f) * prev_weights + eta_f * w_target
        w_exec = self._safe_normalize_weights(w_exec)

        turnover_target = turnover_l1(prev_weights, w_target)
        turnover_exec = turnover_l1(prev_weights, w_exec)

        portfolio_return = float(np.dot(w_exec, arithmetic_returns))
        cost = self.cfg.transaction_cost * turnover_exec

        log_argument = max(1.0 + portfolio_return, self.cfg.log_clip)
        log_return_gross = math.log(log_argument)
        log_return_net = log_return_gross - cost
        risk_lambda = float(self.cfg.risk_lambda)
        risk_penalty = risk_lambda * (portfolio_return**2)
        reward = log_return_net - risk_penalty

        self.prev_weights = w_exec.astype(np.float32)
        self.current_step += 1

        terminated = self.current_step >= len(self.returns)
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover_exec,
            "turnover_rebalance": turnover_exec,
            "turnover_target_change": turnover_target,
            "turnover_target": turnover_target,
            "turnover_exec": turnover_exec,
            "rebalance_eta": eta,
            "w_target_l1": float(np.abs(w_target).sum()),
            "w_exec_l1": float(np.abs(w_exec).sum()),
            "date": step_date,
            "cost": cost,
            "log_argument": log_argument,
            "log_return_gross": log_return_gross,
            "log_return_net": log_return_net,
            "risk_penalty": risk_penalty,
            "risk_lambda": risk_lambda,
            "reward_no_risk": log_return_net,
        }
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no rendering in v1
        return None
