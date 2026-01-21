from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from .metrics import post_return_weights, turnover_l1, turnover_rebalance_l1


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
        self.current_step = self.window_size
        self.prev_weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
        self.reset_count += 1
        obs = self._get_observation()
        info: Dict[str, Any] = {"reset_count": self.reset_count}
        return obs, info

    def _portfolio_return(self, returns_t: np.ndarray) -> float:
        arithmetic_returns = np.expm1(returns_t)
        return float(np.dot(self.prev_weights, arithmetic_returns))

    def _turnover(self, weights: np.ndarray, arithmetic_returns: np.ndarray) -> float:
        w_post = post_return_weights(self.prev_weights, arithmetic_returns)
        return turnover_rebalance_l1(weights, w_post)

    def step(self, action: np.ndarray):
        z = np.clip(action, self.action_space.low, self.action_space.high)
        weights = stable_softmax(z, scale=self.cfg.logit_scale)

        if self.current_step >= len(self.returns):
            raise RuntimeError("Environment step beyond data length.")

        returns_t = self.returns.iloc[self.current_step].to_numpy(copy=False)
        step_date = self.returns.index[self.current_step]
        arithmetic_returns = np.expm1(returns_t)
        assert self.prev_weights.shape == arithmetic_returns.shape == weights.shape == (self.num_assets,)
        prev_weights = self.prev_weights.copy()
        portfolio_return = float(np.dot(prev_weights, arithmetic_returns))
        turnover = self._turnover(weights, arithmetic_returns)
        turnover_target_change = turnover_l1(weights, prev_weights)
        cost = self.cfg.transaction_cost * turnover

        log_argument = max(1.0 + portfolio_return, self.cfg.log_clip)
        reward = math.log(log_argument) - cost

        self.prev_weights = weights
        self.current_step += 1

        terminated = self.current_step >= len(self.returns)
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "turnover_rebalance": turnover,
            "turnover_target_change": turnover_target_change,
            "date": step_date,
            "log_argument": log_argument,
        }
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no rendering in v1
        return None
