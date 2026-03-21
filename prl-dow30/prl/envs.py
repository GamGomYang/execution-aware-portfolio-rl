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
    eta_mode: str = "legacy"
    rule_vol_window: int = 20
    rule_vol_a: float = 1.0
    eta_clip_min: float = 0.02
    eta_clip_max: float = 0.5
    signal_features: Optional[pd.DataFrame] = None


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
        self.signal_state = False
        self.signal_names: list[str] = []
        self.num_signals = 0
        self.signal_dim = 0
        self.signal_features: pd.DataFrame | None = None

        if cfg.signal_features is not None:
            signal_features = cfg.signal_features.astype(np.float32)
            if not self.returns.index.equals(signal_features.index):
                raise ValueError("Signal features and returns indices must match exactly.")
            if not isinstance(signal_features.columns, pd.MultiIndex) or signal_features.columns.nlevels != 2:
                raise ValueError("signal_features columns must be a 2-level MultiIndex: (signal_name, asset).")

            expected_assets = list(self.returns.columns)
            signal_names = list(pd.Index(signal_features.columns.get_level_values(0)).unique())
            if not signal_names:
                raise ValueError("signal_features must contain at least one signal when provided.")

            ordered_columns: list[tuple[str, str]] = []
            for signal_name in signal_names:
                signal_slice = signal_features.xs(signal_name, axis=1, level=0)
                signal_assets = list(signal_slice.columns)
                if signal_assets != expected_assets:
                    raise ValueError(
                        f"signal_features asset order mismatch for signal={signal_name}: "
                        f"expected={expected_assets}, got={signal_assets}"
                    )
                ordered_columns.extend([(signal_name, asset) for asset in expected_assets])
            signal_features = signal_features.loc[:, pd.MultiIndex.from_tuples(ordered_columns)]
            self.signal_features = signal_features
            self.signal_state = True
            self.signal_names = signal_names
            self.num_signals = len(signal_names)
            self.signal_dim = self.num_assets * self.num_signals

        obs_dim = self.window_size * self.num_assets + 2 * self.num_assets + self.signal_dim
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
        valid_eta_modes = {"legacy", "none", "fixed", "rule_vol"}
        if cfg.eta_mode not in valid_eta_modes:
            raise ValueError(f"eta_mode must be one of {valid_eta_modes}, got: {cfg.eta_mode}")
        if cfg.rebalance_eta is not None:
            eta = float(cfg.rebalance_eta)
            if not np.isfinite(eta) or eta <= 0.0 or eta > 1.0:
                raise ValueError(f"rebalance_eta must satisfy 0 < eta <= 1, got: {cfg.rebalance_eta}")
        if cfg.eta_mode == "fixed":
            if cfg.rebalance_eta is None:
                raise ValueError("rebalance_eta must be set when eta_mode is 'fixed'.")
            eta = float(cfg.rebalance_eta)
            if not np.isfinite(eta) or eta <= 0.0 or eta > 1.0:
                raise ValueError(f"rebalance_eta must satisfy 0 < eta <= 1, got: {cfg.rebalance_eta}")
        if cfg.eta_mode == "rule_vol":
            if float(cfg.rule_vol_a) < 0.0:
                raise ValueError(f"rule_vol_a must be >= 0, got: {cfg.rule_vol_a}")
            eta_clip_min = float(cfg.eta_clip_min)
            eta_clip_max = float(cfg.eta_clip_max)
            if not (0.0 < eta_clip_min <= eta_clip_max <= 1.0):
                raise ValueError(
                    "eta_clip bounds must satisfy 0 < eta_clip_min <= eta_clip_max <= 1, "
                    f"got: eta_clip_min={cfg.eta_clip_min}, eta_clip_max={cfg.eta_clip_max}"
                )

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

    def _get_signal_vector(self) -> np.ndarray:
        if not self.signal_state or self.signal_features is None:
            return np.empty(0, dtype=np.float32)
        idx = self.current_step - 1
        return self.signal_features.iloc[idx].to_numpy(copy=True)

    def _get_observation(self) -> np.ndarray:
        returns_flat = self._get_returns_window()
        vol_vector = self._get_vol_vector()
        parts = [returns_flat, vol_vector, self.prev_weights]
        if self.signal_state:
            parts.append(self._get_signal_vector())
        obs = np.concatenate(parts, dtype=np.float32)
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

    def _compute_eta_t(self) -> tuple[float, float | None]:
        mode = self.cfg.eta_mode
        if mode == "legacy":
            if self.cfg.rebalance_eta is None:
                return 1.0, None
            return float(self.cfg.rebalance_eta), None
        if mode == "none":
            return 1.0, None
        if mode == "fixed":
            return float(self.cfg.rebalance_eta), None
        if mode == "rule_vol":
            sigma_vec = self._get_vol_vector()
            sigma_t = float(np.mean(sigma_vec))
            lambda_t = float(self.cfg.rule_vol_a) * sigma_t
            eta_t = 1.0 / (1.0 + 2.0 * lambda_t)
            eta_t = float(np.clip(eta_t, self.cfg.eta_clip_min, self.cfg.eta_clip_max))
            return eta_t, lambda_t
        raise ValueError(f"Unsupported eta_mode: {mode}")

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

        eta_t, lambda_t = self._compute_eta_t()
        w_exec = (1.0 - eta_t) * prev_weights + eta_t * w_target
        w_exec = self._safe_normalize_weights(w_exec)

        turnover_target = turnover_l1(prev_weights, w_target)
        turnover_exec = turnover_l1(prev_weights, w_exec)
        assert turnover_exec >= 0.0
        assert turnover_target >= 0.0
        assert abs(float(np.sum(w_exec)) - 1.0) < 1e-6

        portfolio_return = float(np.dot(w_exec, arithmetic_returns))
        cost_exec = self.cfg.transaction_cost * turnover_exec
        cost_target = self.cfg.transaction_cost * turnover_target
        net_return_lin_exec = portfolio_return - cost_exec
        net_return_lin_target = portfolio_return - cost_target
        tracking_error_l2 = float(np.linalg.norm(w_exec - w_target, ord=2))

        collapse_flag = False
        collapse_reason = None
        if not np.isfinite(portfolio_return):
            collapse_flag = True
            collapse_reason = "portfolio_return_non_finite"

        raw_log_argument = 1.0 + portfolio_return
        if not np.isfinite(raw_log_argument):
            collapse_flag = True
            if collapse_reason is None:
                collapse_reason = "log_argument_non_finite"
            log_argument = float(self.cfg.log_clip)
        else:
            log_argument = max(raw_log_argument, self.cfg.log_clip)
        if not np.isfinite(log_argument) or log_argument <= 0.0:
            collapse_flag = True
            if collapse_reason is None:
                collapse_reason = "log_argument_invalid_after_clip"
            log_argument = float(self.cfg.log_clip)

        log_return_gross = math.log(log_argument)
        log_return_net = log_return_gross - cost_exec
        log_return_net_target = log_return_gross - cost_target

        cost = cost_exec
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
            "rebalance_eta": self.cfg.rebalance_eta,
            "eta_mode": self.cfg.eta_mode,
            "eta_t": eta_t,
            "lambda_t": lambda_t,
            "w_target_l1": float(np.abs(w_target).sum()),
            "w_exec_l1": float(np.abs(w_exec).sum()),
            "date": step_date,
            "cost": cost,
            "cost_exec": cost_exec,
            "cost_target": cost_target,
            "net_return_lin_exec": net_return_lin_exec,
            "net_return_lin_target": net_return_lin_target,
            "tracking_error_l2": tracking_error_l2,
            "log_argument": log_argument,
            "log_return_gross": log_return_gross,
            "log_return_net": log_return_net,
            "log_return_net_target": log_return_net_target,
            "risk_penalty": risk_penalty,
            "risk_lambda": risk_lambda,
            "reward_no_risk": log_return_net,
            "collapse_flag": collapse_flag,
            "collapse_reason": collapse_reason,
        }
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no rendering in v1
        return None
