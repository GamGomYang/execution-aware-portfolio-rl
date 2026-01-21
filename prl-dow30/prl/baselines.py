from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from .metrics import PortfolioMetrics, compute_metrics, post_return_weights, turnover_rebalance_l1

BASELINE_NAMES = (
    "buy_and_hold_equal_weight",
    "daily_rebalanced_equal_weight",
    "inverse_vol_risk_parity",
)


def normalize_weights(raw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    weights = np.asarray(raw, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= eps:
        return np.full_like(weights, 1.0 / weights.size, dtype=np.float64)
    return (weights / total).astype(np.float64)


def inverse_vol_weights(vol_decision: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vol_decision = np.asarray(vol_decision, dtype=np.float64)
    raw = 1.0 / (vol_decision + eps)
    return normalize_weights(raw, eps=eps)


def run_baseline_strategy(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    strategy: str,
    *,
    transaction_cost: float = 0.0,
    log_clip: float = 1e-8,
    eps: float = 1e-12,
) -> PortfolioMetrics:
    metrics, _ = run_baseline_strategy_detailed(
        returns,
        volatility,
        strategy,
        transaction_cost=transaction_cost,
        log_clip=log_clip,
        eps=eps,
    )
    return metrics


def run_baseline_strategy_detailed(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    strategy: str,
    *,
    transaction_cost: float = 0.0,
    log_clip: float = 1e-8,
    eps: float = 1e-12,
) -> tuple[PortfolioMetrics, dict]:
    if returns.shape != volatility.shape:
        raise ValueError("returns and volatility must have matching shapes")
    if not returns.index.equals(volatility.index):
        raise ValueError("returns and volatility indices must match")
    if strategy not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    num_assets = returns.shape[1]
    equal_weight = np.full(num_assets, 1.0 / num_assets, dtype=np.float64)
    w_prev = equal_weight.copy()

    returns_arr = returns.to_numpy(dtype=np.float64, copy=False)
    vol_arr = volatility.to_numpy(dtype=np.float64, copy=False)

    rewards = []
    portfolio_returns = []
    turnovers = []
    dates = []
    for i in range(returns_arr.shape[0]):
        r_arith = np.expm1(returns_arr[i])
        port_ret = float(np.dot(w_prev, r_arith))
        w_post = post_return_weights(w_prev, r_arith, eps=eps)

        if strategy == "buy_and_hold_equal_weight":
            w_target = w_post
        elif strategy == "daily_rebalanced_equal_weight":
            w_target = equal_weight
        else:  # inverse_vol_risk_parity
            vol_idx = i - 1 if i > 0 else 0
            w_target = inverse_vol_weights(vol_arr[vol_idx], eps=eps)

        turnover = turnover_rebalance_l1(w_target, w_post)
        log_argument = max(1.0 + port_ret, log_clip)
        reward = math.log(log_argument) - transaction_cost * turnover

        rewards.append(reward)
        portfolio_returns.append(port_ret)
        turnovers.append(turnover)
        dates.append(returns.index[i])
        w_prev = w_target

    metrics = compute_metrics(rewards, portfolio_returns, turnovers)
    trace = {
        "dates": dates,
        "rewards": rewards,
        "portfolio_returns": portfolio_returns,
        "turnovers": turnovers,
    }
    return metrics, trace


def run_all_baselines(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float = 0.0,
    log_clip: float = 1e-8,
    eps: float = 1e-12,
) -> Dict[str, PortfolioMetrics]:
    results: Dict[str, PortfolioMetrics] = {}
    for name in BASELINE_NAMES:
        results[name] = run_baseline_strategy(
            returns,
            volatility,
            name,
            transaction_cost=transaction_cost,
            log_clip=log_clip,
            eps=eps,
        )
    return results


def run_all_baselines_detailed(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float = 0.0,
    log_clip: float = 1e-8,
    eps: float = 1e-12,
) -> Dict[str, tuple[PortfolioMetrics, dict]]:
    results: Dict[str, tuple[PortfolioMetrics, dict]] = {}
    for name in BASELINE_NAMES:
        results[name] = run_baseline_strategy_detailed(
            returns,
            volatility,
            name,
            transaction_cost=transaction_cost,
            log_clip=log_clip,
            eps=eps,
        )
    return results
