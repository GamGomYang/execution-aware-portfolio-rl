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
    "minimum_variance",
    "mean_variance_long_only",
)

DEFAULT_LOOKBACK = 252
DEFAULT_HISTORY_MIN = 30
DEFAULT_MEAN_VARIANCE_RISK_AVERSION = 10.0


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


def _regularize_covariance(cov: np.ndarray, ridge_scale: float = 1e-6) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if cov.size == 0:
        return cov
    trace = float(np.trace(cov))
    dim = cov.shape[0]
    ridge = ridge_scale * (trace / dim if np.isfinite(trace) and trace > 0.0 else 1.0)
    return cov + ridge * np.eye(dim, dtype=np.float64)


def minimum_variance_weights(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov_reg = _regularize_covariance(cov)
    ones = np.ones(cov_reg.shape[0], dtype=np.float64)
    try:
        raw = np.linalg.solve(cov_reg, ones)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov_reg) @ ones
    raw = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
    if not np.isfinite(raw).all() or float(raw.sum()) <= eps:
        inv_diag = 1.0 / np.clip(np.diag(cov_reg), 1e-8, None)
        raw = np.clip(inv_diag, 0.0, None)
    return normalize_weights(raw, eps=eps)


def mean_variance_weights(
    mean_return: np.ndarray,
    cov: np.ndarray,
    *,
    risk_aversion: float = DEFAULT_MEAN_VARIANCE_RISK_AVERSION,
    eps: float = 1e-12,
) -> np.ndarray:
    cov_reg = _regularize_covariance(cov)
    mu = np.asarray(mean_return, dtype=np.float64)
    ones = np.ones(cov_reg.shape[0], dtype=np.float64)
    gamma = float(risk_aversion)
    gamma = gamma if np.isfinite(gamma) and gamma > 0.0 else DEFAULT_MEAN_VARIANCE_RISK_AVERSION
    try:
        solve_mu = np.linalg.solve(cov_reg, mu)
        solve_one = np.linalg.solve(cov_reg, ones)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(cov_reg)
        solve_mu = pinv @ mu
        solve_one = pinv @ ones
    denom = float(ones @ solve_one)
    if not np.isfinite(denom) or abs(denom) <= eps:
        return minimum_variance_weights(cov_reg, eps=eps)
    nu = float((ones @ solve_mu - gamma) / denom)
    raw = (solve_mu - nu * solve_one) / gamma
    raw = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
    if not np.isfinite(raw).all() or float(raw.sum()) <= eps:
        inv_diag = 1.0 / np.clip(np.diag(cov_reg), 1e-8, None)
        raw = np.clip(np.maximum(mu, 0.0) * inv_diag, 0.0, None)
    if not np.isfinite(raw).all() or float(raw.sum()) <= eps:
        return minimum_variance_weights(cov_reg, eps=eps)
    return normalize_weights(raw, eps=eps)


def run_baseline_strategy(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    strategy: str,
    *,
    transaction_cost: float = 0.0,
    lookback: int = DEFAULT_LOOKBACK,
    history_min: int = DEFAULT_HISTORY_MIN,
    mean_variance_risk_aversion: float = DEFAULT_MEAN_VARIANCE_RISK_AVERSION,
    log_clip: float = 1e-8,
    eps: float = 1e-12,
) -> PortfolioMetrics:
    metrics, _ = run_baseline_strategy_detailed(
        returns,
        volatility,
        strategy,
        transaction_cost=transaction_cost,
        lookback=lookback,
        history_min=history_min,
        mean_variance_risk_aversion=mean_variance_risk_aversion,
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
    lookback: int = DEFAULT_LOOKBACK,
    history_min: int = DEFAULT_HISTORY_MIN,
    mean_variance_risk_aversion: float = DEFAULT_MEAN_VARIANCE_RISK_AVERSION,
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
    arithmetic_returns_arr = np.expm1(returns_arr)

    rewards = []
    portfolio_returns = []
    turnovers = []
    dates = []
    costs = []
    net_returns_exp = []
    net_returns_lin = []
    log_returns_gross = []
    for i in range(returns_arr.shape[0]):
        r_arith = arithmetic_returns_arr[i]
        port_ret = float(np.dot(w_prev, r_arith))
        w_post = post_return_weights(w_prev, r_arith, eps=eps)

        if strategy == "buy_and_hold_equal_weight":
            w_target = w_post
        elif strategy == "daily_rebalanced_equal_weight":
            w_target = equal_weight
        elif strategy == "inverse_vol_risk_parity":
            vol_idx = i - 1 if i > 0 else 0
            w_target = inverse_vol_weights(vol_arr[vol_idx], eps=eps)
        elif strategy in {"minimum_variance", "mean_variance_long_only"}:
            start = max(0, i - int(lookback))
            history = arithmetic_returns_arr[start:i]
            if int(history.shape[0]) < int(history_min):
                w_target = equal_weight
            else:
                mean_ret = np.nanmean(history, axis=0)
                cov = np.cov(history, rowvar=False)
                if strategy == "minimum_variance":
                    w_target = minimum_variance_weights(cov, eps=eps)
                else:
                    w_target = mean_variance_weights(
                        mean_ret,
                        cov,
                        risk_aversion=float(mean_variance_risk_aversion),
                        eps=eps,
                    )
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        turnover = turnover_rebalance_l1(w_target, w_post)
        log_argument = max(1.0 + port_ret, log_clip)
        cost = transaction_cost * turnover
        reward = math.log(log_argument) - cost

        rewards.append(reward)
        portfolio_returns.append(port_ret)
        turnovers.append(turnover)
        costs.append(cost)
        net_returns_exp.append(math.exp(reward) - 1.0)
        net_returns_lin.append(port_ret - cost)
        log_returns_gross.append(math.log(log_argument))
        dates.append(returns.index[i])
        w_prev = w_target

    metrics = compute_metrics(
        rewards,
        portfolio_returns,
        turnovers,
        net_returns_exp=net_returns_exp,
        net_returns_lin=net_returns_lin,
    )
    trace = {
        "dates": dates,
        "rewards": rewards,
        "portfolio_returns": portfolio_returns,
        "turnovers": turnovers,
        "costs": costs,
        "net_returns_exp": net_returns_exp,
        "net_returns_lin": net_returns_lin,
        "log_returns_gross": log_returns_gross,
    }
    return metrics, trace


def run_all_baselines(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float = 0.0,
    lookback: int = DEFAULT_LOOKBACK,
    history_min: int = DEFAULT_HISTORY_MIN,
    mean_variance_risk_aversion: float = DEFAULT_MEAN_VARIANCE_RISK_AVERSION,
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
            lookback=lookback,
            history_min=history_min,
            mean_variance_risk_aversion=mean_variance_risk_aversion,
            log_clip=log_clip,
            eps=eps,
        )
    return results


def run_all_baselines_detailed(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float = 0.0,
    lookback: int = DEFAULT_LOOKBACK,
    history_min: int = DEFAULT_HISTORY_MIN,
    mean_variance_risk_aversion: float = DEFAULT_MEAN_VARIANCE_RISK_AVERSION,
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
            lookback=lookback,
            history_min=history_min,
            mean_variance_risk_aversion=mean_variance_risk_aversion,
            log_clip=log_clip,
            eps=eps,
        )
    return results
