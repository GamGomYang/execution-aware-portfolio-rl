from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np


def turnover_l1(prev_weights: np.ndarray, weights: np.ndarray) -> float:
    """Full L1 turnover: sum(|w_t - w_{t-1}|)."""
    return float(np.abs(weights - prev_weights).sum())


def post_return_weights(
    w_prev: np.ndarray, r_arith: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    w_prev = np.asarray(w_prev, dtype=np.float64)
    r_arith = np.asarray(r_arith, dtype=np.float64)
    if w_prev.shape != r_arith.shape:
        raise ValueError("w_prev and r_arith must have matching shapes")

    numer = w_prev * (1.0 + r_arith)
    denom = float(numer.sum())
    if denom <= eps:
        denom = float(np.clip(w_prev.sum(), eps, None))
        return (w_prev / denom).astype(np.float64)
    w_post = numer / denom
    return w_post.astype(np.float64)


def turnover_rebalance_l1(w_target: np.ndarray, w_post: np.ndarray) -> float:
    """Rebalance turnover: sum(|w_target - w_post|)."""
    return float(turnover_l1(w_target, w_post))


@dataclass
class PortfolioMetrics:
    total_reward: float
    avg_reward: float
    cumulative_return: float
    # Backward-compatible alias: exec turnover.
    avg_turnover: float
    # Backward-compatible alias: exec turnover.
    total_turnover: float
    sharpe: float
    max_drawdown: float
    steps: int
    avg_turnover_exec: Optional[float] = None
    total_turnover_exec: Optional[float] = None
    avg_turnover_target: Optional[float] = None
    total_turnover_target: Optional[float] = None
    mean_daily_return_gross: Optional[float] = None
    std_daily_return_gross: Optional[float] = None
    cumulative_return_net_exp: Optional[float] = None
    sharpe_net_exp: Optional[float] = None
    max_drawdown_net_exp: Optional[float] = None
    mean_daily_net_return_exp: Optional[float] = None
    std_daily_net_return_exp: Optional[float] = None
    cumulative_return_net_lin: Optional[float] = None
    sharpe_net_lin: Optional[float] = None
    max_drawdown_net_lin: Optional[float] = None
    mean_daily_net_return_lin: Optional[float] = None
    std_daily_net_return_lin: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def _sanitize_returns(arr: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(arr), dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = arr[~np.isnan(arr)]
    return arr


def _compute_return_stats(returns_arr: np.ndarray) -> tuple[float, float, float, float, float]:
    returns_arr = _sanitize_returns(returns_arr)
    if not returns_arr.size:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean = float(returns_arr.mean())
    std = float(returns_arr.std(ddof=0))
    sharpe = 0.0
    if std > 1e-8:
        sharpe = float((mean / std) * np.sqrt(252))
    else:
        std = 0.0

    equity_curve = np.cumprod(1.0 + returns_arr)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0
    max_drawdown = float(drawdowns.min())

    cumulative_return = float(equity_curve[-1] - 1.0)
    return cumulative_return, sharpe, max_drawdown, mean, std


def compute_metrics(
    rewards: Iterable[float],
    portfolio_returns: Iterable[float],
    turnovers: Iterable[float],
    *,
    turnovers_target: Optional[Iterable[float]] = None,
    net_returns_exp: Optional[Iterable[float]] = None,
    net_returns_lin: Optional[Iterable[float]] = None,
) -> PortfolioMetrics:
    rewards_arr = _sanitize_returns(rewards)
    returns_arr = _sanitize_returns(portfolio_returns)
    turnovers_exec_arr = _sanitize_returns(turnovers)
    turnovers_target_arr = _sanitize_returns(turnovers_target) if turnovers_target is not None else None

    total_reward = float(rewards_arr.sum())
    avg_reward = float(rewards_arr.mean()) if rewards_arr.size else 0.0

    avg_turnover_exec = float(turnovers_exec_arr.mean()) if turnovers_exec_arr.size else 0.0
    total_turnover_exec = float(turnovers_exec_arr.sum())
    avg_turnover_target = None
    total_turnover_target = None
    if turnovers_target_arr is not None:
        avg_turnover_target = float(turnovers_target_arr.mean()) if turnovers_target_arr.size else 0.0
        total_turnover_target = float(turnovers_target_arr.sum())

    cumulative_return, sharpe, max_drawdown, mean_gross, std_gross = _compute_return_stats(returns_arr)

    net_exp_stats = None
    if net_returns_exp is not None:
        net_returns_exp_arr = _sanitize_returns(net_returns_exp)
        net_exp_stats = _compute_return_stats(net_returns_exp_arr)
    else:
        # Rewards already contain cost; exp(reward)-1 is the canonical net return.
        net_returns_exp_arr = np.expm1(rewards_arr)
        net_exp_stats = _compute_return_stats(net_returns_exp_arr)
    net_lin_stats = None
    if net_returns_lin is not None:
        net_returns_lin_arr = _sanitize_returns(net_returns_lin)
        net_lin_stats = _compute_return_stats(net_returns_lin_arr)

    return PortfolioMetrics(
        total_reward=total_reward,
        avg_reward=avg_reward,
        cumulative_return=cumulative_return,
        avg_turnover=avg_turnover_exec,
        total_turnover=total_turnover_exec,
        avg_turnover_exec=avg_turnover_exec,
        total_turnover_exec=total_turnover_exec,
        avg_turnover_target=avg_turnover_target,
        total_turnover_target=total_turnover_target,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        steps=int(rewards_arr.size),
        mean_daily_return_gross=mean_gross,
        std_daily_return_gross=std_gross,
        cumulative_return_net_exp=net_exp_stats[0] if net_exp_stats else None,
        sharpe_net_exp=net_exp_stats[1] if net_exp_stats else None,
        max_drawdown_net_exp=net_exp_stats[2] if net_exp_stats else None,
        mean_daily_net_return_exp=net_exp_stats[3] if net_exp_stats else None,
        std_daily_net_return_exp=net_exp_stats[4] if net_exp_stats else None,
        cumulative_return_net_lin=net_lin_stats[0] if net_lin_stats else None,
        sharpe_net_lin=net_lin_stats[1] if net_lin_stats else None,
        max_drawdown_net_lin=net_lin_stats[2] if net_lin_stats else None,
        mean_daily_net_return_lin=net_lin_stats[3] if net_lin_stats else None,
        std_daily_net_return_lin=net_lin_stats[4] if net_lin_stats else None,
    )
