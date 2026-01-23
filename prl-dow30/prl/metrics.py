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
    avg_turnover: float
    total_turnover: float
    sharpe: float
    max_drawdown: float
    steps: int
    cumulative_return_net_exp: Optional[float] = None
    sharpe_net_exp: Optional[float] = None
    max_drawdown_net_exp: Optional[float] = None
    cumulative_return_net_lin: Optional[float] = None
    sharpe_net_lin: Optional[float] = None
    max_drawdown_net_lin: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def _compute_return_stats(returns_arr: np.ndarray) -> tuple[float, float, float]:
    cumulative_return = float(np.prod(1.0 + returns_arr) - 1.0) if returns_arr.size else 0.0

    return_std = returns_arr.std(ddof=0)
    sharpe = 0.0
    if return_std > 1e-8:
        sharpe = float((returns_arr.mean() / return_std) * np.sqrt(252))

    if returns_arr.size:
        equity_curve = np.cumprod(1.0 + returns_arr)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / running_max - 1.0
        max_drawdown = float(drawdowns.min())
    else:
        max_drawdown = 0.0

    return cumulative_return, sharpe, max_drawdown


def compute_metrics(
    rewards: Iterable[float],
    portfolio_returns: Iterable[float],
    turnovers: Iterable[float],
    *,
    net_returns_exp: Optional[Iterable[float]] = None,
    net_returns_lin: Optional[Iterable[float]] = None,
) -> PortfolioMetrics:
    rewards_arr = np.array(list(rewards), dtype=np.float64)
    returns_arr = np.array(list(portfolio_returns), dtype=np.float64)
    turnovers_arr = np.array(list(turnovers), dtype=np.float64)

    total_reward = float(rewards_arr.sum())
    avg_reward = float(rewards_arr.mean()) if rewards_arr.size else 0.0

    avg_turnover = float(turnovers_arr.mean()) if turnovers_arr.size else 0.0
    total_turnover = float(turnovers_arr.sum())

    cumulative_return, sharpe, max_drawdown = _compute_return_stats(returns_arr)

    net_exp_stats = None
    if net_returns_exp is not None:
        net_returns_exp_arr = np.array(list(net_returns_exp), dtype=np.float64)
        net_exp_stats = _compute_return_stats(net_returns_exp_arr)
    else:
        # Rewards already contain cost; exp(reward)-1 is the canonical net return.
        net_returns_exp_arr = np.expm1(rewards_arr)
        net_exp_stats = _compute_return_stats(net_returns_exp_arr)
    net_lin_stats = None
    if net_returns_lin is not None:
        net_returns_lin_arr = np.array(list(net_returns_lin), dtype=np.float64)
        net_lin_stats = _compute_return_stats(net_returns_lin_arr)

    return PortfolioMetrics(
        total_reward=total_reward,
        avg_reward=avg_reward,
        cumulative_return=cumulative_return,
        avg_turnover=avg_turnover,
        total_turnover=total_turnover,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        steps=int(rewards_arr.size),
        cumulative_return_net_exp=net_exp_stats[0] if net_exp_stats else None,
        sharpe_net_exp=net_exp_stats[1] if net_exp_stats else None,
        max_drawdown_net_exp=net_exp_stats[2] if net_exp_stats else None,
        cumulative_return_net_lin=net_lin_stats[0] if net_lin_stats else None,
        sharpe_net_lin=net_lin_stats[1] if net_lin_stats else None,
        max_drawdown_net_lin=net_lin_stats[2] if net_lin_stats else None,
    )
