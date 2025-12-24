from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class PortfolioMetrics:
    total_reward: float
    avg_reward: float
    cumulative_return: float
    avg_turnover: float
    sharpe: float
    max_drawdown: float
    steps: int

    def to_dict(self):
        return asdict(self)


def compute_metrics(
    rewards: Iterable[float],
    portfolio_returns: Iterable[float],
    turnovers: Iterable[float],
) -> PortfolioMetrics:
    rewards_arr = np.array(list(rewards), dtype=np.float64)
    returns_arr = np.array(list(portfolio_returns), dtype=np.float64)
    turnovers_arr = np.array(list(turnovers), dtype=np.float64)

    total_reward = float(rewards_arr.sum())
    avg_reward = float(rewards_arr.mean()) if rewards_arr.size else 0.0

    cumulative_return = float(np.prod(1.0 + returns_arr) - 1.0)
    avg_turnover = float(turnovers_arr.mean()) if turnovers_arr.size else 0.0

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

    return PortfolioMetrics(
        total_reward=total_reward,
        avg_reward=avg_reward,
        cumulative_return=cumulative_return,
        avg_turnover=avg_turnover,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        steps=int(rewards_arr.size),
    )
