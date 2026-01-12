from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_portfolio_metrics


def assign_vol_regimes(vol_portfolio: pd.Series, *, n_bins: int = 3) -> pd.Series:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    if n_bins == 3:
        labels = ["low", "mid", "high"]
    else:
        labels = [f"q{i}" for i in range(1, n_bins + 1)]
    vol_rank = vol_portfolio.rank(method="first")
    try:
        return pd.qcut(vol_rank, q=n_bins, labels=labels)
    except ValueError:
        quantiles = [i / n_bins for i in range(1, n_bins)]
        thresholds = vol_portfolio.quantile(quantiles).to_list()
        bins = [-np.inf, *thresholds, np.inf]
        return pd.cut(vol_portfolio, bins=bins, labels=labels, include_lowest=True)


def compute_regime_metrics(
    *,
    model_type: str,
    dates: pd.DatetimeIndex,
    portfolio_return: np.ndarray,
    turnover: np.ndarray,
    vol_portfolio: np.ndarray,
    n_bins: int = 3,
) -> pd.DataFrame:
    # MDD here is conditional on the regime-subset sequence (non-contiguous).
    vol_series = pd.Series(vol_portfolio, index=dates)
    regimes = assign_vol_regimes(vol_series, n_bins=n_bins)
    rows = []
    for regime in regimes.dropna().unique():
        mask = regimes == regime
        if not mask.any():
            continue
        sub_returns = np.asarray(portfolio_return)[mask.values]
        sub_turnover = np.asarray(turnover)[mask.values]
        metrics = compute_portfolio_metrics(sub_returns, sub_turnover)
        rows.append(
            {
                "model_type": model_type,
                "regime": str(regime),
                "n_obs": int(mask.sum()),
                "cumulative_return": metrics.cumulative_return,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "avg_turnover": metrics.avg_turnover,
                "total_turnover": metrics.total_turnover,
            }
        )
    return pd.DataFrame(rows)
