from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import turnover_l1


@dataclass
class BaselineTimeseries:
    model_type: str
    dates: pd.DatetimeIndex
    portfolio_return: np.ndarray
    turnover: np.ndarray
    vol_portfolio: np.ndarray | None
    weights_max: np.ndarray | None = None


def _equal_weight(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Number of assets must be positive.")
    return np.ones(n, dtype=np.float64) / n


def _inverse_vol_weights(vol_row: np.ndarray, eps: float) -> np.ndarray | None:
    if not np.all(np.isfinite(vol_row)) or np.any(vol_row <= 0.0):
        return None
    inv = 1.0 / (vol_row + eps)
    if not np.all(np.isfinite(inv)):
        return None
    total = float(inv.sum())
    if total <= 0.0:
        return None
    return inv / total


def run_baseline(
    *,
    model_type: str,
    returns: pd.DataFrame,
    volatility: pd.DataFrame | None,
    lookback: int,
    strategy: str,
    eps: float = 1e-8,
) -> BaselineTimeseries:
    if len(returns) <= lookback:
        raise ValueError("Not enough data for baseline evaluation.")
    if strategy == "invvol_rp" and volatility is None:
        raise ValueError("inverse_vol_risk_parity requires volatility input.")
    if volatility is not None:
        idx = returns.index.intersection(volatility.index)
        returns = returns.loc[idx]
        volatility = volatility.loc[idx]
        if not returns.columns.equals(volatility.columns):
            missing = [c for c in returns.columns if c not in volatility.columns]
            if missing:
                raise ValueError(f"Volatility missing columns: {missing}")
            volatility = volatility[returns.columns]
    if len(returns) <= lookback:
        raise ValueError("Not enough aligned data for baseline evaluation.")

    n_assets = returns.shape[1]
    w_prev = _equal_weight(n_assets)
    start_i = lookback
    dates = returns.index[start_i:]
    portfolio_returns: list[float] = []
    turnovers: list[float] = []
    vol_portfolio: list[float] = []
    weights_max: list[float] = []

    for i in range(start_i, len(returns)):
        returns_t = returns.iloc[i].to_numpy(dtype=float)
        returns_t = np.nan_to_num(returns_t, nan=0.0)
        port_ret = float(np.dot(w_prev, np.expm1(returns_t)))

        if strategy == "buy_and_hold":
            w_next = w_prev
        elif strategy == "daily_equal":
            w_next = _equal_weight(n_assets)
        elif strategy == "invvol_rp":
            vol_row = volatility.iloc[i].to_numpy(dtype=float)
            w_next = _inverse_vol_weights(vol_row, eps=eps)
            if w_next is None:
                w_next = _equal_weight(n_assets)
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        turnover = turnover_l1(w_prev, w_next)
        portfolio_returns.append(port_ret)
        turnovers.append(turnover)
        if volatility is not None:
            vol_row = volatility.iloc[i].to_numpy(dtype=float)
            vol_portfolio.append(float(np.nanmean(vol_row)))
        weights_max.append(float(np.max(w_next)))
        w_prev = w_next

    vol_arr = np.array(vol_portfolio, dtype=np.float64) if volatility is not None else None
    weights_max_arr = np.array(weights_max, dtype=np.float64) if weights_max else None
    return BaselineTimeseries(
        model_type=model_type,
        dates=dates,
        portfolio_return=np.array(portfolio_returns, dtype=np.float64),
        turnover=np.array(turnovers, dtype=np.float64),
        vol_portfolio=vol_arr,
        weights_max=weights_max_arr,
    )


def buy_and_hold_equal_weight(
    *,
    returns: pd.DataFrame,
    volatility: pd.DataFrame | None,
    lookback: int,
) -> BaselineTimeseries:
    return run_baseline(
        model_type="buy_and_hold_equal_weight",
        returns=returns,
        volatility=volatility,
        lookback=lookback,
        strategy="buy_and_hold",
    )


def daily_rebalanced_equal_weight(
    *,
    returns: pd.DataFrame,
    volatility: pd.DataFrame | None,
    lookback: int,
) -> BaselineTimeseries:
    return run_baseline(
        model_type="daily_rebalanced_equal_weight",
        returns=returns,
        volatility=volatility,
        lookback=lookback,
        strategy="daily_equal",
    )


def inverse_vol_risk_parity(
    *,
    returns: pd.DataFrame,
    volatility: pd.DataFrame | None,
    lookback: int,
    eps: float = 1e-8,
) -> BaselineTimeseries:
    return run_baseline(
        model_type="inverse_vol_risk_parity",
        returns=returns,
        volatility=volatility,
        lookback=lookback,
        strategy="invvol_rp",
        eps=eps,
    )
