"""Backtesting utilities for evaluating trained MPRL agents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data_pipeline import MarketDataLoader
from .decision_io import load_decision_records

TRADING_DAYS = 252
REBAL_THRESHOLD = 1e-3


def _load_decisions(path: Path) -> pd.DataFrame:
    records, _ = load_decision_records(path)
    for record in records:
        record["date"] = pd.Timestamp(record["date"])
    if not records:
        raise ValueError(f"No decision records found in {path}")
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _calc_drawdown(equity: np.ndarray) -> Tuple[float, np.ndarray]:
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_max, 1e-12) - 1.0
    max_dd = float(drawdowns.min())
    return max_dd, drawdowns


def _calc_monthly_downside(portfolio: pd.Series) -> float:
    if portfolio.empty:
        return 0.0
    cumulative = (1.0 + portfolio).cumprod()
    monthly = cumulative.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return 0.0
    downside = np.minimum(monthly.values, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def _calc_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    if len(returns) == 0:
        return 0.0
    cutoff = int(np.ceil((1 - alpha) * len(returns)))
    cutoff = max(cutoff, 1)
    tail = np.sort(returns)[:cutoff]
    return float(tail.mean())


def _aggregate_metrics(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
    turnover: np.ndarray,
) -> Dict[str, float]:
    if len(returns) == 0:
        raise ValueError("Cannot compute metrics with zero aligned returns.")
    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(len(returns) / TRADING_DAYS, 1e-6)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    vol = float(np.std(returns) * np.sqrt(TRADING_DAYS))
    equity = np.cumprod(1.0 + returns)
    max_dd, drawdowns = _calc_drawdown(equity)
    cvar = _calc_cvar(returns, alpha=0.95)
    monthly_dd = _calc_monthly_downside(pd.Series(returns, index=dates))
    mean_return = float(np.mean(returns))
    sharpe = float(mean_return / (np.std(returns) + 1e-12) * np.sqrt(TRADING_DAYS))
    downside = np.minimum(returns, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(
        mean_return / (downside_std + 1e-12) * np.sqrt(TRADING_DAYS)
    )
    calmar = float(cagr / (abs(max_dd) + 1e-12))
    turnover_mean = float(np.mean(turnover)) if len(turnover) else 0.0
    rebalance_freq = float(np.mean(turnover > REBAL_THRESHOLD)) if len(turnover) else 0.0
    win_rate = float(np.mean(returns > 0.0))
    stability = float(np.var(weights, axis=0).mean()) if len(weights) else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": max_dd,
        "cvar_95": cvar,
        "monthly_downside_deviation": monthly_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "turnover": turnover_mean,
        "win_rate": win_rate,
        "rebalance_frequency": rebalance_freq,
        "stability": stability,
    }


def run_backtest(
    *,
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    data_dir: str | Path = "data",
    log_path: str | Path = "logs/decisions.jsonl",
    cache: bool = True,
) -> Dict[str, float]:
    cfg = FeatureConfig()
    loader = MarketDataLoader(cfg, start=start, end=end, data_dir=data_dir)
    dataset = loader.load(cache=cache)
    decisions_df = _load_decisions(Path(log_path))
    asset_returns_df = pd.DataFrame(
        dataset.asset_returns, index=dataset.dates, columns=cfg.dow_symbols
    )
    aligned_returns: List[float] = []
    aligned_dates: List[pd.Timestamp] = []
    weights: List[np.ndarray] = []
    turnover: List[float] = []
    prev_weights: np.ndarray | None = None

    used_logged_returns = False
    for _, row in decisions_df.iterrows():
        date = row["date"]
        if date not in asset_returns_df.index:
            continue
        action = np.asarray(row["action"], dtype=np.float64)
        if action.shape[0] != len(cfg.dow_symbols):
            continue
        logged_ret = row.get("portfolio_return")
        if logged_ret is not None:
            port_ret = float(logged_ret)
            used_logged_returns = True
        else:
            asset_ret = asset_returns_df.loc[date].to_numpy(dtype=np.float64)
            port_ret = float(np.dot(action, asset_ret))
        aligned_returns.append(port_ret)
        aligned_dates.append(date)
        weights.append(action)
        if prev_weights is None:
            turnover.append(0.0)
        else:
            turnover.append(float(np.abs(action - prev_weights).sum()))
        prev_weights = action

    if not aligned_returns:
        raise ValueError(
            "No overlapping dates between decision log and dataset. "
            "Ensure start/end dates match the training run."
        )

    returns_arr = np.asarray(aligned_returns, dtype=np.float64)
    dates_idx = pd.DatetimeIndex(aligned_dates)
    metrics = _aggregate_metrics(
        returns_arr,
        dates_idx,
        np.asarray(weights, dtype=np.float64),
        np.asarray(turnover, dtype=np.float64),
    )
    direct_total = float(np.prod(1.0 + returns_arr) - 1.0)
    years = max(len(returns_arr) / TRADING_DAYS, 1e-6)
    direct_cagr = float((1.0 + direct_total) ** (1.0 / years) - 1.0)
    metrics["total_return_np_prod"] = direct_total
    metrics["cagr_np_prod"] = direct_cagr
    metrics["used_logged_returns"] = used_logged_returns
    return metrics


def _format_metrics(metrics: Dict[str, float]) -> str:
    lines = ["Backtest metrics:"]
    for key, value in metrics.items():
        if "ratio" in key or "frequency" in key or key == "win_rate":
            lines.append(f"- {key}: {value:.4f}")
        elif "vol" in key or "deviation" in key or "turnover" in key or key == "stability":
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value:.4%}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest decisions.jsonl outputs.")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2023-12-31", help="Backtest end date")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with cached data")
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/decisions.jsonl",
        help="Path to decisions.jsonl produced during training",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cached dataset reload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_backtest(
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
        log_path=args.log_path,
        cache=not args.no_cache,
    )
    print(_format_metrics(metrics))


if __name__ == "__main__":
    main()
