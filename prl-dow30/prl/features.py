from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class VolatilityFeatures:
    volatility: pd.DataFrame
    portfolio_scalar: pd.Series
    stats_path: Path
    mean: float
    std: float


def _safe_std(series: pd.Series) -> float:
    std = float(series.std(ddof=0))
    return std if std > 1e-8 else 1e-8


def save_vol_stats(path: Path, mean: float, std: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": mean, "std": std}
    path.write_text(json.dumps(payload, indent=2))


def load_vol_stats(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    return float(data["mean"]), float(data["std"])


def compute_volatility_features(
    returns: pd.DataFrame,
    lv: int,
    train_start: str,
    train_end: str,
    processed_dir: str | Path = "data/processed",
    stats_filename: str = "vol_stats.json",
) -> VolatilityFeatures:
    """Compute rolling volatility and scalar portfolio volatility."""

    if lv <= 1:
        raise ValueError("Lv must be greater than 1")

    vol = returns.rolling(window=lv, min_periods=lv).std(ddof=0)
    vol = vol.dropna(how="any")
    portfolio_scalar = vol.mean(axis=1)

    train_slice = portfolio_scalar.loc[train_start:train_end]
    mean = float(train_slice.mean())
    std = _safe_std(train_slice)

    stats_path = Path(processed_dir) / stats_filename
    save_vol_stats(stats_path, mean, std)

    return VolatilityFeatures(
        volatility=vol,
        portfolio_scalar=portfolio_scalar,
        stats_path=stats_path,
        mean=mean,
        std=std,
    )


def normalize_portfolio_vol(portfolio_scalar: pd.Series, mean: float, std: float) -> pd.Series:
    z = (portfolio_scalar - mean) / std
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
