from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


AVAILABLE_SIGNALS: tuple[str, ...] = (
    "cs_mom_3_12",
    "cs_mom_6_1",
    "vol_scaled_mom",
    "residual_mom_beta_neutral",
    "short_term_reversal",
    "reversal_5d",
)


def parse_signal_list(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return list(AVAILABLE_SIGNALS)
    if isinstance(spec, str):
        raw = spec.strip()
        if not raw or raw.lower() == "all":
            return list(AVAILABLE_SIGNALS)
        names = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in spec if str(part).strip()]
    if not names:
        return list(AVAILABLE_SIGNALS)

    invalid = [name for name in names if name not in AVAILABLE_SIGNALS]
    if invalid:
        raise ValueError(f"Unknown signals requested: {invalid}. Available={list(AVAILABLE_SIGNALS)}")
    return names


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=1, skipna=True), axis=0)
    scale = frame.std(axis=1, ddof=0, skipna=True).replace(0.0, np.nan)
    out = centered.div(scale, axis=0)
    return out.replace([np.inf, -np.inf], np.nan)


def _align_inputs(prices: pd.DataFrame, returns_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = prices.index.intersection(returns_log.index)
    cols = prices.columns.intersection(returns_log.columns)
    if idx.empty or cols.empty:
        raise ValueError("Cannot build signals: prices/returns intersection is empty.")
    return prices.loc[idx, cols], returns_log.loc[idx, cols]


def _cs_mom_3_12_raw(returns_log: pd.DataFrame) -> pd.DataFrame:
    # 12m-1m momentum family using trading-day approximation.
    long_12m = returns_log.rolling(window=252, min_periods=252).sum()
    short_1m = returns_log.rolling(window=21, min_periods=21).sum()
    return long_12m - short_1m


def _vol_scaled_mom_raw(returns_log: pd.DataFrame) -> pd.DataFrame:
    mom_3m = returns_log.rolling(window=63, min_periods=63).sum()
    vol_3m = returns_log.rolling(window=63, min_periods=63).std(ddof=0).replace(0.0, np.nan)
    return mom_3m / vol_3m


def _cs_mom_6_1_raw(returns_log: pd.DataFrame) -> pd.DataFrame:
    # 6m-1m momentum family using trading-day approximation.
    long_6m = returns_log.rolling(window=126, min_periods=126).sum()
    short_1m = returns_log.rolling(window=21, min_periods=21).sum()
    return long_6m - short_1m


def _residual_mom_beta_neutral_raw(returns_log: pd.DataFrame) -> pd.DataFrame:
    arithmetic = np.expm1(returns_log)
    market_proxy = arithmetic.mean(axis=1)

    beta_window = 126
    resid_window = 63
    market_var = market_proxy.rolling(window=beta_window, min_periods=beta_window).var(ddof=0)

    beta = pd.DataFrame(index=arithmetic.index, columns=arithmetic.columns, dtype=np.float64)
    for col in arithmetic.columns:
        cov = arithmetic[col].rolling(window=beta_window, min_periods=beta_window).cov(market_proxy)
        beta[col] = cov / market_var

    residual = arithmetic - beta.mul(market_proxy, axis=0)
    return residual.rolling(window=resid_window, min_periods=resid_window).sum()


def _short_term_reversal_raw(returns_log: pd.DataFrame) -> pd.DataFrame:
    # Negative short-window momentum (10-day) as reversal proxy.
    return -returns_log.rolling(window=10, min_periods=10).sum()


def _reversal_5d_raw(returns_log: pd.DataFrame) -> pd.DataFrame:
    # Fast reversal variant (5-day).
    return -returns_log.rolling(window=5, min_periods=5).sum()


def compute_signal_frames(
    prices: pd.DataFrame,
    returns_log: pd.DataFrame,
    *,
    signals: str | Sequence[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    selected = parse_signal_list(signals)
    _, aligned_returns = _align_inputs(prices, returns_log)

    builders = {
        "cs_mom_3_12": _cs_mom_3_12_raw,
        "cs_mom_6_1": _cs_mom_6_1_raw,
        "vol_scaled_mom": _vol_scaled_mom_raw,
        "residual_mom_beta_neutral": _residual_mom_beta_neutral_raw,
        "short_term_reversal": _short_term_reversal_raw,
        "reversal_5d": _reversal_5d_raw,
    }

    out: Dict[str, pd.DataFrame] = {}
    for name in selected:
        raw = builders[name](aligned_returns)
        out[name] = cross_sectional_zscore(raw)
    return out


def flatten_signal_frames(signal_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for signal_name, frame in signal_frames.items():
        stacked = frame.stack(dropna=False).rename("signal_value").reset_index()
        stacked.columns = ["date", "asset", "signal_value"]
        stacked["signal"] = signal_name
        rows.append(stacked)
    if not rows:
        return pd.DataFrame(columns=["date", "asset", "signal", "signal_value"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", "asset", "signal", "signal_value"]]


def signal_names_from_frames(signal_frames: Dict[str, pd.DataFrame]) -> Iterable[str]:
    return signal_frames.keys()
