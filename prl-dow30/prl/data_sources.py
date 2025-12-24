from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import certifi
import numpy as np
import pandas as pd
import requests
import yfinance as yf

LOGGER = logging.getLogger(__name__)


def _build_session(session_opts: Optional[Dict] = None) -> requests.Session:
    opts = session_opts or {}
    verify_ssl = opts.get("verify_ssl", True)
    ca_bundle_path = opts.get("ca_bundle_path")
    session = requests.Session()
    if ca_bundle_path:
        session.verify = ca_bundle_path
    elif verify_ssl:
        session.verify = certifi.where()
    else:
        session.verify = False
    return session


def _normalize_adj_close(frame: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("No data returned from source.")
    if isinstance(frame.columns, pd.MultiIndex):
        if "Adj Close" not in frame.columns.get_level_values(0):
            raise RuntimeError("Adj Close field missing from source data.")
        adj_close = frame["Adj Close"]
    else:
        if "Adj Close" not in frame:
            raise RuntimeError("Adj Close field missing from source data.")
        adj_close = frame["Adj Close"]
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])
    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close.index.name = "Date"
    adj_close = adj_close.sort_index()
    return adj_close


def fetch_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    session_opts: Optional[Dict] = None,
) -> pd.DataFrame:
    tickers_list = list(tickers)
    session = _build_session(session_opts)
    LOGGER.info("Fetching from yfinance for %s (%s -> %s)", tickers_list, start, end)
    try:
        data = yf.download(
            tickers=tickers_list,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            session=session,
        )
    except TypeError:
        data = yf.download(
            tickers=tickers_list,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    adj_close = _normalize_adj_close(data, tickers_list)
    return adj_close


def _flat_fraction(prices: pd.Series) -> float:
    diffs = prices.diff().fillna(0.0)
    flat_days = (diffs.abs() < 1e-9).sum()
    return float(flat_days) / float(len(prices)) if len(prices) else 1.0


def data_quality_check(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    min_days: int = 500,
    min_vol_std: float = 0.002,
    min_max_abs_return: float = 0.01,
    max_flat_fraction: float = 0.30,
    missing_fraction: Dict[str, float] | None = None,
    max_missing_fraction: float = 0.05,
) -> None:
    if prices_df.empty or returns_df.empty:
        raise RuntimeError("DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: empty data.")
    if len(prices_df) < min_days or len(returns_df) < min_days:
        raise RuntimeError(
            "DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: insufficient history."
        )
    arithmetic_returns = returns_df.apply(np.expm1)
    for ticker in prices_df.columns:
        if ticker not in arithmetic_returns:
            raise RuntimeError(
                "DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: missing returns column."
            )
        r = arithmetic_returns[ticker].dropna()
        p = prices_df[ticker].dropna()
        if len(r) < min_days or len(p) < min_days:
            raise RuntimeError(
                "DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: insufficient data per ticker."
            )
        std = float(r.std())
        if std < min_vol_std:
            raise RuntimeError("DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: low volatility.")
        max_abs = float(r.abs().max())
        if max_abs < min_max_abs_return:
            raise RuntimeError("DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: capped moves.")
        flat_ratio = _flat_fraction(p)
        if flat_ratio > max_flat_fraction:
            raise RuntimeError("DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: flat prices.")
        if missing_fraction is not None:
            miss = missing_fraction.get(ticker, 0.0)
            if miss > max_missing_fraction:
                raise RuntimeError("DATA_QUALITY_FAILED: likely synthetic or corrupted. Reason: missing data.")


def compute_quality_summary(
    prices: pd.DataFrame, returns: pd.DataFrame, missing_fraction: Dict[str, float] | None = None
) -> pd.DataFrame:
    rows = []
    arithmetic_returns = returns.apply(np.expm1)
    missing_fraction = missing_fraction or {}
    for ticker in prices.columns:
        r = arithmetic_returns[ticker].dropna()
        p = prices[ticker].dropna()
        rows.append(
            {
                "ticker": ticker,
                "days": len(p),
                "return_std": float(r.std()),
                "max_abs_return": float(r.abs().max()),
                "flat_fraction": _flat_fraction(p),
                "missing_fraction": missing_fraction.get(ticker, 0.0),
            }
        )
    return pd.DataFrame(rows)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
