from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)

DOW30_TICKERS: Sequence[str] = (
    "AAPL",
    "AMGN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WBA",
    "WMT",
    "DOW",
)


@dataclass
class MarketData:
    """Container for aligned price and return frames."""

    prices: pd.DataFrame
    returns: pd.DataFrame


def _ticker_path(raw_dir: Path, ticker: str) -> Path:
    return raw_dir / f"{ticker}.csv"


def _download_ticker(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series:
    LOGGER.info("Downloading %s from yfinance (%s -> %s)", ticker, start, end)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if "Adj Close" not in data:
        raise ValueError(f"Adj Close field missing for {ticker}")
    series = data["Adj Close"].copy()
    series.index.name = "Date"
    return series


def _load_cached_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if len(df.columns) == 1:
        return df.iloc[:, 0]
    raise ValueError(f"Unexpected columns in {path}: {df.columns}")


def _store_series(path: Path, series: pd.Series) -> None:
    df = series.to_frame(name="Adj Close")
    df.to_csv(path)


def _load_ticker_series(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
    force_refresh: bool,
) -> pd.Series:
    cache_path = _ticker_path(raw_dir, ticker)
    if cache_path.exists() and not force_refresh:
        try:
            return _load_cached_series(cache_path)
        except Exception as exc:  # pragma: no cover - log path
            LOGGER.warning("Failed to load cache for %s (%s). Redownloading.", ticker, exc)
    try:
        series = _download_ticker(ticker, start, end)
        _store_series(cache_path, series)
        return series
    except Exception as download_exc:
        LOGGER.warning("Download failed for %s (%s)", ticker, download_exc)
        if cache_path.exists():
            LOGGER.info("Falling back to cached CSV for %s", ticker)
            return _load_cached_series(cache_path)
        raise


def _align_prices(series_list: List[pd.Series]) -> pd.DataFrame:
    prices = pd.concat(series_list, axis=1, join="outer")
    prices.columns = [s.name for s in series_list]
    prices = prices.sort_index()
    prices = prices.ffill().bfill()
    prices = prices.dropna(how="any")
    return prices


def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ratio = prices / prices.shift(1)
    returns = pd.DataFrame(np.log(ratio), index=prices.index, columns=prices.columns)
    returns = returns.replace([np.inf, -np.inf], pd.NA)
    returns = returns.dropna(how="any")
    return returns


def load_market_data(
    start_date: str,
    end_date: str,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    tickers: Iterable[str] | None = None,
    force_refresh: bool = False,
) -> MarketData:
    """Download (or load cached) Adj Close data and compute log returns."""

    tickers = list(tickers or DOW30_TICKERS)
    if len(tickers) != 30:
        LOGGER.warning("Ticker universe size is %d (expected 30)", len(tickers))

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    proc_path.mkdir(parents=True, exist_ok=True)

    series: List[pd.Series] = []
    for ticker in tickers:
        ticker_series = _load_ticker_series(ticker, start_date, end_date, raw_path, force_refresh)
        ticker_series.name = ticker
        series.append(ticker_series)

    prices = _align_prices(series)
    prices = prices.loc[start_date:end_date]
    returns = _compute_log_returns(prices)

    prices.to_parquet(proc_path / "prices.parquet")
    returns.to_parquet(proc_path / "returns.parquet")

    return MarketData(prices=prices, returns=returns)


def slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Utility to slice a frame by inclusive date strings."""

    return frame.loc[start:end]


def split_returns(
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test slices for downstream environment construction."""

    train = slice_frame(returns, train_start, train_end)
    test = slice_frame(returns, test_start, test_end)
    return train, test
