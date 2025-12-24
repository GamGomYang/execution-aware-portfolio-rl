from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from .data_sources import compute_quality_summary, data_quality_check, fetch_yfinance, hash_file

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
    prices: pd.DataFrame
    returns: pd.DataFrame


def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ratio = prices / prices.shift(1)
    returns = pd.DataFrame(np.log(ratio), index=prices.index, columns=prices.columns)
    returns = returns.replace([np.inf, -np.inf], pd.NA)
    returns = returns.dropna(how="any")
    return returns


def _align_prices(series_list: List[pd.Series]) -> tuple[pd.DataFrame, dict]:
    raw = pd.concat(series_list, axis=1, join="outer")
    raw.columns = [s.name for s in series_list]
    missing_fraction = raw.isna().mean().to_dict()
    prices = raw.sort_index().ffill().bfill().dropna(how="any")
    return prices, missing_fraction


def _write_manifest(
    processed_dir: Path,
    tickers: list[str],
    start: str,
    end: str,
    processed_hashes: Dict[str, str],
    quality_params: Dict,
    min_history_days: int,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start,
        "end_date": end,
        "tickers": tickers,
        "source": "yfinance_only",
        "price_type": "adj_close",
        "processed_hashes": processed_hashes,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "quality_params": quality_params,
        "min_history_days": min_history_days,
        "python_version": sys.version,
    }
    manifest_path = processed_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _write_quality_summary(prices: pd.DataFrame, returns: pd.DataFrame, missing_fraction: Dict[str, float]) -> None:
    summary = compute_quality_summary(prices, returns, missing_fraction)
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "data_quality_summary.csv", index=False)


def _load_processed_cache(processed_dir: Path) -> MarketData:
    prices_path = processed_dir / "prices.parquet"
    returns_path = processed_dir / "returns.parquet"
    if not prices_path.exists() or not returns_path.exists():
        raise RuntimeError("CACHE_MISSING: processed parquet files not found; run build_cache.py")
    LOGGER.info("Loading processed cache from %s", processed_dir)
    prices = pd.read_parquet(prices_path)
    returns = pd.read_parquet(returns_path)
    return MarketData(prices=prices, returns=returns)


def _download_yfinance_all(
    tickers: list[str],
    start: str,
    end: str,
    session_opts: Dict | None,
) -> tuple[List[pd.Series], Dict[str, str]]:
    series: List[pd.Series] = []
    errors: Dict[str, str] = {}
    for ticker in tickers:
        try:
            df = fetch_yfinance([ticker], start, end, session_opts=session_opts)
            try:
                ser = df[ticker]
            except KeyError:
                ser = df.iloc[:, 0]
            ser = ser.dropna()
            ser.name = ticker
            if ser.empty:
                raise RuntimeError("empty series")
            series.append(ser)
        except Exception as exc:
            errors[ticker] = str(exc)
    return series, errors


def _raise_download_failure(errors: Dict[str, str], total: int) -> None:
    failed = list(errors.keys())
    success_count = total - len(failed)
    msg = (
        f"YFINANCE_DOWNLOAD_FAILED: failed_tickers={failed}; "
        f"success_count={success_count}/{total}; causes={errors}"
    )
    raise RuntimeError(msg)


def load_market_data(
    start_date: str,
    end_date: str,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    tickers: Iterable[str] | None = None,
    force_refresh: bool = True,
    session_opts: Dict | None = None,
    min_history_days: int = 500,
    quality_params: Dict | None = None,
    source: str = "yfinance_only",
    offline: bool = False,
    require_cache: bool = False,
    paper_mode: bool = False,
    cache_only: bool = False,
) -> MarketData:
    if source != "yfinance_only":
        raise ValueError("Unsupported data source; only yfinance_only is permitted.")

    tickers = list(tickers or DOW30_TICKERS)
    if len(tickers) != 30:
        LOGGER.warning("Ticker universe size is %d (expected 30)", len(tickers))

    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    must_use_cache = offline or require_cache or cache_only
    if must_use_cache:
        return _load_processed_cache(proc_path)

    series, errors = _download_yfinance_all(tickers, start_date, end_date, session_opts)
    if errors:
        _raise_download_failure(errors, total=len(tickers))

    prices, missing_fraction = _align_prices(series)
    prices = prices.loc[start_date:end_date]
    returns = _compute_log_returns(prices)
    qp = quality_params or {}
    data_quality_check(
        prices,
        returns,
        min_days=min_history_days,
        min_vol_std=qp.get("min_vol_std", 0.002),
        min_max_abs_return=qp.get("min_max_abs_return", 0.01),
        max_flat_fraction=qp.get("max_flat_fraction", 0.30),
        missing_fraction=missing_fraction,
        max_missing_fraction=qp.get("max_missing_fraction", 0.05),
    )

    prices_path = proc_path / "prices.parquet"
    returns_path = proc_path / "returns.parquet"
    prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)
    processed_hashes = {
        "prices": hash_file(prices_path),
        "returns": hash_file(returns_path),
    }
    _write_manifest(
        proc_path,
        tickers,
        start_date,
        end_date,
        processed_hashes,
        quality_params or {},
        min_history_days,
    )
    _write_quality_summary(prices, returns, missing_fraction)

    return MarketData(prices=prices, returns=returns)


def slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[start:end]


def split_returns(
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = slice_frame(returns, train_start, train_end)
    test = slice_frame(returns, test_start, test_end)
    return train, test
