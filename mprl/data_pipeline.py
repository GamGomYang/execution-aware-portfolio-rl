"""Data loading utilities using yfinance for real-market backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import os
import logging

import numpy as np
import pandas as pd
import certifi
import yfinance as yf

from .config import FeatureConfig


LOGGER = logging.getLogger(__name__)


EXTRA_TICKERS = {
    "vix": "^VIX",
    "move": "^IRX",  
    "vvix": "^VVIX",
    "spx": "^GSPC",
    "nasdaq": "^IXIC",
    "russell": "^RUT",
    "hyg": "HYG",
    "lqd": "LQD",
}

CERT_PATH = Path(certifi.where())
LOCAL_CERT_PATH = Path(__file__).resolve().parent / "cacert.pem"
try:
    data = CERT_PATH.read_bytes()
    if not LOCAL_CERT_PATH.exists() or LOCAL_CERT_PATH.read_bytes() != data:
        LOCAL_CERT_PATH.write_bytes(data)
    cert_target = LOCAL_CERT_PATH
except OSError:
    cert_target = CERT_PATH

os.environ.setdefault("SSL_CERT_FILE", str(cert_target))
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cert_target))


@dataclass
class MarketDataset:
    dates: pd.DatetimeIndex
    asset_returns: np.ndarray
    sector_returns: np.ndarray
    macro_df: pd.DataFrame

    def macro_state(self, idx: int) -> Dict[str, float]:
        row = self.macro_df.iloc[idx]
        return row.to_dict()

    @property
    def length(self) -> int:
        return len(self.dates)

    def slice(self, start: int, end: int) -> "MarketDataset":
        sl = slice(start, end)
        return MarketDataset(
            dates=self.dates[sl],
            asset_returns=self.asset_returns[sl],
            sector_returns=self.sector_returns[sl],
            macro_df=self.macro_df.iloc[sl],
        )


class MarketDataLoader:
    def __init__(
        self,
        cfg: FeatureConfig,
        start: str,
        end: str,
        interval: str = "1d",
        data_dir: str | Path = "data",
    ) -> None:
        self.cfg = cfg
        self.start = start
        self.end = end
        self.interval = interval
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_price_frame(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            name = data.name or "price"
            data = data.to_frame(name=name)

        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            else:
                data.columns = data.columns.get_level_values(-1)
        elif "Adj Close" in data.columns:
            data = data["Adj Close"]

        if isinstance(data, pd.Series):
            data = data.to_frame()

        return data.sort_index()

    def _download_single_ticker(self, ticker: str) -> pd.Series | None:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=True,
            )
        except Exception as exc:
            LOGGER.warning("Failed to download %s via history API: %s", ticker, exc)
            return None

        if hist.empty:
            LOGGER.warning("No data received for ticker %s via history API.", ticker)
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(-1)
        price_col = "Close"
        if "Adj Close" in hist.columns:
            price_col = "Adj Close"
        elif "Close" not in hist.columns:
            price_col = hist.columns[0]

        series = hist[price_col].rename(ticker).sort_index()
        return series

    def _download(self, tickers: list[str]) -> pd.DataFrame:
        data = yf.download(
            tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )
        data = self._normalize_price_frame(data)
        initial_rows = len(data)

        if initial_rows < self.cfg.lookback + 1:
            LOGGER.warning(
                "Bulk download returned only %d rows for %d tickers; retrying sequential downloads.",
                initial_rows,
                len(tickers),
            )
            series_list = []
            for ticker in tickers:
                series = self._download_single_ticker(ticker)
                if series is not None:
                    series_list.append(series)
            if series_list:
                data = pd.concat(series_list, axis=1).sort_index()
            else:
                data = pd.DataFrame()

        if data.empty:
            msg = f"No price data retrieved for tickers: {', '.join(tickers)}"
            raise RuntimeError(msg)

        if data.columns.has_duplicates:
            LOGGER.warning(
                "Duplicate columns detected for tickers %s; keeping the first occurrence.",
                [col for col, dup in zip(data.columns, data.columns.duplicated()) if dup],
            )
            data = data.loc[:, ~data.columns.duplicated()]

        missing_cols = [ticker for ticker in tickers if ticker not in data.columns]
        if missing_cols:
            LOGGER.warning(
                "Missing columns for %d tickers %s; creating flat placeholder series.",
                len(missing_cols),
                missing_cols,
            )
            placeholder = pd.DataFrame(1.0, index=data.index, columns=missing_cols)
            data = pd.concat([data, placeholder], axis=1)

        data = data.reindex(columns=tickers)
        data = data.ffill().bfill()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    def _validate_dataset(self, dataset: MarketDataset) -> bool:
        min_length = self.cfg.lookback + 1
        if dataset.length < min_length:
            LOGGER.error(
                "Dataset length %d is smaller than required minimum %d (lookback=%d).",
                dataset.length,
                min_length,
                self.cfg.lookback,
            )
            return False
        return True

    def load(self, cache: bool = True) -> MarketDataset:
        cache_path = self.data_dir / f"mprl_{self.start}_{self.end}_{self.interval}.npz"
        if cache and cache_path.exists():
            dataset = self._load_cache_file(cache_path)
            if dataset is not None and self._validate_dataset(dataset):
                return dataset
            LOGGER.warning("Cached dataset at %s is invalid; rebuilding.", cache_path)
            try:
                cache_path.unlink()
            except OSError:
                LOGGER.warning("Failed to remove cache file %s", cache_path)

        asset_prices = self._download(self.cfg.dow_symbols)
        sector_prices = self._download(self.cfg.sector_etfs)
        extra_prices = self._download(list(EXTRA_TICKERS.values()))

        asset_returns_df = asset_prices.pct_change(fill_method=None)
        if len(asset_returns_df) > 0:
            asset_returns_df = asset_returns_df.iloc[1:]
        asset_returns_df = asset_returns_df.fillna(0.0)

        sector_returns_df = sector_prices.pct_change(fill_method=None)
        if len(sector_returns_df) > 0:
            sector_returns_df = sector_returns_df.iloc[1:]
        sector_returns_df = sector_returns_df.reindex(asset_returns_df.index).fillna(0.0)
        extra_prices = extra_prices.reindex(asset_returns_df.index).ffill().bfill()

        macro_df = pd.DataFrame(index=asset_returns_df.index)

        def _safe_series(name: str, fallback: pd.Series | float = 0.0) -> pd.Series:
            ticker = EXTRA_TICKERS[name]
            series = extra_prices.get(ticker)
            if series is not None:
                return series
            LOGGER.warning("Missing data for %s (%s); using fallback.", name, ticker)
            if isinstance(fallback, pd.Series):
                return fallback
            return pd.Series(fallback, index=macro_df.index, dtype=np.float32)

        vix_series = _safe_series("vix")
        macro_df["vix"] = vix_series
        move_series = _safe_series("move")
        macro_df["move"] = move_series
        vvix_fallback = macro_df["vix"].rolling(5).std().fillna(0.0)
        macro_df["vvix"] = _safe_series("vvix", fallback=vvix_fallback)
        macro_df["ted_spread"] = move_series.diff().fillna(0.0) / 100.0
        hyg_series = _safe_series("hyg")
        lqd_series = _safe_series("lqd")
        macro_df["hy_ig_spread"] = (
            (hyg_series.pct_change(fill_method=None) - lqd_series.pct_change(fill_method=None))
            .rolling(5)
            .mean()
            .fillna(0.0)
        )
        macro_df["spx_return"] = _safe_series("spx").pct_change(fill_method=None).fillna(0.0)
        macro_df["nasdaq_return"] = _safe_series("nasdaq").pct_change(fill_method=None).fillna(0.0)
        macro_df["russell_return"] = _safe_series("russell").pct_change(fill_method=None).fillna(0.0)
        # Placeholders for correlation stats; actual values filled inside env each step.
        macro_df["mean_corr"] = 0.0
        macro_df["max_corr"] = 0.0
        macro_df["eig1"] = 0.0
        macro_df["eig2"] = 0.0
        macro_df = macro_df.ffill().bfill()

        LOGGER.info(
            "Constructed market dataset with %d samples (assets=%s sectors=%s)",
            len(asset_returns_df),
            asset_prices.shape,
            sector_prices.shape,
        )

        dataset = MarketDataset(
            dates=asset_returns_df.index,
            asset_returns=asset_returns_df.to_numpy(dtype=np.float32),
            sector_returns=sector_returns_df.to_numpy(dtype=np.float32),
            macro_df=macro_df,
        )

        if not self._validate_dataset(dataset):
            raise ValueError(
                "Market dataset is too short; try expanding the date range or reducing lookback."
            )

        if cache:
            self._write_cache(cache_path, dataset)

        return dataset

    def _load_cache_file(self, cache_path: Path) -> MarketDataset | None:
        def build_dataset(npz_obj: np.lib.npyio.NpzFile) -> MarketDataset:
            dates_raw = np.asarray(npz_obj["dates"])
            dates = pd.to_datetime(dates_raw.astype(str))
            asset_returns = npz_obj["asset_returns"]
            sector_returns = npz_obj["sector_returns"]
            macro_cols = np.asarray(npz_obj["macro_columns"]).astype(str)
            macro_matrix = npz_obj["macro_matrix"]
            macro_df = pd.DataFrame(macro_matrix, index=dates, columns=macro_cols)
            return MarketDataset(dates, asset_returns, sector_returns, macro_df)

        def load_with_allow_pickle(flag: bool) -> MarketDataset:
            npz_obj = np.load(cache_path, allow_pickle=flag)
            try:
                return build_dataset(npz_obj)
            finally:
                npz_obj.close()

        try:
            return load_with_allow_pickle(False)
        except ValueError as exc:
            if "Object arrays cannot be loaded" not in str(exc):
                raise
            dataset = load_with_allow_pickle(True)
            self._write_cache(cache_path, dataset)
            return dataset

    def _write_cache(self, cache_path: Path, dataset: MarketDataset) -> None:
        dates_array = np.asarray(dataset.dates.astype(str), dtype="U32")
        macro_columns = np.asarray(dataset.macro_df.columns, dtype="U32")
        np.savez_compressed(
            cache_path,
            dates=dates_array,
            asset_returns=dataset.asset_returns,
            sector_returns=dataset.sector_returns,
            macro_matrix=dataset.macro_df.to_numpy(dtype=np.float32),
            macro_columns=macro_columns,
        )
