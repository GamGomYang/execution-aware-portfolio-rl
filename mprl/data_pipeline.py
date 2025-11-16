"""Data loading utilities using yfinance for real-market backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import os

import numpy as np
import pandas as pd
import certifi
import yfinance as yf

from .config import FeatureConfig


EXTRA_TICKERS = {
    "vix": "^VIX",
    "move": "^IRX",  # short-rate proxy
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

    def _download(self, tickers: list[str]) -> pd.DataFrame:
        data = yf.download(
            tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        data = data.ffill().bfill()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    def load(self, cache: bool = True) -> MarketDataset:
        cache_path = self.data_dir / f"mprl_{self.start}_{self.end}_{self.interval}.npz"
        if cache and cache_path.exists():
            dataset = self._load_cache_file(cache_path)
            if dataset is not None:
                return dataset

        asset_prices = self._download(self.cfg.dow_symbols)
        sector_prices = self._download(self.cfg.sector_etfs)
        extra_prices = self._download(list(EXTRA_TICKERS.values()))

        asset_returns_df = asset_prices.pct_change().dropna()
        sector_returns_df = sector_prices.pct_change().reindex(asset_returns_df.index).fillna(0.0)
        extra_prices = extra_prices.reindex(asset_returns_df.index).ffill().bfill()

        macro_df = pd.DataFrame(index=asset_returns_df.index)
        macro_df["vix"] = extra_prices[EXTRA_TICKERS["vix"]]
        macro_df["move"] = extra_prices[EXTRA_TICKERS["move"]]
        macro_df["vvix"] = extra_prices.get(EXTRA_TICKERS["vvix"], macro_df["vix"].rolling(5).std())
        macro_df["ted_spread"] = (
            extra_prices[EXTRA_TICKERS["move"]].diff().fillna(0.0) / 100.0
        )
        macro_df["hy_ig_spread"] = (
            (extra_prices[EXTRA_TICKERS["hyg"]].pct_change() - extra_prices[EXTRA_TICKERS["lqd"]].pct_change())
            .rolling(5)
            .mean()
            .fillna(0.0)
        )
        macro_df["spx_return"] = extra_prices[EXTRA_TICKERS["spx"]].pct_change().fillna(0.0)
        macro_df["nasdaq_return"] = extra_prices[EXTRA_TICKERS["nasdaq"]].pct_change().fillna(0.0)
        macro_df["russell_return"] = extra_prices[EXTRA_TICKERS["russell"]].pct_change().fillna(0.0)
        # Placeholders for correlation stats; actual values filled inside env each step.
        macro_df["mean_corr"] = 0.0
        macro_df["max_corr"] = 0.0
        macro_df["eig1"] = 0.0
        macro_df["eig2"] = 0.0
        macro_df = macro_df.ffill().bfill()

        dataset = MarketDataset(
            dates=asset_returns_df.index,
            asset_returns=asset_returns_df.to_numpy(dtype=np.float32),
            sector_returns=sector_returns_df.to_numpy(dtype=np.float32),
            macro_df=macro_df,
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
