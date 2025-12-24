import json

import pandas as pd
import pytest

from prl.data import load_market_data


def _fake_fetch_success(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    return pd.DataFrame({ticker: 100.0 + pd.RangeIndex(len(idx))}, index=idx)


def test_build_cache_yfinance_success(tmp_path, monkeypatch):
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch_success)
    processed_dir = tmp_path / "processed"
    market = load_market_data(
        start_date="2020-01-01",
        end_date="2020-01-10",
        processed_dir=processed_dir,
        tickers=["AAA"],
        force_refresh=True,
        source="yfinance_only",
        min_history_days=5,
        quality_params={"min_vol_std": 0.0, "min_max_abs_return": 0.0, "max_missing_fraction": 1.0},
    )
    assert processed_dir.joinpath("prices.parquet").exists()
    assert processed_dir.joinpath("returns.parquet").exists()
    manifest = json.loads((processed_dir / "data_manifest.json").read_text())
    assert manifest["source"] == "yfinance_only"
    assert manifest["price_type"] == "adj_close"
    assert market.prices.shape[1] == 1


def test_yfinance_partial_failure_raises(tmp_path, monkeypatch):
    def _fake_fetch(tickers, start, end, session_opts=None):
        ticker = list(tickers)[0]
        if ticker == "BBB":
            raise RuntimeError("boom")
        idx = pd.date_range(start=start, end=end, freq="B")
        return pd.DataFrame({ticker: 100.0 + pd.RangeIndex(len(idx))}, index=idx)

    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    processed_dir = tmp_path / "processed"
    with pytest.raises(RuntimeError, match="YFINANCE_DOWNLOAD_FAILED"):
        load_market_data(
            start_date="2020-01-01",
            end_date="2020-01-10",
            processed_dir=processed_dir,
            tickers=["AAA", "BBB"],
            force_refresh=True,
            source="yfinance_only",
        )
