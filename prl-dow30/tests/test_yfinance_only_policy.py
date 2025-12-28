import json

import pandas as pd

from prl.data import load_market_data


def _fake_fetch_success(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    return pd.DataFrame({ticker: 100.0 + pd.RangeIndex(len(idx))}, index=idx)


def _make_cfg(processed_dir, data_overrides=None):
    data_cfg = {
        "processed_dir": str(processed_dir),
        "source": "yfinance_only",
        "force_refresh": True,
        "offline": False,
        "require_cache": False,
        "paper_mode": False,
        "min_history_days": 5,
        "history_tolerance_days": 0,
        "min_assets": 1,
        "universe_policy": "availability_filtered",
        "quality_params": {"min_vol_std": 0.0, "min_max_abs_return": 0.0, "max_missing_fraction": 1.0, "max_flat_fraction": 1.0},
        "ticker_substitutions": {},
    }
    if data_overrides:
        data_cfg.update(data_overrides)
    cfg = {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-01-10"},
        "data": data_cfg,
    }
    return cfg


def test_build_cache_yfinance_success(tmp_path, monkeypatch):
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch_success)
    processed_dir = tmp_path / "processed"
    cfg = _make_cfg(processed_dir)
    prices, returns, manifest, quality_summary = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )
    assert processed_dir.joinpath("prices.parquet").exists()
    assert processed_dir.joinpath("returns.parquet").exists()
    assert processed_dir.joinpath("quality_summary.csv").exists()
    assert manifest["source"] == "yfinance_only"
    assert manifest["price_type"] == "adj_close"
    assert manifest["start"] == "2020-01-01"
    assert manifest["end"] == "2020-01-10"
    assert manifest["kept_tickers"] == ["AAA"]
    assert manifest["dropped_tickers"] == []
    assert prices.shape[1] == 1
    assert returns.shape[1] == 1
    assert not quality_summary.empty


def test_yfinance_partial_failure_records_drop(tmp_path, monkeypatch):
    def _fake_fetch(tickers, start, end, session_opts=None):
        ticker = list(tickers)[0]
        if ticker == "BBB":
            raise RuntimeError("boom")
        idx = pd.date_range(start=start, end=end, freq="B")
        return pd.DataFrame({ticker: 100.0 + pd.RangeIndex(len(idx))}, index=idx)

    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    processed_dir = tmp_path / "processed"
    cfg = _make_cfg(processed_dir)
    prices, returns, manifest, _ = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )
    assert manifest["kept_tickers"] == ["AAA"]
    assert manifest["dropped_tickers"] == ["BBB"]
    assert manifest["dropped_reasons"]["BBB"] == "YFINANCE_EMPTY"
    assert "BBB" not in prices.columns
    assert "BBB" not in returns.columns
