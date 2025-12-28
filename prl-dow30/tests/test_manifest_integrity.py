import json

import pandas as pd

from prl.data import load_market_data


def _fake_fetch(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    return pd.DataFrame({ticker: 50.0 + pd.RangeIndex(len(idx))}, index=idx)


def test_manifest_contains_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    processed_dir = tmp_path / "processed"
    cfg = {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-01-10"},
        "data": {
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
            "ticker_substitutions": {},
            "quality_params": {"min_vol_std": 0.0, "min_max_abs_return": 0.0, "max_missing_fraction": 1.0, "max_flat_fraction": 1.0},
            "tickers": ["AAA", "BBB"],
        },
    }
    load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )
    manifest = json.loads((processed_dir / "data_manifest.json").read_text())
    assert manifest["source"] == "yfinance_only"
    assert manifest["price_type"] == "adj_close"
    assert set(manifest["requested_tickers"]) == {"AAA", "BBB"}
    assert set(manifest["kept_tickers"]) == {"AAA", "BBB"}
    hashes = manifest["processed_hashes"]
    assert hashes["prices.parquet"]
    assert hashes["returns.parquet"]
    assert manifest["start"] == "2020-01-01"
    assert manifest["end"] == "2020-01-10"
    assert "quality_params" in manifest
    assert "min_history_days" in manifest
    assert "python_version" in manifest
