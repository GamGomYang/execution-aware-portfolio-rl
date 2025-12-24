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
    load_market_data(
        start_date="2020-01-01",
        end_date="2020-01-10",
        processed_dir=processed_dir,
        tickers=["AAA", "BBB"],
        force_refresh=True,
        source="yfinance_only",
        min_history_days=5,
        quality_params={"min_vol_std": 0.0, "min_max_abs_return": 0.0, "max_missing_fraction": 1.0},
    )
    manifest = json.loads((processed_dir / "data_manifest.json").read_text())
    assert manifest["source"] == "yfinance"
    assert manifest["price_type"] == "adj_close"
    assert set(manifest["tickers"]) == {"AAA", "BBB"}
    hashes = manifest["processed_hashes"]
    assert hashes["prices"]
    assert hashes["returns"]
    assert manifest["start_date"] == "2020-01-01"
    assert manifest["end_date"] == "2020-01-10"
    assert "quality_params" in manifest
    assert "min_history_days" in manifest
    assert "python_version" in manifest
