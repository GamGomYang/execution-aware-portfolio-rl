import json

import numpy as np
import pandas as pd

from prl.data import load_market_data


def _fake_fetch(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    return pd.DataFrame({ticker: 100.0 + np.arange(len(idx))}, index=idx)


def test_manifest_contains_universe_fields(tmp_path, monkeypatch):
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
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_missing_fraction": 1.0,
                "max_flat_fraction": 1.0,
            },
            "tickers": ["AAA", "BBB"],
        },
        "env": {"L": 2, "Lv": 2, "c_tc": 0.0001},
    }
    load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )
    manifest = json.loads((processed_dir / "data_manifest.json").read_text())
    for key in [
        "asset_list",
        "num_assets",
        "L",
        "Lv",
        "obs_dim_expected",
        "env_signature_hash",
        "data_manifest_hash",
    ]:
        assert key in manifest
    assert manifest["num_assets"] == len(manifest["asset_list"])
    assert manifest["obs_dim_expected"] == len(manifest["asset_list"]) * (manifest["L"] + 2)
    assert manifest["env_signature_hash"]
    assert manifest["data_manifest_hash"]
