import json

import numpy as np
import pandas as pd
import pytest

from prl.data import load_market_data


def _fake_fetch(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    values = 100.0 + np.arange(len(idx), dtype=float)
    return pd.DataFrame({ticker: values}, index=idx)


def test_manifest_atomic_write_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    sentinel = {"sentinel": True}
    manifest_path = processed_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(sentinel))

    cfg = {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-01-20"},
        "data": {
            "processed_dir": str(processed_dir),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 3,
            "force_refresh": True,
            "offline": False,
            "require_cache": False,
            "paper_mode": False,
            "min_history_days": 5,
            "history_tolerance_days": 0,
            "ticker_substitutions": {},
            "tickers": ["AAA", "BBB"],
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_flat_fraction": 1.0,
                "max_missing_fraction": 1.0,
            },
        },
    }

    with pytest.raises(RuntimeError, match="DATA_UNIVERSE_TOO_SMALL"):
        load_market_data(
            cfg,
            offline=False,
            require_cache=False,
            cache_only=False,
            force_refresh=True,
        )

    assert json.loads(manifest_path.read_text()) == sentinel
    failed_manifest_path = processed_dir / "data_manifest_failed.json"
    assert failed_manifest_path.exists()
    failed_payload = json.loads(failed_manifest_path.read_text())
    assert failed_payload["error_code"] == "DATA_UNIVERSE_TOO_SMALL"
    assert failed_payload["min_assets"] == 3
    assert failed_payload["N_assets_final"] == 2
    assert (processed_dir / "data_quality_failed.csv").exists()
