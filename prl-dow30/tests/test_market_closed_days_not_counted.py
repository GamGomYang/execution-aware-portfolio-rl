import numpy as np
import pandas as pd

from prl.data import load_market_data


def test_market_closed_days_not_counted(tmp_path, monkeypatch):
    idx_full = pd.bdate_range(start="2020-01-01", periods=100)
    closed_days = [idx_full[10], idx_full[20]]
    idx_partial = idx_full.difference(closed_days)

    def _fake_fetch(tickers, start, end, session_opts=None):
        ticker = list(tickers)[0]
        values = 100.0 + np.arange(len(idx_partial), dtype=float)
        return pd.DataFrame({ticker: values}, index=idx_partial)

    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)

    cfg = {
        "dates": {"train_start": str(idx_full[0].date()), "test_end": str(idx_full[-1].date())},
        "data": {
            "processed_dir": str(tmp_path / "processed"),
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
            "tickers": ["AAA", "BBB", "CCC"],
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_flat_fraction": 1.0,
                "max_missing_fraction": 0.01,
            },
        },
    }

    (
        prices,
        _returns,
        manifest,
        quality_summary,
        raw_prices_clean,
        _filled_prices,
        raw_missing_fraction,
        _drop_decisions,
        market_closed_days_removed,
        market_closed_fraction_removed,
    ) = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
        debug_return_intermediates=True,
    )

    assert market_closed_days_removed == 2
    assert market_closed_fraction_removed == 2 / len(idx_full)
    assert raw_missing_fraction["AAA"] == 0.0
    assert raw_missing_fraction["BBB"] == 0.0
    assert raw_missing_fraction["CCC"] == 0.0
    assert set(manifest["kept_tickers"]) == {"AAA", "BBB", "CCC"}
    assert quality_summary.loc[quality_summary["ticker"] == "AAA", "missing_fraction_raw"].iloc[0] == 0.0
    assert len(raw_prices_clean) == len(idx_full) - 2
    assert len(prices) == len(idx_full) - 2
