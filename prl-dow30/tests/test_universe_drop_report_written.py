import numpy as np
import pandas as pd

from prl.data import load_market_data


def _fake_fetch(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    values = (100.0 + np.arange(len(idx))).astype(float)
    if ticker == "BBB":
        values[1::3] = np.nan
    return pd.DataFrame({ticker: values}, index=idx)


def test_universe_drop_report_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    cfg = {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-03-31"},
        "data": {
            "processed_dir": str(tmp_path / "processed"),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 2,
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
                "max_missing_fraction": 0.10,
            },
        },
    }

    load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )

    drop_report = tmp_path / "outputs" / "reports" / "universe_drop_report.csv"
    assert drop_report.exists()
    report_df = pd.read_csv(drop_report)
    row = report_df.loc[report_df["ticker"] == "BBB", "reason"]
    assert not row.empty
    assert row.iloc[0] != ""
