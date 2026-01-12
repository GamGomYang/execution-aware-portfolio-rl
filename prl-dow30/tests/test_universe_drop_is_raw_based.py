import pandas as pd
import numpy as np

from prl.data import load_market_data


def _fake_fetch(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, end=end, freq="B")
    values = (100.0 + pd.RangeIndex(len(idx))).to_numpy(dtype=float)
    if ticker == "BBB":
        values = values.astype(float)
        values[1::3] = np.nan  # 약 33% 결측 주입 (첫 번째 값은 유지)
    data = pd.DataFrame({ticker: values}, index=idx)
    return data


def test_drop_decisions_use_raw_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    cfg = {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-06-30"},
        "data": {
            "processed_dir": str(tmp_path / "processed"),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 2,
            "force_refresh": True,
            "offline": False,
            "require_cache": False,
            "paper_mode": False,
            "min_history_days": 50,
            "history_tolerance_days": 0,
            "ticker_substitutions": {},
            "tickers": ["AAA", "BBB", "CCC"],
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_flat_fraction": 1.0,
                "max_missing_fraction": 0.20,
            },
        },
    }

    (
        prices,
        returns,
        manifest,
        _quality_summary,
        _raw_prices_clean,
        filled_prices,
        raw_missing_fraction,
        drop_decisions,
        _market_closed_info,
        debug_info,
    ) = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
        debug_return_intermediates=True,
    )

    assert raw_missing_fraction["BBB"] > 0.2
    assert drop_decisions["BBB"] == "RAW_MISSING_FRACTION_EXCEEDED"
    assert "BBB" not in manifest["kept_tickers"]
    assert "BBB" not in prices.columns
    assert "BBB" not in returns.columns
    assert "BBB" not in filled_prices.columns  # drop 결정 후 유지되지 않음

    # 채워진 값으로 결측이 사라져도 drop 결정은 바뀌지 않음을 보장
    raw_aligned = debug_info["raw_aligned"]
    assert raw_aligned["BBB"].isna().mean() > 0.2
