import pandas as pd
import pytest

from prl.data import load_market_data


def _paper_cfg(processed_dir, quality_params=None):
    data_cfg = {
        "processed_dir": str(processed_dir),
        "source": "yfinance_only",
        "force_refresh": False,
        "offline": True,
        "require_cache": True,
        "paper_mode": True,
        "min_history_days": 5,
        "history_tolerance_days": 0,
        "min_assets": 1,
        "universe_policy": "availability_filtered",
        "ticker_substitutions": {},
        "quality_params": quality_params or {"min_vol_std": 0.0, "min_max_abs_return": 0.0, "max_missing_fraction": 1.0, "max_flat_fraction": 1.0},
    }
    return {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-01-10"},
        "data": data_cfg,
    }


def test_paper_mode_requires_cache_missing(tmp_path):
    processed_dir = tmp_path / "processed"
    cfg = _paper_cfg(processed_dir)
    with pytest.raises(RuntimeError, match="CACHE_MISSING"):
        load_market_data(
            cfg,
            offline=True,
            require_cache=True,
            cache_only=True,
            force_refresh=False,
        )


def test_paper_mode_uses_existing_cache(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    prices = pd.DataFrame({"AAA": 100.0 + pd.RangeIndex(len(dates))}, index=dates)
    returns = prices.pct_change().dropna()
    prices.to_parquet(processed_dir / "prices.parquet")
    returns.to_parquet(processed_dir / "returns.parquet")

    cfg = _paper_cfg(processed_dir)
    loaded_prices, loaded_returns, _, _ = load_market_data(
        cfg,
        offline=True,
        require_cache=True,
        cache_only=True,
        force_refresh=False,
    )
    assert loaded_prices.equals(prices)
    assert loaded_returns.equals(returns)
