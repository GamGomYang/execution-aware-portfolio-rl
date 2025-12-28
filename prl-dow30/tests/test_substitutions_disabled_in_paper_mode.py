import pytest

from prl.data import load_market_data


def test_substitutions_blocked_in_paper_mode(tmp_path):
    cfg = {
        "dates": {"train_start": "2020-01-01", "test_end": "2020-01-10"},
        "data": {
            "processed_dir": str(tmp_path / "processed"),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 1,
            "force_refresh": False,
            "offline": True,
            "require_cache": True,
            "paper_mode": True,
            "min_history_days": 1,
            "history_tolerance_days": 0,
            "ticker_substitutions": {"WBA": "CVS"},
            "quality_params": {"min_vol_std": 0.0, "min_max_abs_return": 0.0, "max_flat_fraction": 1.0, "max_missing_fraction": 1.0},
        },
    }

    with pytest.raises(ValueError, match="SUBSTITUTIONS_DISABLED_IN_PAPER_MODE"):
        load_market_data(
            cfg,
            offline=True,
            require_cache=True,
            cache_only=True,
            force_refresh=False,
        )
