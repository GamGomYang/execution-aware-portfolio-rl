from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prl.data import MarketData
from prl.features import VolatilityFeatures
from prl.baselines import BaselineTimeseries
from prl.metrics import PortfolioMetrics


def test_run_all_writes_metrics_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "mode": "smoke",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-10",
            "test_start": "2020-01-01",
            "test_end": "2020-01-10",
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 1,
            "history_tolerance_days": 0,
            "min_history_days": 5,
            "require_cache": True,
            "paper_mode": True,
            "offline": True,
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_missing_fraction": 1.0,
                "max_flat_fraction": 1.0,
            },
            "ticker_substitutions": {},
        },
        "env": {"L": 2, "Lv": 2, "c_tc": 0.0, "logit_scale": 1.0},
        "prl": {
            "alpha0": 0.2,
            "beta": 1.0,
            "lambdav": 2.0,
            "bias": 0.0,
            "alpha_min": 0.01,
            "alpha_max": 1.0,
        },
        "sac": {
            "learning_rate": 0.001,
            "batch_size": 8,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 1000,
            "total_timesteps": 5,
            "ent_coef": 0.2,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    market = MarketData(
        prices=pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"]),
        returns=pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"]),
    )
    features = VolatilityFeatures(
        volatility=pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"]),
        portfolio_scalar=pd.Series(0.0, index=dates),
        mean=0.0,
        std=1.0,
        stats_path=tmp_path / "stats.json",
    )
    metrics = PortfolioMetrics(
        total_reward=0.1,
        avg_reward=0.01,
        cumulative_return=0.02,
        avg_turnover=0.03,
        total_turnover=0.3,
        sharpe=0.5,
        max_drawdown=-0.1,
        steps=10,
    )

    def _fake_prepare_market_and_features(*args, **kwargs):
        return market, features

    def _fake_run_training(*args, **kwargs):
        model_type = kwargs.get("model_type", "baseline")
        seed = kwargs.get("seed", 0)
        model_path = Path("outputs/models") / f"{model_type}_seed{seed}_final.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("stub")
        return model_path

    def _fake_build_env_for_range(*args, **kwargs):
        return "env"

    def _fake_load_model(*args, **kwargs):
        return object()

    def _fake_run_backtest_timeseries(*args, **kwargs):
        dates_ts = pd.date_range("2020-01-01", periods=4, freq="B")
        ts = {
            "dates": dates_ts,
            "portfolio_return": [0.01, 0.0, -0.005, 0.002],
            "turnover": [0.1, 0.2, 0.1, 0.0],
            "vol_portfolio": [0.2, 0.21, 0.22, 0.23],
        }
        return metrics, ts

    def _make_baseline_ts(model_type: str) -> BaselineTimeseries:
        return BaselineTimeseries(
            model_type=model_type,
            dates=pd.date_range("2020-01-01", periods=4, freq="B"),
            portfolio_return=np.array([0.01, 0.01, 0.01, 0.01]),
            turnover=np.array([0.0, 0.0, 0.0, 0.0]),
            vol_portfolio=np.array([0.2, 0.2, 0.2, 0.2]),
            weights_max=np.array([0.5, 0.5, 0.5, 0.5]),
        )

    monkeypatch.setattr("scripts.run_all.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_all.run_training", _fake_run_training)
    monkeypatch.setattr("scripts.run_all.build_env_for_range", _fake_build_env_for_range)
    monkeypatch.setattr("scripts.run_all.load_model", _fake_load_model)
    monkeypatch.setattr("scripts.run_all.run_backtest_timeseries", _fake_run_backtest_timeseries)
    monkeypatch.setattr("scripts.run_all.buy_and_hold_equal_weight", lambda **kwargs: _make_baseline_ts("buy_and_hold_equal_weight"))
    monkeypatch.setattr(
        "scripts.run_all.daily_rebalanced_equal_weight",
        lambda **kwargs: _make_baseline_ts("daily_rebalanced_equal_weight"),
    )
    monkeypatch.setattr(
        "scripts.run_all.inverse_vol_risk_parity",
        lambda **kwargs: _make_baseline_ts("inverse_vol_risk_parity"),
    )
    monkeypatch.setattr(
        "scripts.run_all.compute_regime_metrics",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "model_type": kwargs["model_type"],
                    "regime": "low",
                    "n_obs": 4,
                    "cumulative_return": 0.01,
                    "sharpe": 0.1,
                    "max_drawdown": -0.01,
                    "avg_turnover": 0.1,
                    "total_turnover": 0.2,
                }
            ]
        ),
    )
    monkeypatch.setattr("scripts.run_all.create_scheduler", lambda *args, **kwargs: None)

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["run_all.py", "--config", str(cfg_path), "--seeds", "0", "--model-types", "baseline", "prl", "--offline"],
    )

    from scripts import run_all as run_all_script

    run_all_script.main()

    metrics_path = Path("outputs/reports/metrics.csv")
    assert metrics_path.exists()
    df = pd.read_csv(metrics_path)
    assert set(df["model_type"]) == {
        "baseline",
        "prl",
        "buy_and_hold_equal_weight",
        "daily_rebalanced_equal_weight",
        "inverse_vol_risk_parity",
    }
    regime_path = Path("outputs/reports/regime_metrics.csv")
    assert regime_path.exists()
