import pytest
from pathlib import Path

from prl.train import run_training
from prl.data import MarketData
from prl.envs import EnvConfig, Dow30PortfolioEnv
from prl.eval import load_model, run_backtest_episode
import pandas as pd
import numpy as np


def _tiny_config(tmp_path):
    return {
        "mode": "smoke",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-10",
            "test_start": "2020-01-01",
            "test_end": "2020-01-10",
        },
        "data": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "min_history_days": 5,
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_missing_fraction": 1.0,
                "max_flat_fraction": 1.0,
            },
            "source": "yfinance_only",
            "force_refresh": False,
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


def _fake_market():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
    prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=["AAA", "BBB"])
    return MarketData(prices=prices, returns=returns)


def _build_env(market):
    cfg = EnvConfig(
        returns=market.returns,
        volatility=pd.DataFrame(0.02, index=market.returns.index, columns=market.returns.columns),
        window_size=2,
        transaction_cost=0.0,
        logit_scale=1.0,
    )
    env = Dow30PortfolioEnv(cfg)
    env.reset()
    return env


def test_model_naming_and_run_eval_default(tmp_path, monkeypatch):
    market = _fake_market()

    def _fake_load_market_data(*args, **kwargs):
        return market

    def _fake_compute_vol(*args, **kwargs):
        class FV:
            def __init__(self, vol):
                self.volatility = vol
                self.stats_path = tmp_path / "stats.npz"

        return FV(pd.DataFrame(0.02, index=market.returns.index, columns=market.returns.columns))

    monkeypatch.setattr("prl.train.load_market_data", _fake_load_market_data)
    monkeypatch.setattr("prl.train.compute_volatility_features", _fake_compute_vol)

    cfg = _tiny_config(tmp_path)
    out_dir = tmp_path / "models"
    model_path = run_training(cfg, "baseline", seed=0, raw_dir="data/raw", processed_dir="data/processed", output_dir=out_dir, force_refresh=False)

    assert model_path.name == "baseline_seed0_final.zip"
    assert model_path.exists()

    env = _build_env(market)
    model = load_model(model_path, "baseline", env, scheduler=None)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[0] == market.returns.shape[1]
