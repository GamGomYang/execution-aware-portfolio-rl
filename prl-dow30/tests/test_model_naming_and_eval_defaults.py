import pytest
from pathlib import Path

from prl.train import run_training
from prl.data import MarketData
from prl.envs import EnvConfig, Dow30PortfolioEnv
from prl.eval import load_model, run_backtest_episode
from prl.features import VolatilityFeatures
from prl.metrics import PortfolioMetrics
import pandas as pd
import numpy as np
import yaml


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
            "universe_policy": "availability_filtered",
            "min_assets": 1,
            "history_tolerance_days": 0,
            "min_history_days": 5,
            "require_cache": False,
            "offline": False,
            "paper_mode": False,
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_missing_fraction": 1.0,
                "max_flat_fraction": 1.0,
            },
            "source": "yfinance_only",
            "force_refresh": False,
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
        return market.prices, market.returns, {"kept_tickers": list(market.returns.columns)}, pd.DataFrame()

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


def test_run_eval_default_path_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outputs_models = tmp_path / "outputs" / "models"
    outputs_models.mkdir(parents=True, exist_ok=True)
    default_model = outputs_models / "baseline_seed0_final.zip"
    default_model.write_text("stub")

    # Build minimal config file
    cfg = {
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
            "require_cache": False,
            "paper_mode": False,
            "force_refresh": False,
            "min_history_days": 5,
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
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    market = MarketData(
        prices=pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"]),
        returns=pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"]),
    )
    features = VolatilityFeatures(
        volatility=pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"]),
        portfolio_scalar=pd.Series(0.0, index=dates),
        mean=pd.Series(0.0, index=["AAA", "BBB"]),
        std=pd.Series(1.0, index=["AAA", "BBB"]),
        stats_path=tmp_path / "stats.npz",
    )

    calls = {}

    def _fake_prepare_market_and_features(*args, **kwargs):
        calls["prepare_called"] = True
        return market, features

    def _fake_load_model(model_path, model_type, env, scheduler=None):
        calls["model_path"] = model_path
        calls["model_type"] = model_type

        class DummyModel:
            def predict(self, obs, deterministic=True):
                return np.zeros_like(obs[0]), None

        return DummyModel()

    def _fake_run_backtest_episode(model, env):
        calls["backtest_called"] = True
        return PortfolioMetrics(
            total_reward=0.0,
            avg_reward=0.0,
            cumulative_return=0.0,
            avg_turnover=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            steps=0,
        )

    monkeypatch.setattr("scripts.run_eval.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_eval.build_env_for_range", lambda *args, **kwargs: "env")
    monkeypatch.setattr("scripts.run_eval.create_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_eval.load_model", _fake_load_model)
    monkeypatch.setattr("scripts.run_eval.run_backtest_episode", _fake_run_backtest_episode)

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--config", str(cfg_path), "--model-type", "baseline", "--seed", "0"])

    from scripts import run_eval as run_eval_script

    run_eval_script.main()

    assert calls.get("prepare_called", False)
    assert calls["model_path"].name == "baseline_seed0_final.zip"
    assert calls.get("backtest_called", False)
