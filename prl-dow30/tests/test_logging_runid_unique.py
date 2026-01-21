from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from prl.data import MarketData
from prl.features import VolatilityFeatures
from prl.train import run_training


class DummyModel:
    def __init__(self):
        self.saved_path = None
        self.logger = SimpleNamespace(
            name_to_value={
                "train/actor_loss": 0.0,
                "train/critic_loss": 0.0,
                "train/entropy_loss": 0.0,
                "train/ent_coef": 0.2,
            }
        )

    def save(self, path):
        self.saved_path = path

    def learn(self, total_timesteps, callback=None):
        callbacks = callback if isinstance(callback, list) else ([callback] if callback else [])
        for cb in callbacks:
            cb.model = self
        if callbacks:
            callbacks[0].num_timesteps = 1
            callbacks[0]._on_step()
            callbacks[0].num_timesteps = 2
            callbacks[0]._on_step()
            callbacks[0]._on_training_end()


def _minimal_config(tmp_path):
    return {
        "mode": "smoke",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-05",
            "test_start": "2020-01-01",
            "test_end": "2020-01-05",
        },
        "data": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "processed")},
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
            "log_interval_steps": 1,
        },
    }


def test_logging_runid_unique(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _fake_prepare_market_and_features(*args, **kwargs):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
        prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=["AAA", "BBB"])
        market = MarketData(prices=prices, returns=returns)
        features = VolatilityFeatures(
            volatility=pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"]),
            portfolio_scalar=pd.Series(0.0, index=dates),
            mean=0.0,
            std=1.0,
            stats_path=Path("stats.json"),
        )
        return market, features

    monkeypatch.setattr("prl.train.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("prl.train.build_env_for_range", lambda *args, **kwargs: object())
    monkeypatch.setattr("prl.train.train_baseline_model", lambda *args, **kwargs: DummyModel())

    cfg = _minimal_config(tmp_path)
    run_training(cfg, "baseline", seed=0, force_refresh=False)
    run_training(cfg, "baseline", seed=0, force_refresh=False)

    logs = sorted((tmp_path / "outputs" / "logs").glob("train_*.csv"))
    assert len(logs) == 2
    assert logs[0].name != logs[1].name
