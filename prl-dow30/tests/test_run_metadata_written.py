import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from prl.data import MarketData


class DummyModel:
    def __init__(self):
        self.saved_path = None
        self.logger = SimpleNamespace(name_to_value={"train/actor_loss": 0.0, "train/critic_loss": 0.0, "train/entropy_loss": 0.0})

    def save(self, path):
        self.saved_path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("stub")

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


@pytest.fixture(autouse=True)
def patch_train(monkeypatch):
    # minimal market/env pipeline
    def _fake_prepare_market_and_features(*args, **kwargs):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
        prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=["AAA", "BBB"])
        from prl.features import VolatilityFeatures

        vf = VolatilityFeatures(
            volatility=pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"]),
            portfolio_scalar=pd.Series(0.0, index=dates),
            mean=pd.Series(0.0, index=["AAA", "BBB"]),
            std=pd.Series(1.0, index=["AAA", "BBB"]),
            stats_path=Path("stats.npz"),
        )
        return MarketData(prices=prices, returns=returns), vf

    def _fake_build_env_for_range(*args, **kwargs):
        class DummyEnv:
            def __init__(self):
                self.observation_space = None

            def reset(self, seed=None):
                return np.zeros(1), {}

        return DummyEnv()

    def _fake_train_baseline_model(env, sac_cfg, seed):
        return DummyModel()

    monkeypatch.setattr("prl.train.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("prl.train.build_env_for_range", _fake_build_env_for_range)
    monkeypatch.setattr("prl.train.train_baseline_model", _fake_train_baseline_model)


def test_run_metadata_written(tmp_path, monkeypatch):
    from prl.train import run_training

    cfg = {
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

    run_training(cfg, "baseline", seed=0, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", output_dir=tmp_path / "models", force_refresh=False)

    meta_dir = Path("outputs/reports")
    files = sorted(meta_dir.glob("run_metadata_*.json"))
    assert files, "run_metadata_*.json not created"
    data = json.loads(files[-1].read_text())
    for key in ["run_id", "seed", "mode", "model_type", "config_hash", "python_version", "packages", "created_at", "artifacts"]:
        assert key in data
    assert "model_path" in data["artifacts"]
    assert "train_log_path" in data["artifacts"]
    assert Path(data["artifacts"]["model_path"]).exists()
    assert Path(data["artifacts"]["train_log_path"]).exists()
