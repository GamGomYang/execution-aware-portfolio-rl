import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import SAC

from prl.data import MarketData
from prl.train import build_env_for_range, run_training


class _FakeVolFeatures:
    def __init__(self, volatility: pd.DataFrame, stats_path: str):
        self.volatility = volatility
        self.stats_path = stats_path


def _deterministic_market():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
    prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=["AAA", "BBB"])
    return MarketData(prices=prices, returns=returns)


def test_reproducible_training_with_seed(tmp_path, monkeypatch):
    market = _deterministic_market()
    vol = pd.DataFrame(0.02, index=market.returns.index, columns=market.returns.columns)
    stats_path = tmp_path / "vol_stats.npz"
    stats_path.write_bytes(b"stub")

    def _fake_load_market(*args, **kwargs):
        return market

    def _fake_compute_volatility_features(*args, **kwargs):
        return _FakeVolFeatures(volatility=vol, stats_path=stats_path)

    monkeypatch.setattr("prl.train.load_market_data", _fake_load_market)
    monkeypatch.setattr("prl.train.compute_volatility_features", _fake_compute_volatility_features)

    base_cfg = {
        "mode": "smoke",
        "dates": {
            "train_start": str(market.returns.index.min().date()),
            "train_end": str(market.returns.index.max().date()),
            "test_start": str(market.returns.index.min().date()),
            "test_end": str(market.returns.index.max().date()),
        },
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
        "env": {"L": 5, "Lv": 5, "c_tc": 0.0},
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
            "batch_size": 32,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 1000,
            "total_timesteps": 20,
            "ent_coef": 0.2,
        },
    }

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    path1 = run_training(base_cfg, "baseline", seed=123, raw_dir="data/raw", processed_dir="data/processed", output_dir=out1, force_refresh=False)
    path2 = run_training(base_cfg, "baseline", seed=123, raw_dir="data/raw", processed_dir="data/processed", output_dir=out2, force_refresh=False)

    env = build_env_for_range(
        market=market,
        features=_FakeVolFeatures(volatility=vol, stats_path=stats_path),
        start=base_cfg["dates"]["train_start"],
        end=base_cfg["dates"]["train_end"],
        window_size=base_cfg["env"]["L"],
        c_tc=base_cfg["env"]["c_tc"],
        seed=123,
    )

    model1 = SAC.load(path1, env=env)
    model2 = SAC.load(path2, env=env)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    action1, _ = model1.predict(obs, deterministic=True)
    action2, _ = model2.predict(obs, deterministic=True)

    assert np.allclose(action1, action2, atol=1e-6)
