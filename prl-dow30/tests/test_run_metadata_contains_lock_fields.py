import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from prl.data import MarketData
from prl.utils.signature import canonical_json, compute_env_signature, sha256_bytes


class DummyModel:
    def __init__(self):
        self.logger = SimpleNamespace(name_to_value={"train/actor_loss": 0.0, "train/critic_loss": 0.0, "train/entropy_loss": 0.0})

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("stub")

    def learn(self, total_timesteps, callback=None):
        callbacks = callback if isinstance(callback, list) else ([callback] if callback else [])
        for cb in callbacks:
            cb.model = self
        if callbacks:
            callbacks[0].num_timesteps = 1
            callbacks[0]._on_step()
            callbacks[0]._on_training_end()


def test_run_metadata_contains_lock_fields(tmp_path, monkeypatch):
    from prl.train import run_training
    from prl.features import VolatilityFeatures

    monkeypatch.chdir(tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    asset_list = ["AAA", "BBB"]
    feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True}
    cost_params = {"transaction_cost": 0.0}
    env_signature = compute_env_signature(asset_list, 2, 2, feature_flags, cost_params, "v1")
    manifest = {
        "asset_list": asset_list,
        "num_assets": 2,
        "L": 2,
        "Lv": 2,
        "obs_dim_expected": 8,
        "env_signature_hash": env_signature,
        "feature_flags": feature_flags,
        "cost_params": cost_params,
        "env_schema_version": "v1",
    }
    manifest["data_manifest_hash"] = sha256_bytes(canonical_json(manifest))
    (processed_dir / "data_manifest.json").write_text(json.dumps(manifest))

    def _fake_prepare_market_and_features(*args, **kwargs):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        returns = pd.DataFrame(0.001, index=dates, columns=asset_list)
        prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=asset_list)
        vf = VolatilityFeatures(
            volatility=pd.DataFrame(0.02, index=dates, columns=asset_list),
            portfolio_scalar=pd.Series(0.0, index=dates),
            mean=pd.Series(0.0, index=asset_list),
            std=pd.Series(1.0, index=asset_list),
            stats_path=Path("stats.npz"),
        )
        return MarketData(prices=prices, returns=returns), vf

    monkeypatch.setattr("prl.train.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("prl.train.build_env_for_range", lambda *args, **kwargs: "env")
    monkeypatch.setattr("prl.train.train_baseline_model", lambda *args, **kwargs: DummyModel())

    cfg = {
        "mode": "smoke",
        "config_path": "configs/test.yaml",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-05",
            "test_start": "2020-01-01",
            "test_end": "2020-01-05",
        },
        "data": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(processed_dir)},
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

    run_training(cfg, "baseline", seed=0, raw_dir=tmp_path / "raw", processed_dir=processed_dir, output_dir=tmp_path / "models", force_refresh=False)

    meta_dir = tmp_path / "outputs" / "reports"
    files = sorted(meta_dir.glob("run_metadata_*.json"))
    assert files, "run_metadata_*.json not created"
    data = json.loads(files[-1].read_text())
    for key in [
        "config_path",
        "data_manifest_path",
        "data_manifest_hash",
        "asset_list",
        "num_assets",
        "L",
        "Lv",
        "obs_dim_expected",
        "env_signature_hash",
    ]:
        assert key in data
    assert data["asset_list"] == asset_list
    assert data["obs_dim_expected"] == 8
