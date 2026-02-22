import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from prl.data import MarketData
from prl.features import VolatilityFeatures
from prl.metrics import PortfolioMetrics
from prl.utils.signature import compute_env_signature


def test_run_eval_falls_back_to_config_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / "custom_outputs"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = "20200101T000000Z_deadbeef_seed0_baseline_abcd"
    model_path = models_dir / f"{run_id}_final.zip"
    model_path.write_text("stub")
    asset_list = ["AAA", "BBB"]
    env_signature = compute_env_signature(
        asset_list,
        2,
        2,
        feature_flags={"returns_window": True, "volatility": True, "prev_weights": True},
        cost_params={"transaction_cost": 0.0},
        schema_version="v1",
    )
    meta = {
        "run_id": run_id,
        "seed": 0,
        "model_type": "baseline",
        "created_at": "2020-01-01T00:00:00+00:00",
        "asset_list": asset_list,
        "num_assets": 2,
        "L": 2,
        "Lv": 2,
        "obs_dim_expected": 8,
        "env_signature_hash": env_signature,
        "artifact_paths": {"model_path": str(model_path), "train_log_path": str(output_root / "logs" / "train_stub.csv")},
    }
    (reports_dir / f"run_metadata_{run_id}.json").write_text(json.dumps(meta))

    cfg = {
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-10",
            "test_start": "2020-01-01",
            "test_end": "2020-01-10",
        },
        "output": {"root": str(output_root)},
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
        mean=0.0,
        std=1.0,
        stats_path=tmp_path / "stats.json",
    )
    features.stats_path.write_text(json.dumps({"mean": 0.0, "std": 1.0}))

    calls = {}

    def _fake_prepare_market_and_features(*args, **kwargs):
        calls["prepare_called"] = True
        return market, features

    def _fake_load_model(path, model_type, env, scheduler=None):
        calls["model_path"] = path
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
            total_turnover=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            steps=0,
        )

    class DummyEnv:
        def __init__(self, returns):
            self.returns = returns
            self.num_assets = returns.shape[1]
            self.window_size = 2
            self.observation_space = SimpleNamespace(shape=(self.num_assets * (self.window_size + 2),))
            self.cfg = SimpleNamespace(transaction_cost=0.0)

    monkeypatch.setattr("scripts.run_eval.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_eval.build_env_for_range", lambda *args, **kwargs: DummyEnv(returns=market.returns))
    monkeypatch.setattr("scripts.run_eval.create_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_eval.load_model", _fake_load_model)
    monkeypatch.setattr("scripts.run_eval.run_backtest_episode", _fake_run_backtest_episode)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_eval.py",
            "--config",
            str(cfg_path),
            "--model-type",
            "baseline",
            "--seed",
            "0",
        ],
    )

    from scripts import run_eval as run_eval_script

    run_eval_script.main()

    assert calls.get("prepare_called", False)
    assert calls.get("backtest_called", False)
    assert calls["model_path"] == model_path
    assert (output_root / "reports" / "metrics.csv").exists()
