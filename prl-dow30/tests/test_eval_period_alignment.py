import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from prl.baselines import BASELINE_NAMES
from prl.data import MarketData
from prl.features import VolatilityFeatures
from prl.metrics import PortfolioMetrics
from prl.train import _write_run_metadata


def test_eval_period_alignment(tmp_path, monkeypatch):
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
        "seeds": [0],
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
    volatility = pd.DataFrame(0.1, index=dates, columns=["AAA", "BBB"])
    market = MarketData(prices=pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"]), returns=returns)

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": 0.0, "std": 1.0}))
    features = VolatilityFeatures(
        volatility=volatility,
        portfolio_scalar=pd.Series(0.0, index=dates),
        mean=0.0,
        std=1.0,
        stats_path=stats_path,
    )

    def _fake_prepare_market_and_features(*args, **kwargs):
        return market, features

    def _fake_run_training(*args, **kwargs):
        config = kwargs.get("config", cfg)
        model_type = kwargs.get("model_type", "baseline")
        seed = kwargs.get("seed", 0)
        processed_dir = Path(config["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "asset_list": ["AAA", "BBB"],
            "num_assets": 2,
            "L": config["env"]["L"],
            "Lv": config["env"]["Lv"],
            "obs_dim_expected": 2 * (config["env"]["L"] + 2),
            "env_schema_version": "v1",
        }
        (processed_dir / "data_manifest.json").write_text(json.dumps(manifest))
        run_id = f"runid_{model_type}_seed{seed}"
        model_path = Path("outputs/models") / f"{run_id}_final.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("stub")
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"train_{run_id}.csv"
        log_path.write_text("schema_version,run_id,model_type,seed,timesteps\n1.1,run,baseline,0,1\n")
        _write_run_metadata(Path("outputs/reports"), config, seed, config.get("mode", ""), model_type, run_id, model_path, log_path)
        return model_path

    def _fake_build_env_for_range(*args, **kwargs):
        class DummyEnv:
            def __init__(self, returns):
                self.returns = returns
                self.num_assets = returns.shape[1]
                self.window_size = 2
                self.observation_space = SimpleNamespace(shape=(self.num_assets * (self.window_size + 2),))
                self.cfg = SimpleNamespace(transaction_cost=0.0)

        return DummyEnv(returns=market.returns)

    def _fake_load_model(*args, **kwargs):
        return object()

    short_dates = list(dates[:3])

    def _fake_run_backtest_episode_detailed(*args, **kwargs):
        metrics = PortfolioMetrics(
            total_reward=0.3,
            avg_reward=0.1,
            cumulative_return=0.0,
            avg_turnover=0.0,
            total_turnover=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            steps=len(short_dates),
        )
        trace = {
            "dates": short_dates,
            "rewards": [0.1] * len(short_dates),
            "portfolio_returns": [0.0] * len(short_dates),
            "turnovers": [0.0] * len(short_dates),
            "turnover_target_changes": [0.0] * len(short_dates),
        }
        return metrics, trace

    monkeypatch.setattr("scripts.run_all.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_all.run_training", _fake_run_training)
    monkeypatch.setattr("scripts.run_all.build_env_for_range", _fake_build_env_for_range)
    monkeypatch.setattr("scripts.run_all.load_model", _fake_load_model)
    monkeypatch.setattr("prl.eval.run_backtest_episode_detailed", _fake_run_backtest_episode_detailed)
    monkeypatch.setattr("scripts.run_all.create_scheduler", lambda *args, **kwargs: None)

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["run_all.py", "--config", str(cfg_path), "--seeds", "0", "--model-types", "baseline", "prl", "--offline"],
    )

    from scripts import run_all as run_all_script

    run_all_script.main()

    metrics_df = pd.read_csv("outputs/reports/metrics.csv")
    baseline_df = metrics_df[metrics_df["model_type"].isin(BASELINE_NAMES)]
    assert not baseline_df.empty
    assert set(baseline_df["steps"]) == {len(short_dates)}
