import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from prl.data import MarketData, slice_frame
from prl.features import VolatilityFeatures
from prl.metrics import PortfolioMetrics


def test_eval_window_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "mode": "smoke",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-10",
            "test_start": "2019-01-01",
            "test_end": "2025-12-31",
        },
        "eval": {"eval_start": "2024-01-01", "eval_end": "2025-12-30", "name": "W1"},
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed", "require_cache": True, "paper_mode": True, "offline": True},
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

    dates = pd.date_range("2023-12-15", "2025-12-30", freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
    volatility = pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"])
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
        processed_dir = Path(cfg["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "asset_list": ["AAA", "BBB"],
            "num_assets": 2,
            "L": cfg["env"]["L"],
            "Lv": cfg["env"]["Lv"],
            "obs_dim_expected": 2 * (cfg["env"]["L"] + 2),
            "env_schema_version": "v1",
        }
        (processed_dir / "data_manifest.json").write_text(json.dumps(manifest))
        model_path = Path("outputs/models/runid_prl_seed0_final.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("stub")
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "train_runid_prl_seed0.csv"
        log_path.write_text("schema_version,run_id,model_type,seed,timesteps\n1.1,run,prl,0,1\n")
        from prl.train import _write_run_metadata

        _write_run_metadata(Path("outputs/reports"), cfg, 0, cfg.get("mode", ""), "prl", "runid_prl_seed0", model_path, log_path)
        return model_path

    def _fake_build_env_for_range(*, market, features, start, end, **kwargs):
        returns_slice = slice_frame(market.returns, start, end)
        class DummyEnv:
            def __init__(self, returns):
                self.returns = returns
                self.num_assets = returns.shape[1]
                self.window_size = cfg["env"]["L"]
                self.observation_space = SimpleNamespace(shape=(self.num_assets * (self.window_size + 2),))
                self.cfg = SimpleNamespace(transaction_cost=cfg["env"]["c_tc"])

        return DummyEnv(returns_slice)

    def _fake_load_model(*args, **kwargs):
        return object()

    def _fake_run_backtest_episode_detailed(model, env):
        eval_dates = list(env.returns.index[:3])
        metrics = PortfolioMetrics(
            total_reward=0.3,
            avg_reward=0.1,
            cumulative_return=0.0,
            avg_turnover=0.0,
            total_turnover=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            steps=len(eval_dates),
        )
        trace = {
            "dates": eval_dates,
            "rewards": [0.1] * len(eval_dates),
            "portfolio_returns": [0.0] * len(eval_dates),
            "turnovers": [0.0] * len(eval_dates),
            "turnover_target_changes": [0.0] * len(eval_dates),
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
        ["run_all.py", "--config", str(cfg_path), "--seeds", "0", "--model-types", "prl", "--offline"],
    )

    from scripts import run_all as run_all_script

    run_all_script.main()

    trace_path = Path("outputs/reports/trace_runid_prl_seed0.parquet")
    assert trace_path.exists()
    trace_df = pd.read_parquet(trace_path)
    assert trace_df["date"].min() >= pd.Timestamp("2024-01-01")
    assert trace_df["date"].max() <= pd.Timestamp("2025-12-30")
    assert set(trace_df["eval_window"]) == {"W1"}

    meta_data = json.loads(Path("outputs/reports/run_metadata_runid_prl_seed0.json").read_text())
    assert meta_data.get("eval_start_date") >= "2024-01-01"
    assert meta_data.get("eval_end_date") <= "2025-12-30"
    assert meta_data.get("evaluation", {}).get("eval_window") == "W1"
