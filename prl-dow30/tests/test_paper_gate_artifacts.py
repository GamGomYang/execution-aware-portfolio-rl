import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from prl.data import MarketData
from prl.metrics import PortfolioMetrics
from prl.train import _generate_run_id, _write_run_metadata


def test_paper_gate_artifacts(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "asset_list": ["AAA", "BBB"],
        "num_assets": 2,
        "L": 2,
        "Lv": 2,
        "obs_dim_expected": 8,
        "env_schema_version": "v1",
    }
    (processed_dir / "data_manifest.json").write_text(json.dumps(manifest))

    cfg = {
        "mode": "paper_gate",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-10",
            "test_start": "2020-01-01",
            "test_end": "2020-01-10",
        },
        "data": {
            "processed_dir": str(processed_dir),
            "source": "yfinance_only",
            "offline": True,
            "require_cache": True,
            "paper_mode": True,
            "universe_policy": "availability_filtered",
        },
        "env": {"L": 2, "Lv": 2, "c_tc": 0.0, "logit_scale": 10.0},
        "prl": {
            "alpha0": 0.2,
            "beta": 1.0,
            "lambdav": 2.0,
            "bias": 0.0,
            "alpha_min": 0.01,
            "alpha_max": 1.0,
        },
        "sac": {
            "learning_rate": 0.0003,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 20000,
            "total_timesteps": 2000,
            "ent_coef": 0.2,
        },
        "seeds": [0],
    }

    config_path = tmp_path / "paper_gate.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    def _fake_prepare_market_and_features(*args, **kwargs):
        assert kwargs.get("offline") is True
        assert kwargs.get("cache_only") is True
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
        prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=["AAA", "BBB"])
        from prl.features import VolatilityFeatures

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps({"mean": 0.0, "std": 1.0}))
        vf = VolatilityFeatures(
            volatility=pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"]),
            portfolio_scalar=pd.Series(0.0, index=dates),
            mean=pd.Series(0.0, index=["AAA", "BBB"]),
            std=pd.Series(1.0, index=["AAA", "BBB"]),
            stats_path=stats_path,
        )
        return MarketData(prices=prices, returns=returns), vf

    def _fake_build_env_for_range(*args, **kwargs):
        class DummyEnv:
            def __init__(self):
                dates = pd.date_range("2020-01-01", periods=5, freq="B")
                self.returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
                self.num_assets = self.returns.shape[1]
                self.window_size = 2
                self.observation_space = SimpleNamespace(shape=(self.num_assets * (self.window_size + 2),))
                self.cfg = SimpleNamespace(transaction_cost=0.0)

        return DummyEnv()

    def _fake_run_training(*args, **kwargs):
        config = kwargs.get("config", cfg)
        model_type = kwargs.get("model_type", "baseline")
        seed = kwargs.get("seed", 0)
        output_dir = Path(kwargs.get("output_dir", "outputs/models"))
        reports_dir = Path(kwargs.get("reports_dir", "outputs/reports"))
        logs_dir = Path(kwargs.get("logs_dir", "outputs/logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = _generate_run_id(config, seed, model_type)
        model_path = output_dir / f"{run_id}_final.zip"
        model_path.write_bytes(b"dummy")

        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"train_{run_id}.csv"
        log_path.write_text(
            "schema_version,run_id,model_type,seed,timesteps,actor_loss,critic_loss,entropy_loss,ent_coef,ent_coef_loss,alpha_obs_mean,alpha_next_mean\n"
            "1.1,run,baseline,0,1,0.0,0.0,,0.2,,,\n"
        )

        reports_dir.mkdir(parents=True, exist_ok=True)
        _write_run_metadata(
            reports_dir,
            config,
            seed,
            config.get("mode", ""),
            model_type,
            run_id,
            model_path,
            log_path,
        )
        return model_path

    def _fake_load_model(*args, **kwargs):
        return object()

    def _fake_run_backtest_episode_detailed(*args, **kwargs):
        metrics = PortfolioMetrics(
            total_reward=1.0,
            avg_reward=0.1,
            cumulative_return=0.05,
            avg_turnover=0.12,
            total_turnover=1.2,
            sharpe=1.0,
            max_drawdown=-0.1,
            steps=10,
        )
        trace = {
            "dates": list(pd.date_range("2020-01-01", periods=3, freq="B")),
            "rewards": [0.1, 0.1, 0.1],
            "portfolio_returns": [0.01, 0.01, 0.01],
            "turnovers": [0.1, 0.1, 0.1],
            "turnover_target_changes": [0.05, 0.05, 0.05],
        }
        return metrics, trace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.run_all.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_all.build_env_for_range", _fake_build_env_for_range)
    monkeypatch.setattr("scripts.run_all.create_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_all.run_training", _fake_run_training)
    monkeypatch.setattr("scripts.run_all.load_model", _fake_load_model)
    monkeypatch.setattr("prl.eval.run_backtest_episode_detailed", _fake_run_backtest_episode_detailed)

    import scripts.run_all as run_all

    monkeypatch.setattr(
        "sys.argv",
        ["run_all", "--config", str(config_path), "--seeds", "0", "--offline"],
    )
    run_all.main()

    reports_dir = tmp_path / "outputs" / "reports"
    metrics_path = reports_dir / "metrics.csv"
    summary_path = reports_dir / "summary.csv"
    assert metrics_path.exists()
    assert summary_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    assert {"avg_turnover", "total_turnover"}.issubset(metrics_df.columns)
    assert set(metrics_df["model_type"]) == {
        "baseline_sac",
        "prl_sac",
        "buy_and_hold_equal_weight",
        "daily_rebalanced_equal_weight",
        "inverse_vol_risk_parity",
    }

    summary_df = pd.read_csv(summary_path)
    assert "avg_turnover_mean" in summary_df.columns
    assert "total_turnover_mean" in summary_df.columns

    meta_files = sorted(reports_dir.glob("run_metadata_*.json"))
    assert meta_files
    meta = json.loads(meta_files[-1].read_text())
    for key in [
        "seed",
        "mode",
        "model_type",
        "created_at",
        "config_hash",
        "python_version",
        "torch_version",
        "yfinance_version",
        "sb3_version",
        "data_manifest_hash",
        "artifact_paths",
    ]:
        assert key in meta
    assert "model_path" in meta["artifact_paths"]
    assert "train_log_path" in meta["artifact_paths"]
    assert Path(meta["artifact_paths"]["model_path"]).exists()
    assert Path(meta["artifact_paths"]["train_log_path"]).exists()
