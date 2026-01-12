import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prl.data import MarketData
from prl.metrics import PortfolioMetrics
from prl.train import _write_run_metadata


def test_paper_gate_artifacts(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "data_manifest.json").write_text(json.dumps({"dummy": True}))

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
            observation_space = None

        return DummyEnv()

    def _fake_run_training(
        config,
        model_type,
        seed,
        raw_dir="data/raw",
        processed_dir="data/processed",
        output_dir="outputs/models",
        force_refresh=True,
        offline=False,
        cache_only=False,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_type}_seed{seed}_final.zip"
        model_path.write_bytes(b"dummy")

        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_type}_seed{seed}_train_log.csv"
        log_path.write_text("timesteps,actor_loss\n1,0.0\n2,0.0\n")

        _write_run_metadata(Path("outputs/reports"), config, seed, config.get("mode", ""), model_type, model_path, log_path)
        return model_path

    def _fake_load_model(*args, **kwargs):
        return object()

    def _fake_run_backtest_episode(*args, **kwargs):
        return PortfolioMetrics(
            total_reward=1.0,
            avg_reward=0.1,
            cumulative_return=0.05,
            avg_turnover=0.12,
            total_turnover=1.2,
            sharpe=1.0,
            max_drawdown=-0.1,
            steps=10,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.run_all.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_all.build_env_for_range", _fake_build_env_for_range)
    monkeypatch.setattr("scripts.run_all.create_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_all.run_training", _fake_run_training)
    monkeypatch.setattr("scripts.run_all.load_model", _fake_load_model)
    monkeypatch.setattr("scripts.run_all.run_backtest_episode", _fake_run_backtest_episode)

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
    assert set(metrics_df["model_type"]) == {"baseline", "prl"}

    summary_df = pd.read_csv(summary_path)
    assert "avg_turnover_mean" in summary_df.columns
    assert "total_turnover_mean" in summary_df.columns

    assert (tmp_path / "outputs" / "models" / "baseline_seed0_final.zip").exists()
    assert (tmp_path / "outputs" / "models" / "prl_seed0_final.zip").exists()

    log_base = tmp_path / "outputs" / "logs"
    baseline_log = log_base / "baseline_seed0_train_log.csv"
    prl_log = log_base / "prl_seed0_train_log.csv"
    assert baseline_log.exists() and baseline_log.stat().st_size > 0
    assert prl_log.exists() and prl_log.stat().st_size > 0

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
