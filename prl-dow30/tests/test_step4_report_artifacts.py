import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_step4_report_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run123"
    outputs = Path("outputs")
    logs_dir = outputs / "logs"
    reports_dir = outputs / "reports"
    figs_dir = outputs / "figures" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "schema_version": ["1.1", "1.1"],
            "run_id": [run_id, run_id],
            "model_type": ["prl", "prl"],
            "seed": [0, 0],
            "timesteps": [1, 2],
            "actor_loss": [0.1, 0.2],
            "critic_loss": [0.3, 0.4],
            "entropy_loss": [0.5, 0.6],
            "ent_coef": [0.2, 0.2],
            "ent_coef_loss": [0.0, 0.0],
            "alpha_obs_mean": [0.2, 0.21],
            "alpha_next_mean": [0.2, 0.21],
            "prl_prob_mean": [0.4, 0.5],
            "vz_mean": [0.1, 0.2],
            "alpha_raw_mean": [0.22, 0.23],
            "alpha_clamped_mean": [0.22, 0.23],
            "emergency_rate": [0.0, 0.1],
            "beta_effective_mean": [0.5, 0.6],
        }
    )
    train_df.to_csv(logs_dir / f"train_{run_id}.csv", index=False)

    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    trace_df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.01, -0.005, 0.002],
            "reward": [0.01, -0.005, 0.002],
            "turnover": [0.1, 0.2, 0.1],
            "turnover_target_change": [0.05, 0.05, 0.05],
            "run_id": [run_id] * 3,
            "model_type": ["prl_sac"] * 3,
            "seed": [0] * 3,
            "vz": [0.1, 0.5, 1.0],
            "regime": ["low", "mid", "high"],
        }
    )
    trace_df.to_parquet(reports_dir / f"trace_{run_id}.parquet", index=False)

    regime_df = pd.DataFrame(
        {
            "run_id": [run_id, run_id, run_id],
            "model_type": ["prl_sac", "prl_sac", "prl_sac"],
            "seed": [0, 0, 0],
            "regime": ["low", "mid", "high"],
            "total_reward": [0.1, 0.2, 0.3],
            "avg_reward": [0.01, 0.02, 0.03],
            "cumulative_return": [0.01, 0.02, 0.03],
            "avg_turnover": [0.1, 0.2, 0.3],
            "total_turnover": [1.0, 2.0, 3.0],
            "sharpe": [0.5, 0.6, 0.7],
            "max_drawdown": [-0.1, -0.2, -0.3],
            "steps": [3, 3, 3],
        }
    )
    regime_df.to_csv(reports_dir / "regime_metrics.csv", index=False)

    thresholds = {"q33": 0.2, "q66": 0.7}
    (reports_dir / f"regime_thresholds_{run_id}.json").write_text(json.dumps(thresholds))

    meta = {
        "run_id": run_id,
        "model_type": "prl",
        "seed": 0,
        "created_at": "2020-01-01T00:00:00Z",
        "config_hash": "deadbeef",
        "artifact_paths": {"train_log_path": str(logs_dir / f"train_{run_id}.csv")},
    }
    (reports_dir / f"run_metadata_{run_id}.json").write_text(json.dumps(meta))

    from scripts import make_step4_report

    monkeypatch.setattr("sys.argv", ["make_step4_report.py", "--run-id", run_id])
    make_step4_report.main()

    assert (reports_dir / f"step4_report_{run_id}.md").exists()
    for name in [
        "train_losses.png",
        "alpha_beta_emergency.png",
        "equity_curve.png",
        "equity_by_regime.png",
        "turnover_by_regime.png",
    ]:
        fig_path = figs_dir / name
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0
