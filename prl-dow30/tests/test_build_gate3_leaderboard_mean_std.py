import json
import tempfile
from pathlib import Path

import pandas as pd
from scripts import build_gate3_leaderboard


def _make_run_index(tmpdir: Path, name: str, rows: list[dict]) -> Path:
    reports = tmpdir / name / "reports"
    logs = tmpdir / name / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    metrics_path = reports / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    # create minimal PRL train logs for gate checks
    for row in rows:
        if "prl" not in str(row.get("model_type", "")):
            continue
        run_id = row["run_id"]
        seed = row.get("seed", 0)
        log_df = pd.DataFrame(
            [
                {
                    "schema_version": "1.1",
                    "run_id": run_id,
                    "model_type": "prl",
                    "seed": seed,
                    "timesteps": 100,
                    "emergency_rate": 0.0,
                    "prl_prob_p05": 0.1,
                    "prl_prob_p95": 0.9,
                    "prl_prob_std": 0.3,
                    "prl_prob_min": 0.0,
                    "prl_prob_max": 1.0,
                }
            ]
        )
        log_df.to_csv(logs / f"train_{run_id}.csv", index=False)
    run_index = {
        "run_ids": [r["run_id"] for r in rows],
        "metrics_path": str(metrics_path),
        "regime_metrics_path": str(reports / "regime_metrics.csv"),
        "reports_dir": str(reports),
        "logs_dir": str(logs),
        "config_path": "dummy.yaml",
        "exp_name": name,
    }
    run_index_path = reports / "run_index.json"
    run_index_path.write_text(json.dumps(run_index))
    return run_index_path


def test_build_gate3_leaderboard_produces_mean_std_and_deltas(tmp_path):
    # reference metrics (W1)
    ref_rows = [
        {
            "run_id": "ref0",
            "eval_window": "W1",
            "model_type": "baseline_sac",
            "seed": 0,
            "sharpe_net_exp": 1.0,
            "max_drawdown_net_exp": -0.2,
            "avg_turnover": 0.1,
            "mean_daily_net_return_exp": 0.01,
            "std_daily_net_return_exp": 0.02,
        },
        {
            "run_id": "ref1",
            "eval_window": "W1",
            "model_type": "baseline_sac",
            "seed": 1,
            "sharpe_net_exp": 1.1,
            "max_drawdown_net_exp": -0.2,
            "avg_turnover": 0.1,
            "mean_daily_net_return_exp": 0.011,
            "std_daily_net_return_exp": 0.02,
        },
    ]
    cand_rows = [
        {
            "run_id": "cand0",
            "eval_window": "W1",
            "model_type": "prl_sac",
            "seed": 0,
            "sharpe_net_exp": 1.4,
            "max_drawdown_net_exp": -0.19,
            "avg_turnover": 0.09,
            "mean_daily_net_return_exp": 0.015,
            "std_daily_net_return_exp": 0.018,
        },
        {
            "run_id": "cand1",
            "eval_window": "W1",
            "model_type": "prl_sac",
            "seed": 1,
            "sharpe_net_exp": 1.3,
            "max_drawdown_net_exp": -0.19,
            "avg_turnover": 0.09,
            "mean_daily_net_return_exp": 0.014,
            "std_daily_net_return_exp": 0.018,
        },
    ]
    ref_idx = _make_run_index(tmp_path, "ref_pack", ref_rows)
    cand_idx = _make_run_index(tmp_path, "cand_pack", cand_rows)

    out_dir = tmp_path / "out"
    args = [
        "--reference-run-index",
        str(ref_idx),
        "--candidate-run-indexes",
        str(cand_idx),
        "--output-dir",
        str(out_dir),
        "--pass-delta-sharpe",
        "0.2",
    ]
    build_gate3_leaderboard.main(args)

    leaderboard_path = out_dir / "Gate3_leaderboard.csv"
    assert leaderboard_path.exists()
    df = pd.read_csv(leaderboard_path)
    assert "delta_mean_daily_net_return_exp_vs_ref" in df.columns
    assert "delta_std_daily_net_return_exp_vs_ref" in df.columns
    assert (df["decision"] == "PASS").all()
