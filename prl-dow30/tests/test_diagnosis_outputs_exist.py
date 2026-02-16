from pathlib import Path

import pandas as pd

from scripts import diagnosis_decomposition


def test_diagnosis_outputs_exist(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "eval_id": "r1",
                "eval_window": "W1",
                "model_type": "baseline_sac",
                "seed": 0,
                "period": "test",
                "cumulative_return": 0.05,
                "cumulative_return_net_exp": 0.04,
                "sharpe_net_exp": 1.2,
                "steps": 10,
            },
            {
                "run_id": "r1p",
                "eval_id": "r1p",
                "eval_window": "W1",
                "model_type": "prl_sac",
                "seed": 0,
                "period": "test",
                "cumulative_return": 0.06,
                "cumulative_return_net_exp": 0.05,
                "sharpe_net_exp": 1.4,
                "steps": 10,
            },
        ]
    )
    metrics_path = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    regime_metrics = pd.DataFrame(
        [
            {"run_id": "r1", "model_type": "baseline_sac", "seed": 0, "regime": "low", "period": "test", "cumulative_return": 0.01, "cumulative_return_net_exp": 0.009, "sharpe": 0.5, "sharpe_net_exp": 0.6, "avg_turnover": 0.1},
            {"run_id": "r1", "model_type": "baseline_sac", "seed": 0, "regime": "mid", "period": "test", "cumulative_return": 0.02, "cumulative_return_net_exp": 0.018, "sharpe": 0.6, "sharpe_net_exp": 0.7, "avg_turnover": 0.1},
            {"run_id": "r1", "model_type": "baseline_sac", "seed": 0, "regime": "high", "period": "test", "cumulative_return": 0.02, "cumulative_return_net_exp": 0.013, "sharpe": 0.7, "sharpe_net_exp": 0.8, "avg_turnover": 0.1},
            {"run_id": "r1p", "model_type": "prl_sac", "seed": 0, "regime": "low", "period": "test", "cumulative_return": 0.02, "cumulative_return_net_exp": 0.018, "sharpe": 0.6, "sharpe_net_exp": 0.7, "avg_turnover": 0.1},
            {"run_id": "r1p", "model_type": "prl_sac", "seed": 0, "regime": "mid", "period": "test", "cumulative_return": 0.02, "cumulative_return_net_exp": 0.016, "sharpe": 0.7, "sharpe_net_exp": 0.8, "avg_turnover": 0.1},
            {"run_id": "r1p", "model_type": "prl_sac", "seed": 0, "regime": "high", "period": "test", "cumulative_return": 0.02, "cumulative_return_net_exp": 0.015, "sharpe": 0.8, "sharpe_net_exp": 0.9, "avg_turnover": 0.1},
        ]
    )
    regime_path = tmp_path / "regime_metrics.csv"
    regime_metrics.to_csv(regime_path, index=False)

    trace_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "model_type": ["baseline_sac"] * 5,
            "turnover": [0.1, 0.2, 0.15, 0.12, 0.2],
        }
    )
    trace_path = tmp_path / "trace.parquet"
    trace_df.to_parquet(trace_path, index=False)

    out_dir = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnosis_decomposition",
            "--metrics",
            str(metrics_path),
            "--regime-metrics",
            str(regime_path),
            "--trace",
            str(trace_path),
            "--output-dir",
            str(out_dir),
        ],
    )

    diagnosis_decomposition.main()

    assert (out_dir / "diagnosis_decomposition.md").exists()
    assert (out_dir / "turnover_distribution.csv").exists()
    assert (out_dir / "regime_breakdown_net.csv").exists()
