import json
from pathlib import Path

import pandas as pd

from scripts import analyze_paper_results


def test_analyze_filters_by_run_index(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        [
            {"run_id": "keep", "eval_id": "keep", "model_type": "baseline_sac", "seed": 0, "period": "test", "sharpe": 0.1, "max_drawdown": -0.1, "cumulative_return": 0.01},
            {"run_id": "keep", "eval_id": "keep", "model_type": "prl_sac", "seed": 0, "period": "test", "sharpe": 0.2, "max_drawdown": -0.09, "cumulative_return": 0.02},
            {"run_id": "drop", "eval_id": "drop", "model_type": "baseline_sac", "seed": 1, "period": "test", "sharpe": 0.3, "max_drawdown": -0.05, "cumulative_return": 0.03},
            {"run_id": "drop", "eval_id": "drop", "model_type": "prl_sac", "seed": 1, "period": "test", "sharpe": 0.4, "max_drawdown": -0.04, "cumulative_return": 0.04},
        ]
    )
    metrics_path = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    regime_metrics = pd.DataFrame(
        [
            {"run_id": "keep", "model_type": "baseline_sac", "seed": 0, "regime": "low", "period": "test", "sharpe": 0.1, "max_drawdown": -0.1, "cumulative_return": 0.01, "avg_turnover": 0.0},
            {"run_id": "keep", "model_type": "prl_sac", "seed": 0, "regime": "low", "period": "test", "sharpe": 0.2, "max_drawdown": -0.09, "cumulative_return": 0.02, "avg_turnover": 0.0},
            {"run_id": "drop", "model_type": "baseline_sac", "seed": 1, "regime": "low", "period": "test", "sharpe": 0.3, "max_drawdown": -0.05, "cumulative_return": 0.03, "avg_turnover": 0.0},
            {"run_id": "drop", "model_type": "prl_sac", "seed": 1, "regime": "low", "period": "test", "sharpe": 0.4, "max_drawdown": -0.04, "cumulative_return": 0.04, "avg_turnover": 0.0},
        ]
    )
    regime_path = tmp_path / "regime_metrics.csv"
    regime_metrics.to_csv(regime_path, index=False)

    run_index_path = tmp_path / "run_index.json"
    run_index_path.write_text(json.dumps({"run_ids": ["keep"]}))

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_paper_results",
            "--metrics",
            str(metrics_path),
            "--regime-metrics",
            str(regime_path),
            "--output-dir",
            str(out_dir),
            "--run-index",
            str(run_index_path),
        ],
    )

    analyze_paper_results.main()

    paired_diffs = pd.read_csv(out_dir / "paired_seed_diffs.csv")
    assert len(paired_diffs) == 1
    assert set(paired_diffs["seed"]) == {0}

    summary_path = out_dir / "regime_seed_summary.csv"
    if summary_path.exists():
        regime_summary = pd.read_csv(summary_path)
        assert set(regime_summary["model_type"]) == {"baseline_sac", "prl_sac"}
