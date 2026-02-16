from pathlib import Path
import json
import pandas as pd

from scripts import make_roadmap_report


def test_make_roadmap_report_outputs(tmp_path, monkeypatch):
    reports_dir = tmp_path / "exp_run" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_index = {
        "exp_name": "gate_test",
        "config_path": "configs/exp/gate0_smoke_W1.yaml",
        "run_ids": ["r1"],
        "metrics_path": str(reports_dir / "metrics.csv"),
        "regime_metrics_path": str(reports_dir / "regime_metrics.csv"),
        "reports_dir": str(reports_dir),
    }
    (reports_dir / "run_index.json").write_text(json.dumps(run_index))

    metrics = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "eval_id": "r1",
                "eval_window": "W1",
                "model_type": "prl_sac",
                "seed": 0,
                "period": "test",
                "cumulative_return": 0.05,
                "cumulative_return_net_exp": 0.04,
                "sharpe": 1.0,
                "sharpe_net_exp": 1.1,
                "steps": 10,
            }
        ]
    )
    metrics.to_csv(reports_dir / "metrics.csv", index=False)
    regime = pd.DataFrame(
        [
            {"run_id": "r1", "model_type": "prl_sac", "seed": 0, "regime": "mid", "period": "test", "sharpe_net_exp": 0.8, "cumulative_return_net_exp": 0.01}
        ]
    )
    regime.to_csv(reports_dir / "regime_metrics.csv", index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["make_roadmap_report", "--run-index-paths", str(reports_dir / "run_index.json"), "--output-dir", str(tmp_path / "out")],
    )
    make_roadmap_report.main()

    assert (tmp_path / "out" / "roadmap_results.md").exists()
    assert (tmp_path / "out" / "final_decision.md").exists()
    assert (tmp_path / "out" / "final_candidates_table.csv").exists()
