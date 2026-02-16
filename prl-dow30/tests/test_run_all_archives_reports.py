from pathlib import Path

import pandas as pd

from tests.helpers import run_stubbed_run_all


def test_run_all_archives_reports(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    eval_cfg = {
        "run_baselines": False,
        "write_step4": False,
        "write_trace": False,
    }

    run_stubbed_run_all(
        tmp_path,
        monkeypatch,
        output_root=out_root,
        model_types=["baseline"],
        seeds=[0],
        eval_cfg=eval_cfg,
    )

    reports_dir = Path(out_root) / "reports"
    metrics_first = pd.read_csv(reports_dir / "metrics.csv")
    assert set(metrics_first["seed"]) == {0}

    run_stubbed_run_all(
        tmp_path,
        monkeypatch,
        output_root=out_root,
        model_types=["baseline"],
        seeds=[1],
        eval_cfg=eval_cfg,
    )

    metrics_latest = pd.read_csv(reports_dir / "metrics.csv")
    assert set(metrics_latest["seed"]) == {1}

    archive_dir = reports_dir / "archive"
    assert len(list(archive_dir.glob("metrics_*.csv"))) >= 2
    assert len(list(archive_dir.glob("summary_*.csv"))) >= 2
    assert len(list(archive_dir.glob("regime_metrics_*.csv"))) >= 2
