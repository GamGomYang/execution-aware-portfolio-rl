from pathlib import Path

from tests.helpers import run_stubbed_run_all


def test_fast_no_trace_still_writes_metrics(tmp_path, monkeypatch):
    ctx = run_stubbed_run_all(
        tmp_path,
        monkeypatch,
        eval_cfg={"write_trace": False, "run_baselines": True, "write_step4": False},
        model_types=["baseline"],
        seeds=[0],
    )
    reports_dir = ctx["reports_dir"]
    metrics_path = reports_dir / "metrics.csv"
    regime_path = reports_dir / "regime_metrics.csv"
    assert metrics_path.exists()
    assert regime_path.exists()
    trace_files = list(reports_dir.glob("trace_*.parquet"))
    assert not trace_files
    df = __import__("pandas").read_csv(metrics_path)
    assert "cumulative_return_net_exp" in df.columns
