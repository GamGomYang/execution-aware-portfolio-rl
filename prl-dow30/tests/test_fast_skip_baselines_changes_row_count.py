from prl.baselines import BASELINE_NAMES
from tests.helpers import run_stubbed_run_all


def test_fast_skip_baselines_changes_row_count(tmp_path, monkeypatch):
    ctx = run_stubbed_run_all(
        tmp_path,
        monkeypatch,
        eval_cfg={"run_baselines": False, "write_step4": False},
        model_types=["baseline", "prl"],
        seeds=[0],
    )
    metrics_path = ctx["reports_dir"] / "metrics.csv"
    df = __import__("pandas").read_csv(metrics_path)
    assert df[df["model_type"].isin(BASELINE_NAMES)].empty
    run_ids = set(ctx["run_index"]["run_ids"])
    assert "baseline_strategies_seed0" not in run_ids
