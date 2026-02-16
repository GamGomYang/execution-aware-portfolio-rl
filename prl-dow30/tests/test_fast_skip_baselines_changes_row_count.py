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
    baseline_models = {"buy_and_hold_equal_weight", "daily_rebalanced_equal_weight", "inverse_vol_risk_parity"}
    assert df[df["model_type"].isin(baseline_models)].empty
    run_ids = set(ctx["run_index"]["run_ids"])
    assert "baseline_strategies_seed0" not in run_ids
