import pandas as pd

from scripts.analyze_paper_results import analyze_metrics


def test_analyze_metrics_wilcoxon_all_zero(tmp_path):
    # baseline/prl identical -> diffs all zero
    rows = []
    for seed in [0, 1]:
        rows.append(
            {
                "run_id": f"b{seed}",
                "eval_id": "e1",
                "eval_window": "W1",
                "model_type": "baseline_sac",
                "seed": seed,
                "period": "test",
                "sharpe_net_exp": 1.0,
                "mean_daily_net_return_exp": 0.01,
                "std_daily_net_return_exp": 0.02,
            }
        )
        rows.append(
            {
                "run_id": f"p{seed}",
                "eval_id": "e1",
                "eval_window": "W1",
                "model_type": "prl_sac",
                "seed": seed,
                "period": "test",
                "sharpe_net_exp": 1.0,
                "mean_daily_net_return_exp": 0.01,
                "std_daily_net_return_exp": 0.02,
            }
        )
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    out_dir = tmp_path / "out"
    analyze_metrics(
        metrics_path,
        output_dir=out_dir,
        baseline_model_type="baseline_sac",
        prl_model_type="prl_sac",
        n_boot=10,
        run_ids=None,
    )
    summary_path = out_dir / "paired_stats_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert "wilcoxon_skipped_reason" in summary.columns
    # all-zero diffs should set p_value_wilcoxon to 1 and note reason
    assert (summary["p_value_wilcoxon"] == 1.0).any()
    assert (summary["wilcoxon_skipped_reason"] == "all_zero").any()
