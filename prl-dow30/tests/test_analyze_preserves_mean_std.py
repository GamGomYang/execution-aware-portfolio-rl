import pandas as pd

from scripts.analyze_paper_results import compute_paired_diffs, summarize_seed_stats


def test_summarize_seed_stats_keeps_mean_std_columns():
    rows = []
    for seed in [0, 1]:
        rows.append(
            {
                "run_id": f"b{seed}",
                "model_type": "baseline_sac",
                "seed": seed,
                "period": "test",
                "eval_window": "W1",
                "sharpe": 1.0 + seed * 0.1,
                "mean_daily_net_return_exp": 0.01 * seed,
                "std_daily_net_return_exp": 0.02,
                "avg_turnover": 0.1,
            }
        )
        rows.append(
            {
                "run_id": f"p{seed}",
                "model_type": "prl_sac",
                "seed": seed,
                "period": "test",
                "eval_window": "W1",
                "sharpe": 1.2 + seed * 0.1,
                "mean_daily_net_return_exp": 0.02 * seed,
                "std_daily_net_return_exp": 0.025,
                "avg_turnover": 0.08,
            }
        )
    df = pd.DataFrame(rows)
    summary = summarize_seed_stats(df)
    expected_cols = {
        "mean_daily_net_return_exp_mean",
        "mean_daily_net_return_exp_std",
        "std_daily_net_return_exp_mean",
        "std_daily_net_return_exp_std",
    }
    assert expected_cols.issubset(set(summary.columns))


def test_compute_paired_diffs_includes_mean_std():
    rows = []
    for seed in [0, 1]:
        rows.append(
            {
                "run_id": f"b{seed}",
                "model_type": "baseline_sac",
                "seed": seed,
                "period": "test",
                "eval_window": "W1",
                "mean_daily_net_return_exp": 0.01,
                "std_daily_net_return_exp": 0.02,
            }
        )
        rows.append(
            {
                "run_id": f"p{seed}",
                "model_type": "prl_sac",
                "seed": seed,
                "period": "test",
                "eval_window": "W1",
                "mean_daily_net_return_exp": 0.015,
                "std_daily_net_return_exp": 0.025,
            }
        )
    df = pd.DataFrame(rows)
    diffs = compute_paired_diffs(df, baseline_model_type="baseline_sac", prl_model_type="prl_sac")
    assert "delta_mean_daily_net_return_exp" in diffs.columns
    assert "delta_std_daily_net_return_exp" in diffs.columns
    assert all(abs(val - 0.005) < 1e-9 for val in diffs["delta_mean_daily_net_return_exp"])
    assert all(abs(val - 0.005) < 1e-9 for val in diffs["delta_std_daily_net_return_exp"])
