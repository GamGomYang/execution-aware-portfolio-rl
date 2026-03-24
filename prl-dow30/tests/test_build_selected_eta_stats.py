import pandas as pd

from scripts.build_selected_eta_stats import _build_summary


def test_build_selected_eta_stats_includes_wilcoxon_and_bootstrap_columns():
    rows = []
    for seed, sel_sharpe, base_sharpe in [(0, 1.1, 1.0), (1, 1.2, 1.0), (2, 1.3, 1.1)]:
        rows.append(
            {
                "kappa": 0.001,
                "seed": seed,
                "pair_eta": 0.2,
                "sharpe_net_lin": sel_sharpe,
                "cagr": 0.10 + 0.01 * seed,
                "maxdd": -0.10 - 0.01 * seed,
                "avg_turnover_exec": 0.01,
            }
        )
        rows.append(
            {
                "kappa": 0.001,
                "seed": seed,
                "pair_eta": 1.0,
                "sharpe_net_lin": base_sharpe,
                "cagr": 0.09 + 0.01 * seed,
                "maxdd": -0.11 - 0.01 * seed,
                "avg_turnover_exec": 0.03,
            }
        )

    summary, seedwise = _build_summary(
        pd.DataFrame(rows),
        selected_eta=0.2,
        baseline_eta=1.0,
        n_boot=200,
        bootstrap_alpha=0.05,
        bootstrap_seed=123,
    )
    assert not seedwise.empty
    assert "wilcoxon_two_sided_p_delta_sharpe" in summary.columns
    assert "bootstrap_ci_low_median_delta_sharpe_net_lin" in summary.columns
    assert "bootstrap_ci_high_median_delta_sharpe_net_lin" in summary.columns
    row = summary.iloc[0]
    assert row["bootstrap_ci_low_median_delta_sharpe_net_lin"] <= row["bootstrap_ci_high_median_delta_sharpe_net_lin"]


def test_build_selected_eta_stats_wilcoxon_all_zero_sets_skip_reason():
    rows = []
    for seed in [0, 1]:
        rows.append(
            {
                "kappa": 0.0,
                "seed": seed,
                "pair_eta": 0.2,
                "sharpe_net_lin": 1.0,
                "cagr": 0.1,
                "maxdd": -0.1,
                "avg_turnover_exec": 0.01,
            }
        )
        rows.append(
            {
                "kappa": 0.0,
                "seed": seed,
                "pair_eta": 1.0,
                "sharpe_net_lin": 1.0,
                "cagr": 0.1,
                "maxdd": -0.1,
                "avg_turnover_exec": 0.01,
            }
        )

    summary, _ = _build_summary(
        pd.DataFrame(rows),
        selected_eta=0.2,
        baseline_eta=1.0,
        n_boot=100,
        bootstrap_alpha=0.05,
        bootstrap_seed=0,
    )
    row = summary.iloc[0]
    assert row["wilcoxon_two_sided_p_delta_sharpe"] == 1.0
    assert row["wilcoxon_skipped_reason_delta_sharpe"] == "all_zero"
