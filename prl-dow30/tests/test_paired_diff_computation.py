import pandas as pd
import pytest

from scripts.analyze_paper_results import compute_paired_diffs


def test_paired_diff_computation():
    df = pd.DataFrame(
        [
            {"model_type": "baseline_sac", "seed": 0, "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.2, "cumulative_return": 0.5},
            {"model_type": "prl_sac", "seed": 0, "sharpe": 1.5, "max_drawdown": -0.05, "avg_turnover": 0.25, "cumulative_return": 0.7},
            {"model_type": "baseline_sac", "seed": 1, "sharpe": 0.8, "max_drawdown": -0.2, "avg_turnover": 0.3, "cumulative_return": 0.4},
            {"model_type": "prl_sac", "seed": 1, "sharpe": 0.9, "max_drawdown": -0.15, "avg_turnover": 0.35, "cumulative_return": 0.45},
        ]
    )
    diffs = compute_paired_diffs(df)
    assert diffs.loc[diffs["seed"] == 0, "delta_sharpe"].iloc[0] == pytest.approx(0.5)
    assert diffs.loc[diffs["seed"] == 0, "delta_mdd"].iloc[0] == pytest.approx(0.05)
    assert diffs.loc[diffs["seed"] == 1, "delta_turnover"].iloc[0] == pytest.approx(0.05)
    assert diffs.loc[diffs["seed"] == 1, "delta_cumret"].iloc[0] == pytest.approx(0.05)
