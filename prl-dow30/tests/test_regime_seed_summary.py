import pandas as pd
import pytest

from scripts.analyze_paper_results import compute_regime_seed_summary


def test_regime_seed_summary():
    rows = []
    for model_type in ["baseline_sac", "prl_sac"]:
        for seed in [0, 1]:
            for regime, sharpe in [("low", 0.5), ("mid", 1.0), ("high", 1.5)]:
                rows.append(
                    {
                        "run_id": f"run_{model_type}_{seed}",
                        "model_type": model_type,
                        "seed": seed,
                        "regime": regime,
                        "sharpe": sharpe + (0.1 if seed == 1 else 0.0),
                        "max_drawdown": -0.1,
                        "avg_turnover": 0.2,
                        "cumulative_return": 0.05,
                    }
                )
    df = pd.DataFrame(rows)
    summary = compute_regime_seed_summary(df)
    row = summary[(summary["model_type"] == "baseline_sac") & (summary["regime"] == "mid")].iloc[0]
    assert row["n_seeds"] == 2
    assert row["sharpe_mean"] == pytest.approx(1.05)
    assert "sharpe_ci_low" in summary.columns
    assert "sharpe_ci_high" in summary.columns
