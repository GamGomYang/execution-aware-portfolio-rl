import pandas as pd
import pytest

from scripts.analyze_paper_results import compute_paired_diffs


def test_pairing_baseline_prl_by_seed():
    df = pd.DataFrame(
        [
            {"model_type": "baseline_sac", "seed": 0, "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.2, "cumulative_return": 0.5},
            {"model_type": "baseline_sac", "seed": 1, "sharpe": 1.1, "max_drawdown": -0.1, "avg_turnover": 0.2, "cumulative_return": 0.5},
            {"model_type": "prl_sac", "seed": 0, "sharpe": 1.2, "max_drawdown": -0.1, "avg_turnover": 0.2, "cumulative_return": 0.5},
        ]
    )
    with pytest.raises(ValueError, match="SEED_PAIRING_MISMATCH"):
        compute_paired_diffs(df)
