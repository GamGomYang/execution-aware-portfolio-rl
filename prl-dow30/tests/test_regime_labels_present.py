import pandas as pd
import pytest

from scripts.analyze_paper_results import compute_regime_seed_summary


def test_regime_labels_present():
    rows = [
        {"run_id": "run0", "model_type": "baseline_sac", "seed": 0, "regime": "low", "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.1, "cumulative_return": 0.1},
        {"run_id": "run0", "model_type": "baseline_sac", "seed": 0, "regime": "mid", "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.1, "cumulative_return": 0.1},
        {"run_id": "run0", "model_type": "baseline_sac", "seed": 0, "regime": "high", "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.1, "cumulative_return": 0.1},
        {"run_id": "run1", "model_type": "prl_sac", "seed": 0, "regime": "low", "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.1, "cumulative_return": 0.1},
        {"run_id": "run1", "model_type": "prl_sac", "seed": 0, "regime": "mid", "sharpe": 1.0, "max_drawdown": -0.1, "avg_turnover": 0.1, "cumulative_return": 0.1},
        # missing "high" for prl_sac seed 0
    ]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="REGIME_LABELS_MISSING"):
        compute_regime_seed_summary(df)
