import numpy as np
import pandas as pd
import pytest

from prl.baselines import inverse_vol_weights, run_baseline_strategy


def test_baseline_turnover_semantics():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    arithmetic_returns = np.array(
        [
            [0.0, 0.0],
            [0.01, -0.01],
            [0.02, -0.02],
            [0.01, -0.01],
        ],
        dtype=np.float64,
    )
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    volatility = pd.DataFrame(
        [
            [0.2, 0.1],
            [0.25, 0.1],
            [0.3, 0.15],
            [0.35, 0.2],
        ],
        index=dates,
        columns=["A", "B"],
    )

    buy_hold = run_baseline_strategy(returns, volatility, "buy_and_hold_equal_weight", transaction_cost=0.0)
    assert buy_hold.avg_turnover <= 1e-12

    daily = run_baseline_strategy(returns, volatility, "daily_rebalanced_equal_weight", transaction_cost=0.0)
    assert daily.avg_turnover > 0.0

    inverse = run_baseline_strategy(returns, volatility, "inverse_vol_risk_parity", transaction_cost=0.0)
    assert inverse.avg_turnover >= 0.0

    weights = inverse_vol_weights(np.array([0.2, 0.1], dtype=np.float64))
    assert weights == pytest.approx(np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64))
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)
