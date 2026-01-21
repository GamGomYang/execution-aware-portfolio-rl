import numpy as np
import pandas as pd
import pytest

from prl.baselines import inverse_vol_weights, run_baseline_strategy


def test_inverse_vol_uses_previous_volatility_row():
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    log_returns = np.zeros((2, 2), dtype=np.float64)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    volatility = pd.DataFrame(
        [
            [0.1, 0.2],
            [10.0, 0.1],
        ],
        index=dates,
        columns=["A", "B"],
    )

    metrics = run_baseline_strategy(returns, volatility, "inverse_vol_risk_parity", transaction_cost=0.0)

    w_target0 = inverse_vol_weights(np.array([0.1, 0.2], dtype=np.float64))
    w_prev = np.array([0.5, 0.5], dtype=np.float64)
    expected_turnover = np.abs(w_target0 - w_prev).sum()

    assert metrics.total_turnover == pytest.approx(expected_turnover)
