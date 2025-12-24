import numpy as np
import pytest

from prl.metrics import compute_metrics


def test_sharpe_and_max_drawdown_on_toy_curve():
    rewards = [0.1, 0.05, -0.02, 0.03]
    returns = [0.02, -0.03, 0.01, -0.02]
    turnovers = [0.1, 0.08, 0.05, 0.03]

    metrics = compute_metrics(rewards, returns, turnovers)

    returns_arr = np.array(returns, dtype=np.float64)
    expected_sharpe = (returns_arr.mean() / returns_arr.std(ddof=0)) * np.sqrt(252)
    equity_curve = np.cumprod(1.0 + returns_arr)
    running_max = np.maximum.accumulate(equity_curve)
    expected_mdd = (equity_curve / running_max - 1.0).min()

    assert metrics.sharpe == pytest.approx(expected_sharpe)
    assert metrics.max_drawdown == pytest.approx(expected_mdd)
