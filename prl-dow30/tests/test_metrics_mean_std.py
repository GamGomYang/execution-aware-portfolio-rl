import numpy as np
import pytest

from prl.metrics import compute_metrics


def test_compute_metrics_mean_std_and_sharpe_zero_mean():
    returns = [0.01, -0.01]
    rewards = [np.log1p(r) for r in returns]  # so that exp(reward)-1 == returns
    metrics = compute_metrics(rewards, returns, turnovers=[0.0, 0.0])
    assert pytest.approx(metrics.mean_daily_return_gross, rel=1e-9, abs=1e-9) == 0.0
    assert pytest.approx(metrics.std_daily_return_gross, rel=1e-6) == 0.01
    assert pytest.approx(metrics.sharpe, rel=1e-6) == 0.0
    assert pytest.approx(metrics.mean_daily_net_return_exp, rel=1e-9, abs=1e-9) == 0.0
    assert pytest.approx(metrics.std_daily_net_return_exp, rel=1e-6) == 0.01
    assert pytest.approx(metrics.sharpe_net_exp, rel=1e-6) == 0.0


def test_compute_metrics_zero_std_guard():
    returns = [0.0, 0.0]
    metrics = compute_metrics(returns, returns, turnovers=[0.0, 0.0])
    assert metrics.std_daily_return_gross == 0.0
    assert metrics.sharpe == 0.0
