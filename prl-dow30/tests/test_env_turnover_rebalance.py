import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig
from prl.metrics import post_return_weights


def test_turnover_rebalance_reflects_drift():
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    arithmetic_returns = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    cfg = EnvConfig(
        returns=returns,
        volatility=vol,
        window_size=1,
        transaction_cost=0.0,
        logit_scale=1.0,
    )

    env = Dow30PortfolioEnv(cfg)
    env.reset()
    env.prev_weights = np.array([0.5, 0.5], dtype=np.float32)

    action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    r_arith = arithmetic_returns[1]
    w_post = post_return_weights(np.array([0.5, 0.5], dtype=np.float64), r_arith)
    expected_w_post = np.array([0.5238095238, 0.4761904762], dtype=np.float64)
    assert w_post == pytest.approx(expected_w_post)
    expected_turnover = np.abs(np.array([0.5, 0.5], dtype=np.float64) - w_post).sum()

    assert expected_turnover > 0.0
    assert info["turnover"] == pytest.approx(expected_turnover)
