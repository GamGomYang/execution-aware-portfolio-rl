import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig


def test_turnover_target_change_fixed():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    arithmetic_returns = np.full((6, 2), 0.01, dtype=np.float32)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    env = Dow30PortfolioEnv(
        EnvConfig(
            returns=returns,
            volatility=vol,
            window_size=2,
            transaction_cost=0.0,
            logit_scale=1.0,
        )
    )
    env.reset()
    old_weights = np.array([0.6, 0.4], dtype=np.float32)
    env.prev_weights = old_weights.copy()
    action = np.array([-2.0, 2.0], dtype=np.float32)

    _, _, _, _, info = env.step(action)
    new_weights = env.prev_weights.copy()
    expected = np.abs(new_weights - old_weights).sum()
    assert info["turnover_target_change"] == pytest.approx(expected)
