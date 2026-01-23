import numpy as np
import pandas as pd

from prl.envs import Dow30PortfolioEnv, EnvConfig


def test_env_info_has_date():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    arithmetic_returns = np.full((5, 2), 0.01, dtype=np.float32)
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
    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["date"] == dates[env.window_size]
    assert "cost" in info
    assert info["cost"] == 0.0
