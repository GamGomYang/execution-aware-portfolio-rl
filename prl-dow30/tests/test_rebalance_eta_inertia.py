import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig


def _make_env(rebalance_eta):
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    arithmetic_returns = np.full((len(dates), 2), 0.01, dtype=np.float32)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    return Dow30PortfolioEnv(
        EnvConfig(
            returns=returns,
            volatility=vol,
            window_size=1,
            transaction_cost=0.001,
            logit_scale=1.0,
            rebalance_eta=rebalance_eta,
        )
    )


def test_rebalance_eta_reduces_exec_turnover_and_keeps_weights_valid():
    action = np.array([3.0, -3.0], dtype=np.float32)
    old_weights = np.array([0.5, 0.5], dtype=np.float32)

    env_off = _make_env(None)
    env_off.reset()
    env_off.prev_weights = old_weights.copy()
    _, _, _, _, info_off = env_off.step(action)

    env_on = _make_env(0.1)
    env_on.reset()
    env_on.prev_weights = old_weights.copy()
    _, _, _, _, info_on = env_on.step(action)

    assert float(info_on["turnover_exec"]) <= float(info_off["turnover_exec"])
    assert np.isfinite(env_on.prev_weights).all()
    assert np.all(env_on.prev_weights >= 0.0)
    assert env_on.prev_weights.sum() == pytest.approx(1.0, abs=1e-6)
