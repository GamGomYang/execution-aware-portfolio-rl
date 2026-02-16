import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig, stable_softmax


def test_turnover_rebalance_reflects_inertia_exec_distance():
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
        rebalance_eta=0.2,
    )

    env = Dow30PortfolioEnv(cfg)
    env.reset()
    old_weights = np.array([0.5, 0.5], dtype=np.float32)
    env.prev_weights = old_weights.copy()

    action = np.array([3.0, -3.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    clipped_action = np.clip(action, -1.0, 1.0)
    w_target = stable_softmax(clipped_action, scale=1.0)
    w_exec = (1.0 - 0.2) * old_weights + 0.2 * w_target
    w_exec = np.clip(w_exec, 0.0, None)
    w_exec = w_exec / w_exec.sum()

    expected_turnover_target = float(np.abs(w_target - old_weights).sum())
    expected_turnover_exec = float(np.abs(w_exec - old_weights).sum())

    assert expected_turnover_exec < expected_turnover_target
    assert info["turnover"] == pytest.approx(expected_turnover_exec)
    assert info["turnover_exec"] == pytest.approx(expected_turnover_exec)
    assert info["turnover_target"] == pytest.approx(expected_turnover_target)
