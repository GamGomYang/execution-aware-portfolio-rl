import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig, stable_softmax


def test_target_trace_uses_target_weights_when_eta_is_partial_and_kappa_zero():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    arithmetic_returns = np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.0]], dtype=np.float64)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    env = Dow30PortfolioEnv(
        EnvConfig(
            returns=returns,
            volatility=vol,
            window_size=1,
            transaction_cost=0.0,
            logit_scale=1.0,
            rebalance_eta=0.2,
        )
    )

    env.reset()
    old_weights = np.array([0.8, 0.2], dtype=np.float32)
    env.prev_weights = old_weights.copy()
    action = np.array([-3.0, 3.0], dtype=np.float32)

    _, _, _, _, info = env.step(action)

    clipped_action = np.clip(action, -1.0, 1.0)
    w_target = stable_softmax(clipped_action, scale=1.0)
    w_exec = (1.0 - 0.2) * old_weights + 0.2 * w_target
    w_exec = np.clip(w_exec, 0.0, None)
    w_exec = w_exec / w_exec.sum()
    arithmetic_t = arithmetic_returns[env.current_step - 1]

    expected_exec_return = float(np.dot(w_exec, arithmetic_t))
    expected_target_return = float(np.dot(w_target, arithmetic_t))

    assert info["tracking_error_l2"] > 0.0
    assert info["turnover_exec"] < info["turnover_target"]
    assert info["net_return_lin_exec"] == pytest.approx(expected_exec_return)
    assert info["net_return_lin_target"] == pytest.approx(expected_target_return)
    assert info["net_return_lin_target"] != pytest.approx(info["net_return_lin_exec"])
