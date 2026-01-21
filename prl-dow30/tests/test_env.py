import math

import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig
from prl.metrics import post_return_weights


def build_toy_env(window_size: int = 2) -> Dow30PortfolioEnv:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    arithmetic_returns = np.array(
        [
            [0.01, 0.02],
            [0.015, -0.01],
            [0.02, 0.04],
            [-0.005, 0.01],
            [0.03, -0.02],
            [0.01, 0.015],
        ],
        dtype=np.float32,
    )
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    cfg = EnvConfig(
        returns=returns,
        volatility=vol,
        window_size=window_size,
        transaction_cost=0.001,
    )
    env = Dow30PortfolioEnv(cfg)
    env.reset()
    return env


def test_softmax_action_converts_to_valid_weights():
    env = build_toy_env()
    action = np.array([0.4, -0.7], dtype=np.float32)
    env.step(action)
    weights = env.prev_weights
    assert pytest.approx(weights.sum(), rel=1e-6) == 1.0
    assert np.all(weights >= 0)


def test_observation_shape_matches_spec():
    env = build_toy_env()
    obs, _ = env.reset()
    expected_dim = env.window_size * env.num_assets + 2 * env.num_assets
    assert obs.shape == (expected_dim,)


def test_reward_uses_previous_weights_with_turnover_penalty():
    env = build_toy_env()
    _ = env.reset()
    old_weights = np.array([0.7, 0.3], dtype=np.float32)
    env.prev_weights = old_weights.copy()
    action = np.array([3.0, -3.0], dtype=np.float32)

    _, reward, _, _, info = env.step(action)
    new_weights = env.prev_weights.copy()
    step_idx = env.current_step - 1
    returns_t = env.returns.iloc[step_idx].to_numpy()
    arithmetic_returns = np.expm1(returns_t)
    portfolio_return = float(np.dot(old_weights, arithmetic_returns))
    w_post = post_return_weights(old_weights, arithmetic_returns)
    turnover = np.abs(new_weights - w_post).sum()
    expected = math.log(max(1.0 + portfolio_return, env.cfg.log_clip)) - env.cfg.transaction_cost * turnover

    wrong_portfolio = float(np.dot(new_weights, arithmetic_returns))
    wrong = math.log(max(1.0 + wrong_portfolio, env.cfg.log_clip)) - env.cfg.transaction_cost * turnover

    assert reward == pytest.approx(expected)
    assert not math.isclose(reward, wrong)
    assert info["portfolio_return"] == pytest.approx(portfolio_return)


def test_turnover_matches_rebalance_distance():
    env = build_toy_env()
    env.prev_weights = np.array([0.6, 0.4], dtype=np.float32)
    action = np.array([-2.0, 2.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    new_weights = env.prev_weights
    step_idx = env.current_step - 1
    returns_t = env.returns.iloc[step_idx].to_numpy()
    arithmetic_returns = np.expm1(returns_t)
    w_post = post_return_weights(np.array([0.6, 0.4], dtype=np.float32), arithmetic_returns)
    expected_turnover = np.abs(new_weights - w_post).sum()
    assert info["turnover"] == pytest.approx(expected_turnover)
