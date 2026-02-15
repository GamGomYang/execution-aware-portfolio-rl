import math

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from prl.envs import Dow30PortfolioEnv, EnvConfig
from prl.eval import run_backtest_episode_detailed


def _make_env(risk_lambda: float) -> Dow30PortfolioEnv:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    arithmetic_returns = np.full((len(dates), 2), 0.01, dtype=np.float32)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    cfg = EnvConfig(
        returns=returns,
        volatility=vol,
        window_size=1,
        transaction_cost=0.0,
        logit_scale=1.0,
        risk_lambda=risk_lambda,
        risk_penalty_type="r2",
    )
    env = Dow30PortfolioEnv(cfg)
    env.reset()
    return env


def test_risk_lambda_changes_reward_not_net_log_return():
    action = np.zeros(2, dtype=np.float32)
    env_no_risk = _make_env(0.0)
    env_risk = _make_env(10.0)

    _, reward0, _, _, info0 = env_no_risk.step(action)
    _, reward1, _, _, info1 = env_risk.step(action)

    assert info0["log_return_net"] == pytest.approx(info1["log_return_net"], rel=1e-9)
    assert info1["risk_penalty"] > 0.0
    assert reward1 < reward0
    assert info1["reward_no_risk"] == pytest.approx(info1["log_return_net"], rel=1e-9)


def test_eval_uses_log_return_net_for_net_return_exp():
    env = DummyVecEnv([lambda: _make_env(10.0)])

    class DummyModel:
        def __init__(self, action_dim: int):
            self.action_dim = action_dim

        def predict(self, obs, deterministic: bool = True):
            action = np.zeros((1, self.action_dim), dtype=np.float32)
            return action, None

    model = DummyModel(action_dim=2)
    _, trace = run_backtest_episode_detailed(model, env)

    log_return_net = trace["log_returns_net"][0]
    reward = trace["rewards"][0]
    net_exp = trace["net_returns_exp"][0]

    assert not math.isclose(log_return_net, reward, rel_tol=1e-9)
    assert net_exp == pytest.approx(math.exp(log_return_net) - 1.0, rel=1e-9)
    assert not math.isclose(net_exp, math.exp(reward) - 1.0, rel_tol=1e-9)
