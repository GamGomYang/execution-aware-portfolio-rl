import numpy as np
import pandas as pd

from prl.envs import Dow30PortfolioEnv, EnvConfig


def _make_env(random_reset: bool) -> Dow30PortfolioEnv:
    dates = pd.date_range("2020-01-01", periods=12, freq="B")
    arithmetic_returns = np.full((len(dates), 2), 0.01, dtype=np.float32)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(0.02, index=dates, columns=["A", "B"])
    cfg = EnvConfig(
        returns=returns,
        volatility=vol,
        window_size=3,
        transaction_cost=0.0,
        random_reset=random_reset,
    )
    return Dow30PortfolioEnv(cfg)


def test_env_random_reset_disabled():
    env = _make_env(random_reset=False)
    for _ in range(3):
        _, info = env.reset()
        assert info["start_step"] == env.window_size
        assert env.current_step == env.window_size


def test_env_random_reset_changes_start_step(monkeypatch):
    env = _make_env(random_reset=True)
    expected_steps = [env.window_size, env.window_size + 1, env.window_size + 2]
    step_iter = iter(expected_steps)
    calls = []

    def _fake_randint(low, high=None, size=None, dtype=int):
        calls.append((low, high))
        return next(step_iter)

    monkeypatch.setattr("numpy.random.randint", _fake_randint)

    seen = []
    for _ in range(3):
        _, info = env.reset()
        seen.append(info["start_step"])
        assert env.window_size <= info["start_step"] <= len(env.returns) - 1
        assert info["start_step"] == env.current_step

    assert seen == expected_steps
    assert all(low == env.window_size and high == len(env.returns) for low, high in calls)
