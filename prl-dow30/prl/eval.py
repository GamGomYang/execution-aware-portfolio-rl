from __future__ import annotations

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from .metrics import PortfolioMetrics, compute_metrics
from .prl import PRLAlphaScheduler
from .sb3_prl_sac import PRLSAC


def load_model(model_path: Path, model_type: str, env: DummyVecEnv, scheduler: PRLAlphaScheduler | None = None):
    if model_type == "prl":
        model = PRLSAC.load(model_path, env=env)
        model.scheduler = scheduler
    else:
        model = SAC.load(model_path, env=env)
    return model


def run_backtest_episode(model, env: DummyVecEnv) -> PortfolioMetrics:
    obs = env.reset()
    done = False
    rewards = []
    portfolio_returns = []
    turnovers = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_vec, done_vec, info_list = env.step(action)
        reward = float(reward_vec[0])
        done = bool(done_vec[0])
        rewards.append(reward)
        info = info_list[0]
        portfolio_returns.append(info.get("portfolio_return", 0.0))
        turnovers.append(info.get("turnover", 0.0))
    return compute_metrics(rewards, portfolio_returns, turnovers)


def run_backtest_timeseries(model, env: DummyVecEnv) -> tuple[PortfolioMetrics, dict]:
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    rewards = []
    portfolio_returns = []
    turnovers = []
    dates = []
    vol_portfolio = []
    base_env = env.envs[0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_vec, done_vec, info_list = env.step(action)
        reward = float(reward_vec[0])
        done = bool(done_vec[0])
        rewards.append(reward)
        info = info_list[0]
        portfolio_returns.append(info.get("portfolio_return", 0.0))
        turnovers.append(info.get("turnover", 0.0))
        idx = base_env.current_step - 1
        dates.append(base_env.returns.index[idx])
        vol_portfolio.append(float(base_env.volatility.iloc[idx].mean()))
    metrics = compute_metrics(rewards, portfolio_returns, turnovers)
    ts = {
        "dates": dates,
        "portfolio_return": portfolio_returns,
        "turnover": turnovers,
        "vol_portfolio": vol_portfolio,
    }
    return metrics, ts
