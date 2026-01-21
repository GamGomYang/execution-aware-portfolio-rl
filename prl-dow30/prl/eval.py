from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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


def run_backtest_episode_detailed(model, env: DummyVecEnv) -> Tuple[PortfolioMetrics, Dict[str, List[float]]]:
    obs = env.reset()
    done = False
    rewards: List[float] = []
    portfolio_returns: List[float] = []
    turnovers: List[float] = []
    dates: List = []
    turnover_target_changes: List[float] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_vec, done_vec, info_list = env.step(action)
        reward = float(reward_vec[0])
        done = bool(done_vec[0])
        rewards.append(reward)
        info = info_list[0]
        portfolio_returns.append(info.get("portfolio_return", 0.0))
        turnovers.append(info.get("turnover", 0.0))
        dates.append(info.get("date"))
        turnover_target_changes.append(info.get("turnover_target_change", 0.0))
    metrics = compute_metrics(rewards, portfolio_returns, turnovers)
    trace = {
        "dates": dates,
        "rewards": rewards,
        "portfolio_returns": portfolio_returns,
        "turnovers": turnovers,
        "turnover_target_changes": turnover_target_changes,
    }
    return metrics, trace


def run_backtest_episode(model, env: DummyVecEnv) -> PortfolioMetrics:
    metrics, _ = run_backtest_episode_detailed(model, env)
    return metrics
