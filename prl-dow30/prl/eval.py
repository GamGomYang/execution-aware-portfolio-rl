from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from .metrics import PortfolioMetrics, compute_metrics
from .prl import PRLAlphaScheduler
from .sb3_prl_sac import PRLSAC
from .utils.signature import compute_env_signature


def assert_env_compatible(env: DummyVecEnv, run_metadata: Dict, *, Lv: int | None) -> None:
    base_env = env.envs[0] if hasattr(env, "envs") else env
    returns = getattr(base_env, "returns", None)
    if returns is None:
        raise ValueError("ENV_COMPATIBILITY_MISSING_RETURNS")
    asset_list = list(returns.columns)
    num_assets = int(getattr(base_env, "num_assets", len(asset_list)))
    obs_dim = int(base_env.observation_space.shape[0])
    window_size = int(getattr(base_env, "window_size", 0))
    cost_params = {"transaction_cost": getattr(base_env.cfg, "transaction_cost", None)}
    feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True}

    expected_obs_dim = run_metadata.get("obs_dim_expected")
    expected_num_assets = run_metadata.get("num_assets")
    expected_assets = run_metadata.get("asset_list") or []
    expected_env_signature = run_metadata.get("env_signature_hash")

    current_signature = None
    if Lv is not None:
        current_signature = compute_env_signature(
            asset_list,
            window_size,
            int(Lv),
            feature_flags=feature_flags,
            cost_params=cost_params,
            schema_version="v1",
        )

    missing = [t for t in expected_assets if t not in asset_list]
    extra = [t for t in asset_list if t not in expected_assets] if expected_assets else []
    order_mismatch = bool(expected_assets) and not missing and not extra and expected_assets != asset_list

    errors = []
    if expected_obs_dim is not None and int(expected_obs_dim) != obs_dim:
        errors.append(f"obs_dim_expected={expected_obs_dim} got={obs_dim}")
    if expected_num_assets is not None and int(expected_num_assets) != num_assets:
        errors.append(f"num_assets_expected={expected_num_assets} got={num_assets}")
    if not expected_assets:
        errors.append("expected_asset_list_missing=true")
    if missing:
        errors.append(f"asset_list_missing={missing}")
    if extra:
        errors.append(f"asset_list_extra={extra}")
    if order_mismatch:
        errors.append("asset_list_order_mismatch=true")
    if expected_env_signature and current_signature and expected_env_signature != current_signature:
        errors.append(f"env_signature_expected={expected_env_signature} got={current_signature}")

    if errors:
        raise ValueError("ENV_COMPATIBILITY_MISMATCH: " + " | ".join(errors))


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
