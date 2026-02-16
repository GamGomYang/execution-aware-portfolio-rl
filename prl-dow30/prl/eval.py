from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from .baselines import run_all_baselines_detailed
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
    env_params = run_metadata.get("env_params", {}) or {}
    sig_version = run_metadata.get("env_signature_version")
    if sig_version == "v3" or "rebalance_eta" in env_params:
        rebalance_eta = getattr(base_env.cfg, "rebalance_eta", None)
        cost_params = {
            "transaction_cost": getattr(base_env.cfg, "transaction_cost", None),
            "risk_lambda": float(getattr(base_env.cfg, "risk_lambda", 0.0)),
            "rebalance_eta": float(rebalance_eta) if rebalance_eta is not None else None,
        }
        reward_type = env_params.get("reward_type", "log_net_minus_r2")
        feature_flags = {
            "returns_window": True,
            "volatility": True,
            "prev_weights": True,
            "reward_type": reward_type,
            "action_smoothing": rebalance_eta is not None,
        }
    elif sig_version == "v2" or "risk_lambda" in env_params or "reward_type" in env_params:
        cost_params = {
            "transaction_cost": getattr(base_env.cfg, "transaction_cost", None),
            "risk_lambda": float(getattr(base_env.cfg, "risk_lambda", 0.0)),
        }
        reward_type = env_params.get("reward_type", "log_net_minus_r2")
        feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True, "reward_type": reward_type}
    else:
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
    turnovers_exec: List[float] = []
    turnovers_target: List[float] = []
    dates: List = []
    costs: List[float] = []
    net_returns_exp: List[float] = []
    net_returns_lin: List[float] = []
    log_returns_gross: List[float] = []
    log_returns_net: List[float] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_vec, done_vec, info_list = env.step(action)
        reward = float(reward_vec[0])
        done = bool(done_vec[0])
        rewards.append(reward)
        info = info_list[0]
        port_ret = info.get("portfolio_return", 0.0)
        turnover_exec = info.get("turnover_exec", info.get("turnover", 0.0))
        turnover_target = info.get("turnover_target", info.get("turnover_target_change", turnover_exec))
        cost = float(info.get("cost", 0.0))
        log_return_gross = info.get("log_return_gross")
        log_return_net = info.get("log_return_net")
        portfolio_returns.append(port_ret)
        turnovers_exec.append(turnover_exec)
        turnovers_target.append(turnover_target)
        dates.append(info.get("date"))
        costs.append(cost)
        if log_return_net is not None:
            net_returns_exp.append(math.exp(float(log_return_net)) - 1.0)
        else:
            net_returns_exp.append(math.exp(reward) - 1.0)
        net_returns_lin.append(port_ret - cost)
        log_returns_gross.append(float(log_return_gross) if log_return_gross is not None else float("nan"))
        log_returns_net.append(float(log_return_net) if log_return_net is not None else float("nan"))
    metrics = compute_metrics(
        rewards,
        portfolio_returns,
        turnovers_exec,
        turnovers_target=turnovers_target,
        net_returns_exp=net_returns_exp,
        net_returns_lin=net_returns_lin,
    )
    trace = {
        "dates": dates,
        "rewards": rewards,
        "portfolio_returns": portfolio_returns,
        "turnovers": turnovers_exec,
        "turnovers_exec": turnovers_exec,
        "turnovers_target": turnovers_target,
        "turnover_target_changes": turnovers_target,
        "costs": costs,
        "net_returns_exp": net_returns_exp,
        "net_returns_lin": net_returns_lin,
        "log_returns_gross": log_returns_gross,
        "log_returns_net": log_returns_net,
    }
    return metrics, trace


def run_backtest_episode(model, env: DummyVecEnv) -> PortfolioMetrics:
    metrics, _ = run_backtest_episode_detailed(model, env)
    return metrics


def trace_dict_to_frame(
    trace: Dict[str, List[float]],
    *,
    eval_id: str,
    run_id: str,
    model_type: str,
    seed: int,
) -> pd.DataFrame:
    turnover_exec = trace.get("turnovers_exec")
    turnover_target = trace.get("turnovers_target")
    turnover_target_changes = trace.get("turnover_target_changes")
    turnovers = trace.get("turnovers")
    dates = trace.get("dates", [])
    costs = trace.get("costs")
    net_returns_exp = trace.get("net_returns_exp")
    net_returns_lin = trace.get("net_returns_lin")
    log_returns_gross = trace.get("log_returns_gross")
    log_returns_net = trace.get("log_returns_net")
    if turnovers is None or not turnovers:
        turnovers = [np.nan] * len(dates)
    if turnover_exec is None or not turnover_exec:
        turnover_exec = turnovers
    if turnover_target is None or not turnover_target:
        turnover_target = turnover_target_changes if turnover_target_changes else [np.nan] * len(dates)
    if turnover_target_changes is None or not turnover_target_changes:
        turnover_target_changes = turnover_target
    if costs is None or not costs:
        costs = [np.nan] * len(dates)
    if net_returns_exp is None or not net_returns_exp:
        net_returns_exp = [np.nan] * len(dates)
    if net_returns_lin is None or not net_returns_lin:
        net_returns_lin = [np.nan] * len(dates)
    if log_returns_gross is None or not log_returns_gross:
        log_returns_gross = [np.nan] * len(dates)
    if log_returns_net is None or not log_returns_net:
        log_returns_net = [np.nan] * len(dates)
    df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": trace.get("portfolio_returns", []),
            "reward": trace.get("rewards", []),
            "turnover": turnovers,
            "turnover_exec": turnover_exec,
            "turnover_target": turnover_target,
            "turnover_target_change": turnover_target_changes,
            "cost": costs,
            "net_return_exp": net_returns_exp,
            "net_return_lin": net_returns_lin,
            "log_return_gross": log_returns_gross,
            "log_return_net": log_returns_net,
        }
    )
    df["eval_id"] = eval_id
    df["run_id"] = run_id
    df["model_type"] = model_type
    df["seed"] = seed
    df["date"] = pd.to_datetime(df["date"])
    df["equity_gross"] = np.cumprod(1.0 + df["portfolio_return"].fillna(0.0))
    if "net_return_exp" in df.columns:
        df["equity_net_exp"] = np.cumprod(1.0 + df["net_return_exp"].fillna(0.0))
    if "net_return_lin" in df.columns:
        df["equity_net_lin"] = np.cumprod(1.0 + df["net_return_lin"].fillna(0.0))
    return df


def eval_model_to_trace(
    model,
    env: DummyVecEnv,
    *,
    eval_id: str,
    run_id: str,
    model_type: str,
    seed: int,
) -> Tuple[PortfolioMetrics, pd.DataFrame]:
    metrics, trace = run_backtest_episode_detailed(model, env)
    trace_df = trace_dict_to_frame(
        trace,
        eval_id=eval_id,
        run_id=run_id,
        model_type=model_type,
        seed=seed,
    )
    return metrics, trace_df


def eval_strategies_to_trace(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float,
    eval_id: str,
    run_id: str,
    seed: int,
) -> Tuple[Dict[str, PortfolioMetrics], pd.DataFrame]:
    results = run_all_baselines_detailed(
        returns,
        volatility,
        transaction_cost=transaction_cost,
    )
    frames = []
    metrics_by_name: Dict[str, PortfolioMetrics] = {}
    for name, (metrics, trace) in results.items():
        metrics_by_name[name] = metrics
        frames.append(
            trace_dict_to_frame(
                trace,
                eval_id=eval_id,
                run_id=run_id,
                model_type=name,
                seed=seed,
            )
        )
    trace_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return metrics_by_name, trace_df


def compute_regime_labels(trace_df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    if trace_df.empty:
        return trace_df
    if "vz" not in trace_df.columns:
        raise ValueError("REGIME_LABELS_MISSING_VZ")
    q33 = float(thresholds["q33"])
    q66 = float(thresholds["q66"])
    df = trace_df.copy()
    regimes = pd.Series(index=df.index, dtype="object")
    vz = df["vz"].astype(float)
    regimes[vz < q33] = "low"
    regimes[(vz >= q33) & (vz < q66)] = "mid"
    regimes[vz >= q66] = "high"
    df["regime"] = regimes
    df["regime_label"] = df["regime"]
    df = df.dropna(subset=["regime"])
    return df


def summarize_regime_metrics(
    trace_df: pd.DataFrame,
    *,
    period: str = "test",
    include_all: bool = True,
) -> List[Dict]:
    rows: List[Dict] = []
    if trace_df.empty:
        return rows
    has_eval_id = "eval_id" in trace_df.columns
    has_eval_window = "eval_window" in trace_df.columns
    group_cols = ["run_id", "model_type", "seed"]
    if has_eval_id:
        group_cols = ["eval_id"] + group_cols
    if has_eval_window:
        group_cols = ["eval_window"] + group_cols
    for group_keys, group in trace_df.groupby(group_cols):
        if has_eval_id:
            if has_eval_window:
                eval_window, eval_id, run_id, model_type, seed = group_keys
            else:
                eval_window = None
                eval_id, run_id, model_type, seed = group_keys
        else:
            eval_id = None
            if has_eval_window:
                eval_window, run_id, model_type, seed = group_keys
            else:
                eval_window = None
                run_id, model_type, seed = group_keys
        seed_val = int(seed)
        for regime, regime_group in group.groupby("regime"):
            metrics = compute_metrics(
                regime_group["reward"].tolist(),
                regime_group["portfolio_return"].tolist(),
                regime_group["turnover"].tolist(),
                net_returns_exp=regime_group.get("net_return_exp"),
                net_returns_lin=regime_group.get("net_return_lin"),
            )
            row = {
                "run_id": run_id,
                "model_type": model_type,
                "seed": seed_val,
                "regime": regime,
                "period": period,
                **metrics.to_dict(),
            }
            if has_eval_id:
                row["eval_id"] = eval_id
            if has_eval_window:
                row["eval_window"] = eval_window
            rows.append(row)
        if include_all:
            metrics = compute_metrics(
                group["reward"].tolist(),
                group["portfolio_return"].tolist(),
                group["turnover"].tolist(),
                net_returns_exp=group.get("net_return_exp"),
                net_returns_lin=group.get("net_return_lin"),
            )
            row = {
                "run_id": run_id,
                "model_type": model_type,
                "seed": seed_val,
                "regime": "all",
                "period": period,
                **metrics.to_dict(),
            }
            if has_eval_id:
                row["eval_id"] = eval_id
            if has_eval_window:
                row["eval_window"] = eval_window
            rows.append(row)
    return rows
