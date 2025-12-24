from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from .data import MarketData, load_market_data, slice_frame
from .envs import Dow30PortfolioEnv, EnvConfig
from .features import VolatilityFeatures, compute_volatility_features, load_vol_stats
from .prl import PRLAlphaScheduler, PRLConfig
from .sb3_prl_sac import PRLSAC


def _align_frames(returns: pd.DataFrame, volatility: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vol_clean = volatility.dropna()
    idx = returns.index.intersection(vol_clean.index)
    returns_aligned = returns.loc[idx]
    vol_aligned = vol_clean.loc[idx]
    return returns_aligned, vol_aligned


def build_vec_env(returns: pd.DataFrame, volatility: pd.DataFrame, window_size: int, c_tc: float, seed: int) -> DummyVecEnv:
    returns_aligned, vol_aligned = _align_frames(returns, volatility)
    if len(returns_aligned) <= window_size + 1:
        raise ValueError("Not enough data after alignment to build environment.")

    cfg = EnvConfig(
        returns=returns_aligned,
        volatility=vol_aligned,
        window_size=window_size,
        transaction_cost=c_tc,
    )

    def _init():
        env = Dow30PortfolioEnv(cfg)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])


def create_scheduler(prl_cfg: Dict[str, float], window_size: int, num_assets: int, stats_path: Path) -> PRLAlphaScheduler:
    mean, std = load_vol_stats(stats_path)
    cfg = PRLConfig(
        alpha0=prl_cfg["alpha0"],
        beta=prl_cfg["beta"],
        lambdav=prl_cfg["lambdav"],
        bias=prl_cfg["bias"],
        alpha_min=prl_cfg["alpha_min"],
        alpha_max=prl_cfg["alpha_max"],
        vol_mean=mean,
        vol_std=std,
        window_size=window_size,
        num_assets=num_assets,
    )
    return PRLAlphaScheduler(cfg)


def _shared_model_kwargs(sac_cfg: Dict[str, float], seed: int) -> Dict:
    return {
        "learning_rate": sac_cfg["learning_rate"],
        "buffer_size": sac_cfg["buffer_size"],
        "batch_size": sac_cfg["batch_size"],
        "gamma": sac_cfg["gamma"],
        "tau": sac_cfg["tau"],
        "seed": seed,
        "verbose": 1,
    }


def train_baseline_model(env: DummyVecEnv, sac_cfg: Dict[str, float], seed: int) -> SAC:
    kwargs = _shared_model_kwargs(sac_cfg, seed)
    ent_coef = sac_cfg.get("ent_coef", 0.2)
    model = SAC("MlpPolicy", env, ent_coef=ent_coef, **kwargs)
    model.learn(total_timesteps=sac_cfg["total_timesteps"])
    return model


def train_prl_model(
    env: DummyVecEnv,
    sac_cfg: Dict[str, float],
    prl_cfg: Dict[str, float],
    scheduler: PRLAlphaScheduler,
    seed: int,
) -> PRLSAC:
    kwargs = _shared_model_kwargs(sac_cfg, seed)
    model = PRLSAC("MlpPolicy", env, scheduler=scheduler, **kwargs)
    model.learn(total_timesteps=sac_cfg["total_timesteps"])
    return model


def prepare_market_and_features(
    start_date: str,
    end_date: str,
    train_start: str,
    train_end: str,
    lv: int,
    raw_dir: str | Path,
    processed_dir: str | Path,
    force_refresh: bool = False,
) -> tuple[MarketData, VolatilityFeatures]:
    market = load_market_data(
        start_date=start_date,
        end_date=end_date,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        force_refresh=force_refresh,
    )
    vol_features = compute_volatility_features(
        returns=market.returns,
        lv=lv,
        train_start=train_start,
        train_end=train_end,
        processed_dir=processed_dir,
    )
    return market, vol_features


def build_env_for_range(
    market: MarketData,
    features: VolatilityFeatures,
    start: str,
    end: str,
    window_size: int,
    c_tc: float,
    seed: int,
) -> DummyVecEnv:
    returns_slice = slice_frame(market.returns, start, end)
    vol_slice = slice_frame(features.volatility, start, end)
    return build_vec_env(returns_slice, vol_slice, window_size, c_tc, seed)


def run_training(
    config: Dict,
    model_type: str,
    seed: int,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "outputs/models",
    force_refresh: bool = False,
) -> Path:
    dates = config["dates"]
    env_cfg = config["env"]
    sac_cfg = config["sac"]
    prl_cfg = config.get("prl", {})

    market, features = prepare_market_and_features(
        start_date=dates["train_start"],
        end_date=dates["test_end"],
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        lv=env_cfg["Lv"],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        force_refresh=force_refresh,
    )

    env = build_env_for_range(
        market=market,
        features=features,
        start=dates["train_start"],
        end=dates["train_end"],
        window_size=env_cfg["L"],
        c_tc=env_cfg["c_tc"],
        seed=seed,
    )
    num_assets = market.returns.shape[1]

    if model_type == "prl":
        scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)
        model = train_prl_model(env, sac_cfg, prl_cfg, scheduler, seed)
    else:
        model = train_baseline_model(env, sac_cfg, seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{model_type}_seed{seed}.zip"
    model.save(model_path)
    return model_path
