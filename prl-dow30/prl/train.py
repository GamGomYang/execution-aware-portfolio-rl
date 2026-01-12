from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import hashlib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from stable_baselines3 import SAC, __version__ as sb3_version
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .data import MarketData, load_market_data, slice_frame
from .envs import Dow30PortfolioEnv, EnvConfig
from .features import VolatilityFeatures, compute_volatility_features, load_vol_stats
from .prl import PRLAlphaScheduler, PRLConfig
from .sb3_prl_sac import PRLSAC

LOGGER = logging.getLogger(__name__)


def _align_frames(returns: pd.DataFrame, volatility: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vol_clean = volatility.dropna()
    idx = returns.index.intersection(vol_clean.index)
    returns_aligned = returns.loc[idx]
    vol_aligned = vol_clean.loc[idx]
    return returns_aligned, vol_aligned


def build_vec_env(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    window_size: int,
    c_tc: float,
    seed: int,
    logit_scale: float | None = None,
) -> DummyVecEnv:
    returns_aligned, vol_aligned = _align_frames(returns, volatility)
    if len(returns_aligned) <= window_size + 1:
        raise ValueError("Not enough data after alignment to build environment.")
    if logit_scale is None:
        logit_scale = EnvConfig.logit_scale

    cfg = EnvConfig(
        returns=returns_aligned,
        volatility=vol_aligned,
        window_size=window_size,
        transaction_cost=c_tc,
        logit_scale=float(logit_scale),
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
    return model


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _config_hash(config: Dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class TrainLoggingCallback(BaseCallback):
    def __init__(self, log_path: Path, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_interval = max(1, log_interval)
        self.buffer: List[Dict] = []

    def _on_step(self) -> bool:
        logger_vals = self.model.logger.name_to_value
        record = {
            "timesteps": self.num_timesteps,
            "actor_loss": logger_vals.get("train/actor_loss"),
            "critic_loss": logger_vals.get("train/critic_loss"),
            "entropy_loss": logger_vals.get("train/entropy_loss"),
        }
        self.buffer.append(record)
        if len(self.buffer) >= self.log_interval:
            self._flush()
        return True

    def _flush(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.log_path.exists()
        df.to_csv(self.log_path, mode="a", index=False, header=header)
        self.buffer.clear()

    def _on_training_end(self) -> None:
        self._flush()


def _write_run_metadata(
    base_dir: Path,
    config: Dict,
    seed: int,
    mode: str,
    model_type: str,
    model_path: Path,
    log_path: Path,
) -> Path:
    manifest_hash = ""
    manifest_path = Path(config.get("data", {}).get("processed_dir", "data/processed")) / "data_manifest.json"
    if manifest_path.exists():
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    config_hash_val = _config_hash(config)
    created_at = now.isoformat()
    run_id_ts = now.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_id_ts}_{config_hash_val[:8]}_seed{seed}_{model_type}"
    meta = {
        "run_id": run_id,
        "seed": seed,
        "mode": mode,
        "model_type": model_type,
        "config_hash": config_hash_val,
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "packages": {
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
            "stable_baselines3": sb3_version,
        },
        "created_at": created_at,
        "data_manifest_hash": manifest_hash,
        "artifacts": {
            "model_path": str(model_path),
            "train_log_path": str(log_path),
        },
    }
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"run_metadata_{run_id}.json"
    out_path.write_text(json.dumps(meta, indent=2))
    return out_path


def prepare_market_and_features(
    config: Dict,
    lv: int,
    force_refresh: bool = True,
    offline: bool = False,
    require_cache: bool = False,
    paper_mode: bool = False,
    session_opts: Dict | None = None,
    cache_only: bool = False,
) -> tuple[MarketData, VolatilityFeatures]:
    dates = config["dates"]
    data_cfg = {**config.get("data", {})}
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    data_cfg.update(
        {
            "raw_dir": data_cfg.get("raw_dir", "data/raw"),
            "processed_dir": processed_dir,
            "offline": offline,
            "require_cache": require_cache,
            "paper_mode": paper_mode,
            "session_opts": session_opts if session_opts is not None else data_cfg.get("session_opts"),
        }
    )
    load_cfg = {
        "dates": {"train_start": dates["train_start"], "test_end": dates["test_end"]},
        "data": data_cfg,
    }
    prices, returns, manifest, quality_summary = load_market_data(
        load_cfg,
        offline=offline,
        require_cache=require_cache,
        cache_only=cache_only,
        force_refresh=force_refresh,
    )
    market = MarketData(prices=prices, returns=returns, manifest=manifest, quality_summary=quality_summary)
    vol_features = compute_volatility_features(
        returns=market.returns,
        lv=lv,
        train_start=dates["train_start"],
        train_end=dates["train_end"],
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
    logit_scale: float | None = None,
) -> DummyVecEnv:
    returns_slice = slice_frame(market.returns, start, end)
    vol_slice = slice_frame(features.volatility, start, end)
    if logit_scale is None:
        logit_scale = EnvConfig.logit_scale
    return build_vec_env(returns_slice, vol_slice, window_size, c_tc, seed, logit_scale)


def run_training(
    config: Dict,
    model_type: str,
    seed: int,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "outputs/models",
    force_refresh: bool = True,
    offline: bool = False,
    cache_only: bool = False,
) -> Path:
    dates = config["dates"]
    env_cfg = config["env"]
    sac_cfg = config["sac"]
    prl_cfg = config.get("prl", {})
    mode = config.get("mode", "default")
    data_cfg = {**config.get("data", {})}
    data_cfg.setdefault("raw_dir", raw_dir)
    data_cfg.setdefault("processed_dir", processed_dir)
    config = {**config, "data": data_cfg}
    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    if paper_mode and not require_cache_cfg:
        raise ValueError("paper_mode=true requires require_cache=true.")
    offline = offline or data_cfg.get("offline", False)
    require_cache = require_cache_cfg or paper_mode or offline
    cache_only = (
        cache_only
        or data_cfg.get("paper_mode", False)
        or data_cfg.get("require_cache", False)
        or data_cfg.get("offline", False)
        or offline
    )
    session_opts = data_cfg.get("session_opts", None)
    checkpoint_interval = sac_cfg.get("checkpoint_interval")
    log_interval = sac_cfg.get("log_interval_steps", 1000 if mode == "paper" else 50)

    if mode == "paper":
        if sac_cfg["total_timesteps"] < 100000:
            raise ValueError("Paper mode requires total_timesteps >= 100000.")
        if sac_cfg["buffer_size"] < 100000:
            raise ValueError("Paper mode requires buffer_size >= 100000.")
    if sac_cfg["buffer_size"] < sac_cfg["batch_size"] * 10:
        LOGGER.warning("buffer_size is less than 10x batch_size; training stability may suffer.")
    if sac_cfg.get("ent_coef") == "auto":
        raise ValueError("ent_coef auto tuning is disabled; provide a fixed float.")

    _set_global_seeds(seed)

    market, features = prepare_market_and_features(
        config=config,
        lv=env_cfg["Lv"],
        force_refresh=force_refresh,
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=session_opts,
        cache_only=cache_only,
    )

    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required for training.")
    logit_scale = float(env_cfg["logit_scale"])
    env = build_env_for_range(
        market=market,
        features=features,
        start=dates["train_start"],
        end=dates["train_end"],
        window_size=env_cfg["L"],
        c_tc=env_cfg["c_tc"],
        seed=seed,
        logit_scale=logit_scale,
    )
    num_assets = market.returns.shape[1]
    log_dir = Path("outputs/logs")
    log_path = log_dir / f"{model_type}_seed{seed}_train_log.csv"
    callbacks = [TrainLoggingCallback(log_path, log_interval=log_interval)]
    if mode == "paper" and checkpoint_interval:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_interval,
                save_path=Path(output_dir),
                name_prefix=f"{model_type}_seed{seed}_step",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    if model_type == "prl":
        scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)
        model = train_prl_model(env, sac_cfg, prl_cfg, scheduler, seed)
    else:
        model = train_baseline_model(env, sac_cfg, seed)
    if callbacks:
        model.learn(total_timesteps=sac_cfg["total_timesteps"], callback=callbacks)
    else:
        model.learn(total_timesteps=sac_cfg["total_timesteps"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{model_type}_seed{seed}_final.zip"
    model.save(model_path)
    _write_run_metadata(Path("outputs/reports"), config, seed, mode, model_type, model_path, log_path)
    return model_path
