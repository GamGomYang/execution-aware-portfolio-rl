from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import random
import secrets
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
from .utils.signature import canonical_json, compute_env_signature, sha256_bytes

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
    random_reset: bool = False,
    risk_lambda: float = 0.0,
    risk_penalty_type: str = "r2",
    rebalance_eta: float | None = None,
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
        random_reset=bool(random_reset),
        risk_lambda=float(risk_lambda),
        risk_penalty_type=str(risk_penalty_type),
        rebalance_eta=float(rebalance_eta) if rebalance_eta is not None else None,
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
        center_prob=prl_cfg.get("center_prob", True),
        emergency_mode=prl_cfg.get("emergency_mode", "clamp"),
        emergency_vz_threshold=prl_cfg.get("emergency_vz_threshold", 2.0),
        vol_mean=mean,
        vol_std=std,
        window_size=window_size,
        num_assets=num_assets,
        mid_plasticity_multiplier=prl_cfg.get("mid_plasticity_multiplier", 1.0),
        var_penalty_beta=prl_cfg.get("var_penalty_beta"),
        cvar_penalty_gamma=prl_cfg.get("cvar_penalty_gamma"),
        penalty_clip_ratio=prl_cfg.get("penalty_clip_ratio", 0.2),
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
    return hashlib.sha256(blob).hexdigest()


def _manifest_hash(manifest: Dict) -> str:
    if not manifest:
        return ""
    if "data_manifest_hash" not in manifest:
        payload = {key: value for key, value in manifest.items() if key != "data_manifest_hash"}
        return sha256_bytes(canonical_json(payload))
    return str(manifest.get("data_manifest_hash") or "")


def _vol_stats_filename(config: Dict, lv: int, manifest_hash: str) -> str:
    dates = config["dates"]
    stats_key = manifest_hash or _config_hash(config)
    return f"vol_stats_{stats_key[:8]}_Lv{lv}_{dates['train_start']}_{dates['train_end']}.json"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class TrainLoggingCallback(BaseCallback):
    schema_version = "1.1"

    def __init__(
        self,
        log_path: Path,
        run_id: str,
        model_type: str,
        seed: int,
        log_interval: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_path = log_path
        self.run_id = run_id
        self.model_type = model_type
        self.seed = seed
        self.log_interval = max(1, log_interval)
        self.buffer: List[Dict] = []

    def _on_step(self) -> bool:
        logger_vals = self.model.logger.name_to_value
        entropy_loss = logger_vals.get("train/entropy_loss")
        if entropy_loss is None:
            entropy_loss = logger_vals.get("train/ent_coef_loss", logger_vals.get("train/ent_coef"))
        record = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model_type": self.model_type,
            "seed": self.seed,
            "timesteps": self.num_timesteps,
            "actor_loss": logger_vals.get("train/actor_loss"),
            "critic_loss": logger_vals.get("train/critic_loss"),
            "entropy_loss": entropy_loss,
            "ent_coef": logger_vals.get("train/ent_coef"),
            "ent_coef_loss": logger_vals.get("train/ent_coef_loss"),
            "alpha_obs_mean": logger_vals.get("train/alpha_obs_mean"),
            "alpha_next_mean": logger_vals.get("train/alpha_next_mean"),
            "prl_prob_mean": logger_vals.get("train/prl_prob_mean"),
            "vz_mean": logger_vals.get("train/vz_mean"),
            "alpha_raw_mean": logger_vals.get("train/alpha_raw_mean"),
            "alpha_clamped_mean": logger_vals.get("train/alpha_clamped_mean"),
            "emergency_rate": logger_vals.get("train/emergency_rate"),
            "beta_effective_mean": logger_vals.get("train/beta_effective_mean"),
            "prl_prob_min": logger_vals.get("train/prl_prob_min"),
            "prl_prob_max": logger_vals.get("train/prl_prob_max"),
            "prl_prob_p05": logger_vals.get("train/prl_prob_p05"),
            "prl_prob_p50": logger_vals.get("train/prl_prob_p50"),
            "prl_prob_p95": logger_vals.get("train/prl_prob_p95"),
            "prl_prob_std": logger_vals.get("train/prl_prob_std"),
            "vz_min": logger_vals.get("train/vz_min"),
            "vz_max": logger_vals.get("train/vz_max"),
            "vz_p05": logger_vals.get("train/vz_p05"),
            "vz_p50": logger_vals.get("train/vz_p50"),
            "vz_p95": logger_vals.get("train/vz_p95"),
            "vz_std": logger_vals.get("train/vz_std"),
            "alpha_raw_min": logger_vals.get("train/alpha_raw_min"),
            "alpha_raw_max": logger_vals.get("train/alpha_raw_max"),
            "alpha_raw_p05": logger_vals.get("train/alpha_raw_p05"),
            "alpha_raw_p50": logger_vals.get("train/alpha_raw_p50"),
            "alpha_raw_p95": logger_vals.get("train/alpha_raw_p95"),
            "alpha_raw_std": logger_vals.get("train/alpha_raw_std"),
            "alpha_obs_min": logger_vals.get("train/alpha_obs_min"),
            "alpha_obs_max": logger_vals.get("train/alpha_obs_max"),
            "alpha_obs_p05": logger_vals.get("train/alpha_obs_p05"),
            "alpha_obs_p50": logger_vals.get("train/alpha_obs_p50"),
            "alpha_obs_p95": logger_vals.get("train/alpha_obs_p95"),
            "alpha_obs_std": logger_vals.get("train/alpha_obs_std"),
            "actor_loss_base": logger_vals.get("train/actor_loss_base"),
            "cvar_penalty_raw_mean": logger_vals.get("train/cvar_penalty_raw_mean"),
            "cvar_penalty_weighted_mean": logger_vals.get("train/cvar_penalty_weighted_mean"),
            "var_penalty_raw_mean": logger_vals.get("train/var_penalty_raw_mean"),
            "var_penalty_weighted_mean": logger_vals.get("train/var_penalty_weighted_mean"),
            "penalty_ratio": logger_vals.get("train/penalty_ratio"),
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
    run_id: str,
    model_path: Path,
    log_path: Path,
    vol_stats_path: Path | None = None,
) -> Path:
    manifest_path = Path(config.get("data", {}).get("processed_dir", "data/processed")) / "data_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest_hash = _manifest_hash(manifest)
    now = datetime.now(timezone.utc)
    config_hash_val = _config_hash(config)
    config_path = config.get("config_path", "")
    created_at = now.isoformat()
    asset_list = manifest.get("asset_list") or manifest.get("kept_tickers") or []
    num_assets = int(manifest.get("num_assets", len(asset_list) if asset_list else 0))
    L = manifest.get("L", config.get("env", {}).get("L"))
    Lv = manifest.get("Lv", config.get("env", {}).get("Lv"))
    obs_dim_expected = manifest.get("obs_dim_expected")
    if obs_dim_expected is None and L is not None and num_assets:
        obs_dim_expected = int(num_assets) * (int(L) + 2)
    env_signature_hash = manifest.get("env_signature_hash")
    feature_flags = dict(
        manifest.get(
            "feature_flags",
            {"returns_window": True, "volatility": True, "prev_weights": True},
        )
    )
    reward_type = "log_net_minus_r2"
    feature_flags["reward_type"] = reward_type
    rebalance_eta = config.get("env", {}).get("rebalance_eta")
    feature_flags["action_smoothing"] = rebalance_eta is not None
    # Always compute signature with the configured transaction cost; manifest may carry a different default.
    cost_params_cfg = {
        "transaction_cost": config.get("env", {}).get("c_tc"),
        "risk_lambda": float(config.get("env", {}).get("risk_lambda", 0.0)),
        "rebalance_eta": float(rebalance_eta) if rebalance_eta is not None else None,
    }
    if asset_list and L is not None and Lv is not None:
        env_signature_hash = compute_env_signature(
            asset_list,
            int(L),
            int(Lv),
            feature_flags=feature_flags,
            cost_params=cost_params_cfg,
            schema_version=manifest.get("env_schema_version", "v1"),
        )
    outputs_root = base_dir.parent
    report_paths = {
        "trace_path": str(base_dir / f"trace_{run_id}.parquet"),
        "regime_thresholds_path": str(base_dir / f"regime_thresholds_{run_id}.json"),
        "step4_report_path": str(base_dir / f"step4_report_{run_id}.md"),
        "regime_metrics_path": str(base_dir / "regime_metrics.csv"),
        "figures_dir": str(outputs_root / "figures" / run_id),
    }
    if vol_stats_path is None:
        lv = config.get("env", {}).get("Lv")
        if lv is None:
            lv = manifest.get("Lv")
        if lv is not None:
            stats_filename = _vol_stats_filename(config, int(lv), manifest_hash)
            processed_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
            vol_stats_path = processed_dir / stats_filename

    vol_stats_hash = None
    if vol_stats_path is not None and vol_stats_path.exists():
        vol_stats_hash = _sha256_file(vol_stats_path)

    meta = {
        "run_id": run_id,
        "seed": seed,
        "mode": mode,
        "model_type": model_type,
        "config_path": config_path,
        "config_hash": config_hash_val,
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "yfinance_version": yf.__version__,
        "sb3_version": sb3_version,
        "packages": {
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
            "stable_baselines3": sb3_version,
        },
        "created_at": created_at,
        "data_manifest_path": str(manifest_path),
        "data_manifest_hash": manifest_hash,
        "vol_stats_path": str(vol_stats_path) if vol_stats_path is not None else None,
        "vol_stats_hash": vol_stats_hash,
        "asset_list": asset_list,
        "num_assets": num_assets,
        "L": L,
        "Lv": Lv,
        "obs_dim_expected": obs_dim_expected,
        "env_signature_hash": env_signature_hash,
        "env_signature_version": "v3",
        "env_params": {
            "transaction_cost": cost_params_cfg.get("transaction_cost"),
            "risk_lambda": cost_params_cfg.get("risk_lambda", 0.0),
            "risk_penalty_type": config.get("env", {}).get("risk_penalty_type", "r2"),
            "rebalance_eta": cost_params_cfg.get("rebalance_eta"),
            "reward_type": reward_type,
        },
        "artifacts": {
            "model_path": str(model_path),
            "train_log_path": str(log_path),
        },
        "artifact_paths": {
            "model_path": str(model_path),
            "train_log_path": str(log_path),
        },
        "report_paths": report_paths,
    }
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"run_metadata_{run_id}.json"
    out_path.write_text(json.dumps(meta, indent=2))
    return out_path


def _generate_run_id(config: Dict, seed: int, model_type: str) -> str:
    now = datetime.now(timezone.utc)
    config_hash_val = _config_hash(config)
    run_id_ts = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(2)
    return f"{run_id_ts}_{config_hash_val[:8]}_seed{seed}_{model_type}_{suffix}"


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
    manifest_hash = _manifest_hash(manifest)
    stats_filename = _vol_stats_filename(config, lv, manifest_hash)
    vol_features = compute_volatility_features(
        returns=market.returns,
        lv=lv,
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        processed_dir=processed_dir,
        stats_filename=stats_filename,
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
    random_reset: bool = False,
    risk_lambda: float = 0.0,
    risk_penalty_type: str = "r2",
    rebalance_eta: float | None = None,
) -> DummyVecEnv:
    returns_slice = slice_frame(market.returns, start, end)
    vol_slice = slice_frame(features.volatility, start, end)
    if logit_scale is None:
        logit_scale = EnvConfig.logit_scale
    return build_vec_env(
        returns_slice,
        vol_slice,
        window_size,
        c_tc,
        seed,
        logit_scale,
        random_reset=random_reset,
        risk_lambda=risk_lambda,
        risk_penalty_type=risk_penalty_type,
        rebalance_eta=rebalance_eta,
    )


def run_training(
    config: Dict,
    model_type: str,
    seed: int,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "outputs/models",
    reports_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
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
    run_id = _generate_run_id(config, seed, model_type)

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
    random_reset_train = bool(env_cfg.get("random_reset_train", False))
    risk_lambda = float(env_cfg.get("risk_lambda", 0.0))
    risk_penalty_type = str(env_cfg.get("risk_penalty_type", "r2"))
    rebalance_eta = env_cfg.get("rebalance_eta")
    env = build_env_for_range(
        market=market,
        features=features,
        start=dates["train_start"],
        end=dates["train_end"],
        window_size=env_cfg["L"],
        c_tc=env_cfg["c_tc"],
        seed=seed,
        logit_scale=logit_scale,
        random_reset=random_reset_train,
        risk_lambda=risk_lambda,
        risk_penalty_type=risk_penalty_type,
        rebalance_eta=rebalance_eta,
    )
    num_assets = market.returns.shape[1]
    log_dir = Path(logs_dir) if logs_dir is not None else Path("outputs/logs")
    log_path = log_dir / f"train_{run_id}.csv"
    callbacks = [TrainLoggingCallback(log_path, run_id, model_type, seed, log_interval=log_interval)]
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
    model_path = output_path / f"{run_id}_final.zip"
    model.save(model_path)
    reports_path = Path(reports_dir) if reports_dir is not None else Path("outputs/reports")
    _write_run_metadata(
        reports_path,
        config,
        seed,
        mode,
        model_type,
        run_id,
        model_path,
        log_path,
        vol_stats_path=features.stats_path,
    )
    return model_path
