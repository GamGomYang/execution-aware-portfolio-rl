#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import secrets
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.data import slice_frame
from prl.eval import run_backtest_episode_detailed
from prl.metrics import compute_metrics, turnover_l1
from prl.sb3_prl_sac import PRLSAC
from prl.train import (
    build_env_for_range,
    build_signal_features,
    create_scheduler,
    prepare_market_and_features,
    resolve_signal_configuration,
    run_training,
)


LOGGER = logging.getLogger(__name__)
ALLOWED_ALGOS = ("prl", "sac", "ppo", "td3")
ALLOWED_RULE_BASELINES = ("1overN", "minvar")
PAPER_MIN_TIMESTEPS = 100000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model replacement matrix runner (PRL/SAC/PPO/TD3 + 1/N/MinVar).")
    parser.add_argument("--config", type=str, default="configs/prl_100k_signals_u27.yaml", help="Training config YAML.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for this matrix run.")
    parser.add_argument("--algos", nargs="+", default=list(ALLOWED_ALGOS), help="RL algos to train/evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seed list.")
    parser.add_argument("--etas", nargs="+", type=float, default=[0.079, 0.080, 0.082], help="Validation eta list.")
    parser.add_argument("--kappas", nargs="+", type=float, default=[0.0005, 0.0010], help="Validation kappa list.")
    parser.add_argument("--rl-timesteps", type=int, default=30000, help="Override training timesteps for RL models.")
    parser.add_argument("--eval-start", type=str, default="2022-01-01", help="Validation start.")
    parser.add_argument("--eval-end", type=str, default="2023-12-31", help="Validation end.")
    parser.add_argument("--minvar-lookback", type=int, default=252, help="Lookback for MinVar baseline.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip model training if model file exists.")
    parser.add_argument("--offline", action="store_true", help="Force offline cache mode.")
    parser.add_argument("--append-result-analysis", action="store_true", help="Append summary to result analysis file.")
    parser.add_argument(
        "--result-analysis-file",
        type=str,
        default="/workspace/execution-aware-portfolio-rl/결과 분석",
        help="Result analysis file path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan only.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    invalid_algos = sorted(set(args.algos) - set(ALLOWED_ALGOS))
    if invalid_algos:
        raise ValueError(f"Unsupported algos: {invalid_algos}. allowed={list(ALLOWED_ALGOS)}")
    if not args.seeds:
        raise ValueError("seeds must not be empty")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"duplicate seeds are not allowed: {args.seeds}")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError(f"seeds must be >= 0: {args.seeds}")
    if not args.etas:
        raise ValueError("etas must not be empty")
    if any(eta <= 0.0 or eta > 1.0 for eta in args.etas):
        raise ValueError(f"etas must satisfy 0 < eta <= 1: {args.etas}")
    if not args.kappas:
        raise ValueError("kappas must not be empty")
    if any(kappa < 0.0 for kappa in args.kappas):
        raise ValueError(f"kappas must be >= 0: {args.kappas}")
    if args.rl_timesteps <= 0:
        raise ValueError(f"rl-timesteps must be > 0, got {args.rl_timesteps}")
    if args.minvar_lookback < 30:
        raise ValueError(f"minvar-lookback must be >= 30, got {args.minvar_lookback}")


def _apply_mode_runtime_guards(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    mode = str(cfg.get("mode", "")).strip().lower()
    if mode == "paper" and int(args.rl_timesteps) < PAPER_MIN_TIMESTEPS:
        LOGGER.warning(
            "[ADJUST] mode=paper requires rl_timesteps >= %s; override %s -> %s",
            PAPER_MIN_TIMESTEPS,
            int(args.rl_timesteps),
            PAPER_MIN_TIMESTEPS,
        )
        args.rl_timesteps = PAPER_MIN_TIMESTEPS


def _load_cfg(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    cfg["config_path"] = str(config_path)
    resolve_signal_configuration(cfg)
    return cfg


def _build_training_cfg(base_cfg: dict[str, Any], rl_timesteps: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("sac", {})
    cfg["sac"]["total_timesteps"] = int(rl_timesteps)
    cfg["sac"]["eval_freq"] = int(max(1000, min(5000, rl_timesteps // 5)))
    cfg["sac"]["checkpoint_interval"] = int(max(0, rl_timesteps // 2))
    cfg["sac"]["log_interval_steps"] = int(max(200, min(1000, rl_timesteps // 20)))
    return cfg


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_run_id(prefix: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}_seed{seed}_{secrets.token_hex(2)}"


def _load_algo_model(
    *,
    algo: str,
    model_path: Path,
    env: Any,
    cfg: dict[str, Any],
    features: Any,
    num_assets: int,
) -> Any:
    if algo == "prl":
        prl_cfg = cfg.get("prl", {}) or {}
        scheduler = create_scheduler(prl_cfg, cfg["env"]["L"], num_assets, features.stats_path)
        model = PRLSAC.load(model_path, env=env)
        model.scheduler = scheduler
        return model
    if algo == "sac":
        return SAC.load(model_path, env=env)
    if algo == "ppo":
        return PPO.load(model_path, env=env)
    if algo == "td3":
        return TD3.load(model_path, env=env)
    raise ValueError(f"Unsupported algo: {algo}")


def _train_ppo_or_td3(
    *,
    algo: str,
    seed: int,
    cfg: dict[str, Any],
    market: Any,
    features: Any,
    signal_features: pd.DataFrame | None,
    output_root: Path,
    rl_timesteps: int,
) -> tuple[Path, dict[str, Any]]:
    env_cfg = cfg["env"]
    dates = cfg["dates"]
    env = build_env_for_range(
        market=market,
        features=features,
        start=dates["train_start"],
        end=dates["train_end"],
        window_size=env_cfg["L"],
        c_tc=float(env_cfg["c_tc"]),
        seed=seed,
        logit_scale=float(env_cfg["logit_scale"]),
        random_reset=bool(env_cfg.get("random_reset_train", False)),
        risk_lambda=float(env_cfg.get("risk_lambda", 0.0)),
        risk_penalty_type=str(env_cfg.get("risk_penalty_type", "r2")),
        rebalance_eta=env_cfg.get("rebalance_eta"),
        eta_mode=str(env_cfg.get("eta_mode", "legacy")),
        rule_vol_window=int((env_cfg.get("rule_vol", {}) or {}).get("window", 20)),
        rule_vol_a=float((env_cfg.get("rule_vol", {}) or {}).get("a", 1.0)),
        eta_clip_min=float((env_cfg.get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5])[0]),
        eta_clip_max=float((env_cfg.get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5])[1]),
        signal_features=signal_features,
    )

    run_id = _new_run_id(algo, seed)
    model_base = output_root / "models" / f"{run_id}_final"
    model_path = output_root / "models" / f"{run_id}_final.zip"
    output_root.joinpath("models").mkdir(parents=True, exist_ok=True)
    output_root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    output_root.joinpath("logs").mkdir(parents=True, exist_ok=True)

    started = time.time()
    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
        )
    else:
        action_dim = int(env.action_space.shape[-1])
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim, dtype=np.float32),
            sigma=0.1 * np.ones(action_dim, dtype=np.float32),
        )
        model = TD3(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=100000,
            gamma=0.99,
            action_noise=action_noise,
            train_freq=(1, "step"),
            gradient_steps=1,
        )
    model.learn(total_timesteps=int(rl_timesteps), progress_bar=False)
    model.save(str(model_base))
    elapsed = time.time() - started
    env.close()

    metadata = {
        "run_id": run_id,
        "algo": algo,
        "seed": int(seed),
        "rl_timesteps": int(rl_timesteps),
        "model_path": str(model_path),
        "duration_seconds": float(elapsed),
        "created_at": _now_utc(),
        "config_path": str(cfg.get("config_path")),
    }
    meta_path = output_root / "reports" / f"run_metadata_{run_id}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata


def _train_rl_models(
    *,
    cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    market: Any,
    features: Any,
    signal_features: pd.DataFrame | None,
    args: argparse.Namespace,
    output_root: Path,
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for algo in args.algos:
        for seed in args.seeds:
            LOGGER.info("[TRAIN] algo=%s seed=%s", algo, seed)
            if algo in {"prl", "sac"}:
                model_type = "prl" if algo == "prl" else "baseline"
                started = time.time()
                model_path = run_training(
                    config=train_cfg,
                    model_type=model_type,
                    seed=int(seed),
                    raw_dir=train_cfg.get("data", {}).get("raw_dir", "data/raw"),
                    processed_dir=train_cfg.get("data", {}).get("processed_dir", "data/processed"),
                    output_dir=output_root / "models",
                    reports_dir=output_root / "reports",
                    logs_dir=output_root / "logs",
                    force_refresh=bool(train_cfg.get("data", {}).get("force_refresh", True)),
                    offline=bool(args.offline or train_cfg.get("data", {}).get("offline", False)),
                    cache_only=True,
                )
                elapsed = time.time() - started
                run_id = model_path.stem
                if run_id.endswith("_final"):
                    run_id = run_id[: -len("_final")]
                registry.append(
                    {
                        "algo": algo,
                        "seed": int(seed),
                        "model_path": str(model_path),
                        "run_id": run_id,
                        "duration_seconds": elapsed,
                        "timesteps": int(args.rl_timesteps),
                    }
                )
                continue

            custom_model_path = output_root / "models" / f"{algo}_seed{seed}_final.zip"
            if args.skip_existing and custom_model_path.exists():
                LOGGER.info("[TRAIN] skip existing custom model: %s", custom_model_path)
                registry.append(
                    {
                        "algo": algo,
                        "seed": int(seed),
                        "model_path": str(custom_model_path),
                        "run_id": custom_model_path.stem.removesuffix("_final"),
                        "duration_seconds": 0.0,
                        "timesteps": int(args.rl_timesteps),
                        "skipped": True,
                    }
                )
                continue

            model_path, metadata = _train_ppo_or_td3(
                algo=algo,
                seed=int(seed),
                cfg=train_cfg,
                market=market,
                features=features,
                signal_features=signal_features,
                output_root=output_root,
                rl_timesteps=int(args.rl_timesteps),
            )
            registry.append(
                {
                    "algo": algo,
                    "seed": int(seed),
                    "model_path": str(model_path),
                    "run_id": metadata["run_id"],
                    "duration_seconds": metadata["duration_seconds"],
                    "timesteps": int(args.rl_timesteps),
                }
            )
    return registry


def _eval_rl_model(
    *,
    algo: str,
    seed: int,
    model_path: Path,
    cfg: dict[str, Any],
    market: Any,
    features: Any,
    signal_features: pd.DataFrame | None,
    eta: float,
    kappa: float,
    eval_start: str,
    eval_end: str,
) -> dict[str, Any]:
    env_cfg = cfg["env"]
    eval_env = build_env_for_range(
        market=market,
        features=features,
        start=eval_start,
        end=eval_end,
        window_size=int(env_cfg["L"]),
        c_tc=float(kappa),
        seed=int(seed),
        logit_scale=float(env_cfg["logit_scale"]),
        random_reset=False,
        risk_lambda=float(env_cfg.get("risk_lambda", 0.0)),
        risk_penalty_type=str(env_cfg.get("risk_penalty_type", "r2")),
        rebalance_eta=float(eta),
        eta_mode="fixed",
        rule_vol_window=int((env_cfg.get("rule_vol", {}) or {}).get("window", 20)),
        rule_vol_a=float((env_cfg.get("rule_vol", {}) or {}).get("a", 1.0)),
        eta_clip_min=float((env_cfg.get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5])[0]),
        eta_clip_max=float((env_cfg.get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5])[1]),
        signal_features=signal_features,
    )
    model = _load_algo_model(
        algo=algo,
        model_path=model_path,
        env=eval_env,
        cfg=cfg,
        features=features,
        num_assets=int(market.returns.shape[1]),
    )
    metrics, _ = run_backtest_episode_detailed(model, eval_env)
    eval_env.close()
    return metrics.to_dict()


def _solve_minvar_weights(cov: np.ndarray) -> np.ndarray:
    n = int(cov.shape[0])
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(n, dtype=np.float64) * 1e-6
    x0 = np.full(n, 1.0 / n, dtype=np.float64)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    try:
        res = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, None)
            s = float(w.sum())
            if s > 0.0 and np.isfinite(s):
                return w / s
    except Exception:
        pass

    inv_diag = 1.0 / np.clip(np.diag(cov), 1e-8, None)
    inv_diag = np.clip(inv_diag, 0.0, None)
    denom = float(inv_diag.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(n, 1.0 / n, dtype=np.float64)
    return inv_diag / denom


def _evaluate_rule_baseline(
    *,
    strategy: str,
    market_returns: pd.DataFrame,
    eta: float,
    kappa: float,
    eval_start: str,
    eval_end: str,
    minvar_lookback: int,
) -> dict[str, Any]:
    returns_full = market_returns
    returns_eval = slice_frame(returns_full, eval_start, eval_end)
    if returns_eval.empty:
        raise ValueError(f"No returns in eval window: {eval_start}~{eval_end}")

    num_assets = int(returns_eval.shape[1])
    equal_w = np.full(num_assets, 1.0 / num_assets, dtype=np.float64)
    prev_w = equal_w.copy()

    rewards: list[float] = []
    portfolio_returns: list[float] = []
    turnovers_exec: list[float] = []
    turnovers_target: list[float] = []
    net_returns_lin: list[float] = []

    for dt, row in returns_eval.iterrows():
        if strategy == "1overN":
            w_target = equal_w
        elif strategy == "minvar":
            loc = returns_full.index.get_loc(dt)
            history = returns_full.iloc[max(0, int(loc) - int(minvar_lookback)) : int(loc)]
            if int(history.shape[0]) < 30:
                w_target = equal_w
            else:
                cov = np.cov(history.to_numpy(dtype=np.float64), rowvar=False)
                w_target = _solve_minvar_weights(cov)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        w_exec = (1.0 - float(eta)) * prev_w + float(eta) * w_target
        w_exec = np.clip(w_exec, 0.0, None)
        w_exec_sum = float(w_exec.sum())
        if w_exec_sum <= 0.0 or not np.isfinite(w_exec_sum):
            w_exec = equal_w.copy()
        else:
            w_exec = w_exec / w_exec_sum

        r_arith = np.expm1(row.to_numpy(dtype=np.float64))
        port_ret = float(np.dot(w_exec, r_arith))
        to_exec = turnover_l1(prev_w, w_exec)
        to_target = turnover_l1(prev_w, w_target)
        cost = float(kappa) * to_exec
        reward = float(math.log(max(1.0 + port_ret, 1e-8)) - cost)
        net_lin = float(port_ret - cost)

        rewards.append(reward)
        portfolio_returns.append(port_ret)
        turnovers_exec.append(to_exec)
        turnovers_target.append(to_target)
        net_returns_lin.append(net_lin)
        prev_w = w_exec

    metrics = compute_metrics(
        rewards,
        portfolio_returns,
        turnovers_exec,
        turnovers_target=turnovers_target,
        net_returns_lin=net_returns_lin,
    )
    return metrics.to_dict()


def _summarize_and_write(
    *,
    output_root: Path,
    run_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(run_rows)
    runs_csv = reports_dir / "model_swap_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    rl_df = runs_df[runs_df["family"] == "rl"].copy()
    rule_df = runs_df[runs_df["family"] == "rule"].copy()

    rl_agg = (
        rl_df.groupby(["algo", "eta", "kappa"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            median_sharpe_net_lin=("sharpe_net_lin", "median"),
            mean_sharpe_net_lin=("sharpe_net_lin", "mean"),
            n_positive_sharpe=("sharpe_net_lin", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            median_cumret_net_lin=("cumulative_return_net_lin", "median"),
            median_turnover_exec=("avg_turnover_exec", "median"),
            median_maxdd_net_lin=("max_drawdown_net_lin", "median"),
        )
        .sort_values(["eta", "kappa", "median_sharpe_net_lin"], ascending=[True, True, False])
    )
    rl_agg_csv = reports_dir / "model_swap_rl_aggregate.csv"
    rl_agg.to_csv(rl_agg_csv, index=False)

    rule_agg = (
        rule_df.groupby(["algo", "eta", "kappa"], as_index=False)
        .agg(
            sharpe_net_lin=("sharpe_net_lin", "mean"),
            cumulative_return_net_lin=("cumulative_return_net_lin", "mean"),
            avg_turnover_exec=("avg_turnover_exec", "mean"),
            max_drawdown_net_lin=("max_drawdown_net_lin", "mean"),
        )
        .sort_values(["eta", "kappa", "sharpe_net_lin"], ascending=[True, True, False])
    )
    rule_agg_csv = reports_dir / "model_swap_rule_aggregate.csv"
    rule_agg.to_csv(rule_agg_csv, index=False)

    sac_seed = rl_df[rl_df["algo"] == "sac"][["seed", "eta", "kappa", "sharpe_net_lin"]].rename(
        columns={"sharpe_net_lin": "sac_sharpe_net_lin"}
    )
    vs_sac = rl_df.merge(sac_seed, on=["seed", "eta", "kappa"], how="left")
    vs_sac["delta_sharpe_vs_sac"] = pd.to_numeric(vs_sac["sharpe_net_lin"], errors="coerce") - pd.to_numeric(
        vs_sac["sac_sharpe_net_lin"], errors="coerce"
    )
    vs_sac_summary = (
        vs_sac[vs_sac["algo"] != "sac"]
        .groupby(["algo", "eta", "kappa"], as_index=False)
        .agg(
            n=("seed", "count"),
            median_delta_sharpe_vs_sac=("delta_sharpe_vs_sac", "median"),
            n_positive_delta_vs_sac=("delta_sharpe_vs_sac", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
        )
        .sort_values(["eta", "kappa", "median_delta_sharpe_vs_sac"], ascending=[True, True, False])
    )
    vs_sac_csv = reports_dir / "model_swap_vs_sac.csv"
    vs_sac_summary.to_csv(vs_sac_csv, index=False)

    summary_md = reports_dir / "model_swap_summary.md"
    lines: list[str] = []
    lines.append("# Model Swap Matrix Summary")
    lines.append("")
    lines.append(f"- generated_at_utc: {_now_utc()}")
    lines.append(f"- runs_csv: {runs_csv}")
    lines.append(f"- rl_aggregate_csv: {rl_agg_csv}")
    lines.append(f"- rule_aggregate_csv: {rule_agg_csv}")
    lines.append(f"- vs_sac_csv: {vs_sac_csv}")
    lines.append("")
    lines.append("## RL Aggregate (Top by eta/kappa)")
    if rl_agg.empty:
        lines.append("- no RL rows")
    else:
        for (eta, kappa), grp in rl_agg.groupby(["eta", "kappa"]):
            top = grp.iloc[0]
            lines.append(
                f"- eta={eta}, kappa={kappa}: top_algo={top['algo']}, "
                f"median_sharpe_net_lin={float(top['median_sharpe_net_lin']):.6f}, "
                f"n_positive_sharpe={int(top['n_positive_sharpe'])}/{int(top['n_seeds'])}"
            )
    lines.append("")
    lines.append("## Rule Baseline Aggregate")
    if rule_agg.empty:
        lines.append("- no rule baseline rows")
    else:
        for _, row in rule_agg.iterrows():
            lines.append(
                f"- algo={row['algo']}, eta={row['eta']}, kappa={row['kappa']}, "
                f"sharpe_net_lin={float(row['sharpe_net_lin']):.6f}, "
                f"cumret_net_lin={float(row['cumulative_return_net_lin']):.6f}"
            )
    lines.append("")
    lines.append("## Delta vs SAC")
    if vs_sac_summary.empty:
        lines.append("- no delta rows")
    else:
        for _, row in vs_sac_summary.iterrows():
            lines.append(
                f"- algo={row['algo']}, eta={row['eta']}, kappa={row['kappa']}, "
                f"median_delta_sharpe_vs_sac={float(row['median_delta_sharpe_vs_sac']):.6f}, "
                f"n_positive_delta_vs_sac={int(row['n_positive_delta_vs_sac'])}/{int(row['n'])}"
            )
    lines.append("")
    summary_md.write_text("\n".join(lines) + "\n")

    return {
        "runs_csv": runs_csv,
        "rl_aggregate_csv": rl_agg_csv,
        "rule_aggregate_csv": rule_agg_csv,
        "vs_sac_csv": vs_sac_csv,
        "summary_md": summary_md,
    }


def _append_result_analysis(
    *,
    result_analysis_file: Path,
    output_root: Path,
    summary_paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    result_analysis_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"## Model-Swap Snapshot UTC {_now_utc()}")
    lines.append("")
    lines.append("### Matrix Spec")
    lines.append(f"- output_root: {output_root}")
    lines.append(f"- config: {args.config}")
    lines.append(f"- algos: {args.algos}")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- etas: {args.etas}")
    lines.append(f"- kappas: {args.kappas}")
    lines.append(f"- rl_timesteps: {args.rl_timesteps}")
    lines.append(f"- eval_window: {args.eval_start}~{args.eval_end}")
    lines.append("")
    lines.append("### Artifacts")
    for key, path in summary_paths.items():
        lines.append(f"- {key}: {path}")
    lines.append("")
    lines.append("---")
    lines.append("")
    with result_analysis_file.open("a") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    _validate_args(args)

    config_path = Path(args.config)
    cfg = _load_cfg(config_path)
    _apply_mode_runtime_guards(args, cfg)
    train_cfg = _build_training_cfg(cfg, args.rl_timesteps)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    spec_payload = {
        "generated_at_utc": _now_utc(),
        "config": str(config_path),
        "algos": list(args.algos),
        "seeds": list(args.seeds),
        "etas": list(args.etas),
        "kappas": list(args.kappas),
        "rl_timesteps": int(args.rl_timesteps),
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "minvar_lookback": int(args.minvar_lookback),
    }
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "matrix_spec.json").write_text(json.dumps(spec_payload, indent=2))

    if args.dry_run:
        LOGGER.info("[DRY-RUN] matrix spec: %s", json.dumps(spec_payload, indent=2))
        return

    market, features = prepare_market_and_features(
        config=train_cfg,
        lv=int(train_cfg["env"]["Lv"]),
        force_refresh=bool(train_cfg.get("data", {}).get("force_refresh", True)),
        offline=bool(args.offline or train_cfg.get("data", {}).get("offline", False)),
        require_cache=True,
        paper_mode=bool(train_cfg.get("data", {}).get("paper_mode", False)),
        cache_only=True,
        session_opts=train_cfg.get("data", {}).get("session_opts"),
    )
    signal_features, _ = build_signal_features(market, config=train_cfg)

    registry = _train_rl_models(
        cfg=cfg,
        train_cfg=train_cfg,
        market=market,
        features=features,
        signal_features=signal_features,
        args=args,
        output_root=output_root,
    )
    registry_json = output_root / "reports" / "model_registry.json"
    registry_json.write_text(json.dumps(registry, indent=2))

    run_rows: list[dict[str, Any]] = []
    for item in registry:
        algo = str(item["algo"])
        seed = int(item["seed"])
        model_path = Path(item["model_path"])
        for eta in args.etas:
            for kappa in args.kappas:
                metrics = _eval_rl_model(
                    algo=algo,
                    seed=seed,
                    model_path=model_path,
                    cfg=train_cfg,
                    market=market,
                    features=features,
                    signal_features=signal_features,
                    eta=float(eta),
                    kappa=float(kappa),
                    eval_start=args.eval_start,
                    eval_end=args.eval_end,
                )
                run_rows.append(
                    {
                        "family": "rl",
                        "algo": algo,
                        "seed": seed,
                        "eta": float(eta),
                        "kappa": float(kappa),
                        "model_path": str(model_path),
                        "run_id": item.get("run_id"),
                        "sharpe_net_lin": metrics.get("sharpe_net_lin"),
                        "cumulative_return_net_lin": metrics.get("cumulative_return_net_lin"),
                        "avg_turnover_exec": metrics.get("avg_turnover_exec"),
                        "max_drawdown_net_lin": metrics.get("max_drawdown_net_lin"),
                        "steps": metrics.get("steps"),
                    }
                )

    for strategy in ALLOWED_RULE_BASELINES:
        for eta in args.etas:
            for kappa in args.kappas:
                metrics = _evaluate_rule_baseline(
                    strategy=strategy,
                    market_returns=market.returns,
                    eta=float(eta),
                    kappa=float(kappa),
                    eval_start=args.eval_start,
                    eval_end=args.eval_end,
                    minvar_lookback=int(args.minvar_lookback),
                )
                run_rows.append(
                    {
                        "family": "rule",
                        "algo": strategy,
                        "seed": -1,
                        "eta": float(eta),
                        "kappa": float(kappa),
                        "model_path": "",
                        "run_id": f"{strategy}_deterministic",
                        "sharpe_net_lin": metrics.get("sharpe_net_lin"),
                        "cumulative_return_net_lin": metrics.get("cumulative_return_net_lin"),
                        "avg_turnover_exec": metrics.get("avg_turnover_exec"),
                        "max_drawdown_net_lin": metrics.get("max_drawdown_net_lin"),
                        "steps": metrics.get("steps"),
                    }
                )

    summary_paths = _summarize_and_write(output_root=output_root, run_rows=run_rows)
    LOGGER.info("[DONE] model-swap matrix finished: %s", json.dumps({k: str(v) for k, v in summary_paths.items()}, indent=2))

    if args.append_result_analysis:
        _append_result_analysis(
            result_analysis_file=Path(args.result_analysis_file),
            output_root=output_root,
            summary_paths=summary_paths,
            args=args,
        )


if __name__ == "__main__":
    main()
