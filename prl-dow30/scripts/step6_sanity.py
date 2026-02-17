from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.data import MarketData
from prl.envs import EnvConfig
from prl.eval import load_model, run_backtest_episode_detailed, trace_dict_to_frame
from prl.features import VolatilityFeatures
from prl.prl import PRLAlphaScheduler
from prl.train import build_env_for_range, create_scheduler, prepare_market_and_features
from prl.utils.signature import compute_env_signature


LOGGER = logging.getLogger(__name__)


@dataclass
class EvalContext:
    cfg: dict[str, Any]
    env_cfg: dict[str, Any]
    prl_cfg: dict[str, Any]
    run_meta: dict[str, Any] | None
    run_id: str
    model_path: Path
    model_type: str
    seed: int
    window_size: int
    lv: int
    market: MarketData
    features: VolatilityFeatures
    eval_start: str
    eval_end: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6 sanity checks for execution layer.")
    parser.add_argument("--config", type=str, default="configs/step6_main.yaml", help="YAML config path.")
    parser.add_argument("--model-type", choices=["baseline", "prl"], default="prl", help="Model type.")
    parser.add_argument("--seed", type=int, default=0, help="Seed identifier for model discovery.")
    parser.add_argument("--model-path", type=str, help="Optional explicit model path.")
    parser.add_argument("--model-root", type=str, default="outputs", help="Root containing reports/models.")
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    parser.add_argument("--max-steps", type=int, default=252, help="Max evaluation steps for sanity speed.")
    return parser.parse_args()


def _load_latest_run_metadata(model_type: str, seed: int, reports_dir: Path) -> dict[str, Any] | None:
    if not reports_dir.exists():
        return None
    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in reports_dir.glob("run_metadata_*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("model_type") != model_type:
            continue
        if int(data.get("seed", -1)) != int(seed):
            continue
        candidates.append((str(data.get("created_at", "")), data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_run_metadata_for_model(model_path: Path, reports_dir: Path) -> dict[str, Any] | None:
    run_id = model_path.stem
    if run_id.endswith("_final"):
        run_id = run_id[: -len("_final")]
    meta_path = reports_dir / f"run_metadata_{run_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _resolve_model_path(
    *,
    model_path_arg: str | None,
    model_type: str,
    seed: int,
    model_root: Path,
) -> tuple[Path, dict[str, Any] | None]:
    reports_dir = model_root / "reports"
    models_dir = model_root / "models"

    if model_path_arg:
        model_path = Path(model_path_arg)
        run_meta = _load_run_metadata_for_model(model_path, reports_dir)
        return model_path, run_meta

    latest_meta = _load_latest_run_metadata(model_type, seed, reports_dir)
    if latest_meta:
        artifact_paths = latest_meta.get("artifact_paths") or latest_meta.get("artifacts") or {}
        model_path_value = artifact_paths.get("model_path")
        if model_path_value:
            return Path(model_path_value), latest_meta

    return models_dir / f"{model_type}_seed{seed}_final.zip", None


def _parse_rule_vol_env(env_cfg: dict[str, Any]) -> tuple[int, float, float, float]:
    rule_vol = env_cfg.get("rule_vol", {}) or {}
    window = int(rule_vol.get("window", int(EnvConfig.rule_vol_window)))
    a = float(rule_vol.get("a", float(EnvConfig.rule_vol_a)))
    eta_clip = rule_vol.get("eta_clip")
    if eta_clip is None:
        eta_clip_min = float(EnvConfig.eta_clip_min)
        eta_clip_max = float(EnvConfig.eta_clip_max)
    else:
        if not isinstance(eta_clip, (list, tuple)) or len(eta_clip) != 2:
            raise ValueError("env.rule_vol.eta_clip must be [min, max].")
        eta_clip_min = float(eta_clip[0])
        eta_clip_max = float(eta_clip[1])
    return window, a, eta_clip_min, eta_clip_max


def _get_transaction_cost(env_cfg: dict[str, Any]) -> float:
    c_tc = env_cfg.get("c_tc", env_cfg.get("transaction_cost"))
    if c_tc is None:
        raise ValueError("env.c_tc (or env.transaction_cost) is required.")
    return float(c_tc)


def _clip_eval_window(returns: pd.DataFrame, start: str, end: str, max_steps: int) -> tuple[str, str]:
    if max_steps <= 0:
        return start, end
    sliced = returns.loc[start:end]
    if sliced.empty:
        raise ValueError(f"No returns rows in eval window: {start}..{end}")
    if len(sliced) <= max_steps:
        return start, end
    clipped_end = pd.Timestamp(sliced.index[max_steps - 1]).strftime("%Y-%m-%d")
    return start, clipped_end


def _subset_universe(
    market: MarketData,
    features: VolatilityFeatures,
    expected_assets: list[str],
) -> tuple[MarketData, VolatilityFeatures]:
    if not expected_assets:
        return market, features
    missing = [a for a in expected_assets if a not in market.returns.columns]
    if missing:
        raise ValueError(f"Model metadata assets missing in loaded market data: {missing}")

    market_sub = MarketData(
        prices=market.prices[expected_assets],
        returns=market.returns[expected_assets],
        manifest=market.manifest,
        quality_summary=market.quality_summary,
    )
    vol_sub = features.volatility[expected_assets]
    features_sub = VolatilityFeatures(
        volatility=vol_sub,
        portfolio_scalar=vol_sub.mean(axis=1),
        stats_path=features.stats_path,
        mean=features.mean,
        std=features.std,
    )
    return market_sub, features_sub


def _compute_action_smoothing_flag(*, eta_mode: str, rebalance_eta: float | None) -> bool:
    return (eta_mode in {"fixed", "rule_vol"}) or (eta_mode == "legacy" and rebalance_eta is not None)


def _is_step6_signature_extension_active(
    *,
    eta_mode: str,
    rule_vol_window: int,
    rule_vol_a: float,
    eta_clip_min: float,
    eta_clip_max: float,
) -> bool:
    return not (
        str(eta_mode) == str(EnvConfig.eta_mode)
        and int(rule_vol_window) == int(EnvConfig.rule_vol_window)
        and np.isclose(float(rule_vol_a), float(EnvConfig.rule_vol_a), atol=0.0, rtol=0.0)
        and np.isclose(float(eta_clip_min), float(EnvConfig.eta_clip_min), atol=0.0, rtol=0.0)
        and np.isclose(float(eta_clip_max), float(EnvConfig.eta_clip_max), atol=0.0, rtol=0.0)
    )


def compute_current_env_signature(env, run_meta: dict[str, Any] | None) -> dict[str, Any]:
    base_env = env.envs[0] if hasattr(env, "envs") else env
    env_params = (run_meta or {}).get("env_params", {}) or {}
    sig_version = (run_meta or {}).get("env_signature_version")

    rebalance_eta = getattr(base_env.cfg, "rebalance_eta", None)
    eta_mode = str(getattr(base_env.cfg, "eta_mode", EnvConfig.eta_mode))
    rule_vol_window = int(getattr(base_env.cfg, "rule_vol_window", EnvConfig.rule_vol_window))
    rule_vol_a = float(getattr(base_env.cfg, "rule_vol_a", EnvConfig.rule_vol_a))
    eta_clip_min = float(getattr(base_env.cfg, "eta_clip_min", EnvConfig.eta_clip_min))
    eta_clip_max = float(getattr(base_env.cfg, "eta_clip_max", EnvConfig.eta_clip_max))

    if sig_version == "v3" or "rebalance_eta" in env_params or "eta_mode" in env_params:
        cost_params = {
            "transaction_cost": float(getattr(base_env.cfg, "transaction_cost", 0.0)),
            "risk_lambda": float(getattr(base_env.cfg, "risk_lambda", 0.0)),
            "rebalance_eta": float(rebalance_eta) if rebalance_eta is not None else None,
        }
        if _is_step6_signature_extension_active(
            eta_mode=eta_mode,
            rule_vol_window=rule_vol_window,
            rule_vol_a=rule_vol_a,
            eta_clip_min=eta_clip_min,
            eta_clip_max=eta_clip_max,
        ):
            cost_params.update(
                {
                    "eta_mode": eta_mode,
                    "eta_clip_min": eta_clip_min,
                    "eta_clip_max": eta_clip_max,
                    "rule_vol_window": rule_vol_window,
                    "rule_vol_a": rule_vol_a,
                }
            )
        reward_type = env_params.get("reward_type", "log_net_minus_r2")
        feature_flags = {
            "returns_window": True,
            "volatility": True,
            "prev_weights": True,
            "reward_type": reward_type,
            "action_smoothing": _compute_action_smoothing_flag(
                eta_mode=eta_mode,
                rebalance_eta=float(rebalance_eta) if rebalance_eta is not None else None,
            ),
        }
    elif sig_version == "v2" or "risk_lambda" in env_params or "reward_type" in env_params:
        cost_params = {
            "transaction_cost": float(getattr(base_env.cfg, "transaction_cost", 0.0)),
            "risk_lambda": float(getattr(base_env.cfg, "risk_lambda", 0.0)),
        }
        reward_type = env_params.get("reward_type", "log_net_minus_r2")
        feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True, "reward_type": reward_type}
    else:
        cost_params = {"transaction_cost": float(getattr(base_env.cfg, "transaction_cost", 0.0))}
        feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True}

    assets = list(base_env.returns.columns)
    window_size = int(getattr(base_env, "window_size", 0))
    lv_value = int((run_meta or {}).get("Lv", 0) or 0)
    signature_hash = compute_env_signature(
        assets,
        window_size,
        lv_value if lv_value > 0 else None,
        feature_flags=feature_flags,
        cost_params=cost_params,
        schema_version="v1",
    )
    return {
        "env_signature_hash": signature_hash,
        "feature_flags": feature_flags,
        "cost_params": cost_params,
        "window_size": window_size,
        "Lv": lv_value if lv_value > 0 else None,
        "asset_list": assets,
    }


def build_eval_context(
    *,
    config_path: str,
    model_type: str,
    seed: int,
    model_root: str,
    offline: bool,
    max_steps: int,
    model_path_arg: str | None = None,
    prefer_metadata_config: bool = True,
) -> EvalContext:
    model_root_path = Path(model_root)
    model_path, run_meta = _resolve_model_path(
        model_path_arg=model_path_arg,
        model_type=model_type,
        seed=seed,
        model_root=model_root_path,
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cfg_path = Path(config_path)
    if prefer_metadata_config and run_meta and run_meta.get("config_path"):
        meta_cfg_path = Path(str(run_meta["config_path"]))
        if not meta_cfg_path.is_absolute():
            meta_cfg_path = ROOT / meta_cfg_path
        if meta_cfg_path.exists() and meta_cfg_path.resolve() != cfg_path.resolve():
            LOGGER.info("Using metadata config for model compatibility: %s", meta_cfg_path)
            cfg_path = meta_cfg_path

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["config_path"] = str(cfg_path)
    env_cfg = cfg["env"]
    prl_cfg = cfg.get("prl", {})
    data_cfg = cfg.get("data", {})

    if data_cfg.get("paper_mode", False) and not data_cfg.get("require_cache", False):
        raise ValueError("paper_mode=true requires require_cache=true.")

    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    offline_cfg = data_cfg.get("offline", False)
    resolved_offline = bool(offline or offline_cfg or paper_mode or require_cache_cfg)
    require_cache = bool(require_cache_cfg or paper_mode or resolved_offline)
    cache_only = bool(paper_mode or require_cache_cfg or offline_cfg or resolved_offline)
    session_opts = data_cfg.get("session_opts")

    lv = int(env_cfg["Lv"]) if env_cfg.get("Lv") is not None else int(run_meta.get("Lv"))
    window_size = int(env_cfg["L"]) if env_cfg.get("L") is not None else int(run_meta.get("L"))

    market, features = prepare_market_and_features(
        config=cfg,
        lv=lv,
        force_refresh=data_cfg.get("force_refresh", True),
        offline=resolved_offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        cache_only=cache_only,
        session_opts=session_opts,
    )
    expected_assets = list(run_meta.get("asset_list") or []) if run_meta else []
    market, features = _subset_universe(market, features, expected_assets)

    dates = cfg["dates"]
    eval_start, eval_end = _clip_eval_window(
        market.returns,
        str(dates["test_start"]),
        str(dates["test_end"]),
        int(max_steps),
    )

    run_id = model_path.stem
    if run_id.endswith("_final"):
        run_id = run_id[: -len("_final")]
    if run_meta and run_meta.get("run_id"):
        run_id = str(run_meta["run_id"])

    return EvalContext(
        cfg=cfg,
        env_cfg=env_cfg,
        prl_cfg=prl_cfg,
        run_meta=run_meta,
        run_id=run_id,
        model_path=model_path,
        model_type=model_type,
        seed=int(seed),
        window_size=window_size,
        lv=lv,
        market=market,
        features=features,
        eval_start=eval_start,
        eval_end=eval_end,
    )


def run_eval_case(
    ctx: EvalContext,
    *,
    eta_mode: str,
    rebalance_eta: float | None,
    transaction_cost: float,
    eval_tag: str,
):
    env_cfg = ctx.env_cfg
    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required.")
    rule_vol_window, rule_vol_a, eta_clip_min, eta_clip_max = _parse_rule_vol_env(env_cfg)
    risk_lambda = float(env_cfg.get("risk_lambda", 0.0))
    risk_penalty_type = str(env_cfg.get("risk_penalty_type", "r2"))

    env = build_env_for_range(
        market=ctx.market,
        features=ctx.features,
        start=ctx.eval_start,
        end=ctx.eval_end,
        window_size=ctx.window_size,
        c_tc=float(transaction_cost),
        seed=ctx.seed,
        logit_scale=float(env_cfg["logit_scale"]),
        random_reset=False,
        risk_lambda=risk_lambda,
        risk_penalty_type=risk_penalty_type,
        rebalance_eta=rebalance_eta,
        eta_mode=str(eta_mode),
        rule_vol_window=rule_vol_window,
        rule_vol_a=rule_vol_a,
        eta_clip_min=eta_clip_min,
        eta_clip_max=eta_clip_max,
    )

    scheduler: PRLAlphaScheduler | None = None
    if ctx.model_type == "prl":
        scheduler = create_scheduler(ctx.prl_cfg, ctx.window_size, ctx.market.returns.shape[1], ctx.features.stats_path)

    model = load_model(ctx.model_path, ctx.model_type, env, scheduler=scheduler)
    metrics, trace = run_backtest_episode_detailed(model, env)
    eval_id = f"{ctx.run_id}__{eval_tag}"
    df = trace_dict_to_frame(trace, eval_id=eval_id, run_id=ctx.run_id, model_type=ctx.model_type, seed=ctx.seed)
    signature_info = compute_current_env_signature(env, ctx.run_meta)
    return metrics, trace, df, signature_info


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required trace columns: {missing}")


def _check_turnover(df: pd.DataFrame, label: str) -> None:
    for col in ("turnover_exec", "turnover_target"):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().any():
            raise AssertionError(f"{label}: {col} contains NaN")
        if (s < 0.0).any():
            raise AssertionError(f"{label}: {col} contains negative values")


def run_sanity(ctx: EvalContext) -> None:
    base_tc = _get_transaction_cost(ctx.env_cfg)

    _, _, df_eta1, _ = run_eval_case(
        ctx,
        eta_mode="fixed",
        rebalance_eta=1.0,
        transaction_cost=base_tc,
        eval_tag="sanity_eta1",
    )
    _require_cols(
        df_eta1,
        [
            "cost_target",
            "net_return_lin_target",
            "tracking_error_l2",
            "eta_t",
            "turnover_exec",
            "turnover_target",
            "portfolio_return",
            "net_return_lin",
            "cost",
        ],
    )
    mean_tracking = float(pd.to_numeric(df_eta1["tracking_error_l2"], errors="coerce").mean())
    mean_turnover_gap = float(
        (pd.to_numeric(df_eta1["turnover_exec"], errors="coerce") - pd.to_numeric(df_eta1["turnover_target"], errors="coerce")).mean()
    )
    if not np.isfinite(mean_tracking) or mean_tracking >= 1e-6:
        raise AssertionError(f"TEST1 failed: mean(tracking_error_l2)={mean_tracking} >= 1e-6")
    if not np.isfinite(mean_turnover_gap) or abs(mean_turnover_gap) >= 1e-6:
        raise AssertionError(f"TEST1 failed: abs(mean(turnover_exec-turnover_target))={abs(mean_turnover_gap)} >= 1e-6")

    _, _, df_eta02, _ = run_eval_case(
        ctx,
        eta_mode="fixed",
        rebalance_eta=0.02,
        transaction_cost=base_tc,
        eval_tag="sanity_eta002",
    )
    avg_turnover_eta1 = float(pd.to_numeric(df_eta1["turnover_exec"], errors="coerce").mean())
    avg_turnover_eta02 = float(pd.to_numeric(df_eta02["turnover_exec"], errors="coerce").mean())
    if not (np.isfinite(avg_turnover_eta02) and np.isfinite(avg_turnover_eta1) and avg_turnover_eta02 < 0.5 * avg_turnover_eta1):
        raise AssertionError(
            "TEST2 failed: avg_turnover_exec_eta02 must be < 0.5 * avg_turnover_exec_eta1 "
            f"(eta02={avg_turnover_eta02}, eta1={avg_turnover_eta1})"
        )

    _, _, df_tc0, _ = run_eval_case(
        ctx,
        eta_mode="fixed",
        rebalance_eta=1.0,
        transaction_cost=0.0,
        eval_tag="sanity_tc0",
    )
    mean_abs_cost = float(pd.to_numeric(df_tc0["cost"], errors="coerce").abs().mean())
    mean_abs_net_vs_port = float(
        (pd.to_numeric(df_tc0["net_return_lin"], errors="coerce") - pd.to_numeric(df_tc0["portfolio_return"], errors="coerce"))
        .abs()
        .mean()
    )
    if not np.isfinite(mean_abs_cost) or mean_abs_cost >= 1e-10:
        raise AssertionError(f"TEST3 failed: mean(abs(cost_exec))={mean_abs_cost} >= 1e-10")
    if not np.isfinite(mean_abs_net_vs_port) or mean_abs_net_vs_port >= 1e-10:
        raise AssertionError(f"TEST3 failed: mean(abs(net_return_lin-portfolio_return))={mean_abs_net_vs_port} >= 1e-10")

    _check_turnover(df_eta1, "TEST5/eta1")
    _check_turnover(df_eta02, "TEST5/eta002")
    _check_turnover(df_tc0, "TEST5/tc0")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        ctx = build_eval_context(
            config_path=args.config,
            model_type=args.model_type,
            seed=int(args.seed),
            model_root=args.model_root,
            offline=bool(args.offline),
            max_steps=int(args.max_steps),
            model_path_arg=args.model_path,
            prefer_metadata_config=True,
        )
        run_sanity(ctx)
    except Exception as exc:  # noqa: BLE001 - explicit production gate failure
        print(f"STEP6 SANITY FAILED: {exc}")
        raise SystemExit(1)
    print("STEP6 SANITY PASSED")


if __name__ == "__main__":
    main()
