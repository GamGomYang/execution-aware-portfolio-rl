from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.data import MarketData
from prl.eval import load_model, run_backtest_episode_detailed, trace_dict_to_frame
from prl.features import VolatilityFeatures
from prl.train import build_env_for_range, create_scheduler, prepare_market_and_features


LOGGER = logging.getLogger(__name__)
DEFAULT_ETAS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6 eval-only eta frontier experiment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model-type", choices=["baseline", "prl"], required=True, help="Trained model type.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the trained model.")
    parser.add_argument("--model-path", type=str, help="Optional explicit model path.")
    parser.add_argument(
        "--etas",
        type=str,
        default=",".join(str(x) for x in DEFAULT_ETAS),
        help="Comma-separated eta list. Example: 1.0,0.5,0.2,0.1,0.05,0.02",
    )
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Base directory for reports/models/logs (default: outputs).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/step6/eta_frontier_eval.csv",
        help="Output CSV path (default: outputs/step6/eta_frontier_eval.csv).",
    )
    return parser.parse_args()


def _archive_config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if k != "config_path"}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _load_latest_run_metadata(model_type: str, seed: int, reports_dir: Path) -> dict | None:
    if not reports_dir.exists():
        return None
    candidates = []
    for path in reports_dir.glob("run_metadata_*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("model_type") != model_type or int(data.get("seed", -1)) != seed:
            continue
        created_at = data.get("created_at") or ""
        candidates.append((created_at, data))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_run_metadata_for_model(model_path: Path, reports_dir: Path) -> dict | None:
    run_id = model_path.stem
    if run_id.endswith("_final"):
        run_id = run_id[: -len("_final")]
    meta_path = reports_dir / f"run_metadata_{run_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _parse_etas(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("No eta values provided.")
    etas = [float(p) for p in parts]
    for eta in etas:
        if not np.isfinite(eta) or eta <= 0.0 or eta > 1.0:
            raise ValueError(f"Each eta must satisfy 0 < eta <= 1, got: {eta}")
    return etas


def _compute_sharpe(returns: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(returns)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = float(arr.std(ddof=0))
    if std <= 1e-8:
        return 0.0
    mean = float(arr.mean())
    return float((mean / std) * np.sqrt(252.0))


def _compute_max_drawdown(equity_curve: Iterable[float]) -> float:
    eq = pd.to_numeric(pd.Series(list(equity_curve)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if eq.size == 0:
        return 0.0
    run_max = np.maximum.accumulate(eq)
    dd = eq / run_max - 1.0
    return float(np.min(dd))


def _compute_cagr(equity_curve: Iterable[float], periods_per_year: int = 252) -> float:
    eq = pd.to_numeric(pd.Series(list(equity_curve)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if eq.size == 0:
        return 0.0
    final = float(eq[-1])
    if final <= 0.0:
        return float("nan")
    years = float(eq.size) / float(periods_per_year)
    if years <= 0.0:
        return 0.0
    return float(final ** (1.0 / years) - 1.0)


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Trace frame missing required columns: {missing}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    etas = _parse_etas(args.etas)

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    models_dir = output_root / "models"

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None:
        latest_meta = _load_latest_run_metadata(args.model_type, args.seed, reports_dir)
        if latest_meta:
            artifact_paths = latest_meta.get("artifact_paths") or latest_meta.get("artifacts") or {}
            model_path_value = artifact_paths.get("model_path")
            if model_path_value:
                model_path = Path(model_path_value)
        if model_path is None:
            model_path = models_dir / f"{args.model_type}_seed{args.seed}_final.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_meta = _load_run_metadata_for_model(model_path, reports_dir)
    run_id = model_path.stem
    if run_id.endswith("_final"):
        run_id = run_id[: -len("_final")]
    if run_meta and run_meta.get("run_id"):
        run_id = str(run_meta["run_id"])

    cfg_path = Path(args.config)
    if run_meta and run_meta.get("config_path"):
        meta_cfg_path = Path(str(run_meta["config_path"]))
        if not meta_cfg_path.is_absolute():
            meta_cfg_path = ROOT / meta_cfg_path
        if meta_cfg_path.exists() and meta_cfg_path.resolve() != cfg_path.resolve():
            LOGGER.info("Using model metadata config for compatibility: %s", meta_cfg_path)
            cfg_path = meta_cfg_path

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["config_path"] = str(cfg_path)
    dates = cfg["dates"]
    env_cfg = cfg["env"]
    prl_cfg = cfg.get("prl", {})
    data_cfg = cfg.get("data", {})
    if data_cfg.get("paper_mode", False) and not data_cfg.get("require_cache", False):
        raise ValueError("paper_mode=true requires require_cache=true.")

    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    offline_cfg = data_cfg.get("offline", False)
    offline = args.offline or offline_cfg or paper_mode or require_cache_cfg
    require_cache = require_cache_cfg or paper_mode or offline
    cache_only = paper_mode or require_cache_cfg or offline_cfg or args.offline
    session_opts = data_cfg.get("session_opts", None)

    lv_for_features = int(run_meta.get("Lv")) if run_meta and run_meta.get("Lv") is not None else int(env_cfg["Lv"])
    window_size_for_eval = int(run_meta.get("L")) if run_meta and run_meta.get("L") is not None else int(env_cfg["L"])

    market, features = prepare_market_and_features(
        config=cfg,
        lv=lv_for_features,
        force_refresh=data_cfg.get("force_refresh", True),
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        cache_only=cache_only,
        session_opts=session_opts,
    )

    expected_assets = list(run_meta.get("asset_list") or []) if run_meta else []
    if expected_assets:
        missing_assets = [a for a in expected_assets if a not in market.returns.columns]
        if missing_assets:
            raise ValueError(f"Assets required by model metadata are missing in current market data: {missing_assets}")
        market = MarketData(
            prices=market.prices[expected_assets],
            returns=market.returns[expected_assets],
            manifest=market.manifest,
            quality_summary=market.quality_summary,
        )
        vol_subset = features.volatility[expected_assets]
        features = VolatilityFeatures(
            volatility=vol_subset,
            portfolio_scalar=vol_subset.mean(axis=1),
            stats_path=features.stats_path,
            mean=features.mean,
            std=features.std,
        )

    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required for evaluation.")

    scheduler = None
    if args.model_type == "prl":
        num_assets = market.returns.shape[1]
        scheduler = create_scheduler(prl_cfg, window_size_for_eval, num_assets, features.stats_path)

    rows: list[dict[str, float | int | str]] = []
    config_hash = _archive_config_hash(cfg)
    for eta in etas:
        env = build_env_for_range(
            market=market,
            features=features,
            start=dates["test_start"],
            end=dates["test_end"],
            window_size=window_size_for_eval,
            c_tc=env_cfg["c_tc"],
            seed=args.seed,
            logit_scale=env_cfg["logit_scale"],
            risk_lambda=env_cfg.get("risk_lambda", 0.0),
            risk_penalty_type=env_cfg.get("risk_penalty_type", "r2"),
            rebalance_eta=eta,
            eta_mode="fixed",
            rule_vol_window=env_cfg.get("rule_vol", {}).get("window", 20),
            rule_vol_a=env_cfg.get("rule_vol", {}).get("a", 1.0),
            eta_clip_min=(env_cfg.get("rule_vol", {}).get("eta_clip", [0.02, 0.5])[0]),
            eta_clip_max=(env_cfg.get("rule_vol", {}).get("eta_clip", [0.02, 0.5])[1]),
        )

        model = load_model(model_path, args.model_type, env, scheduler=scheduler)
        metrics, trace = run_backtest_episode_detailed(model, env)
        eval_id = f"{run_id}__eta{eta:.4f}"
        df = trace_dict_to_frame(trace, eval_id=eval_id, run_id=run_id, model_type=args.model_type, seed=args.seed)
        _require_columns(
            df,
            (
                "net_return_lin",
                "net_return_lin_target",
                "equity_net_lin",
                "turnover_exec",
                "tracking_error_l2",
            ),
        )

        misalignment_gap = pd.to_numeric(df["net_return_lin"], errors="coerce") - pd.to_numeric(
            df["net_return_lin_target"], errors="coerce"
        )
        row = {
            "run_id": run_id,
            "model_type": args.model_type,
            "seed": int(args.seed),
            "config_hash": config_hash,
            "window_size": int(window_size_for_eval),
            "Lv": int(lv_for_features),
            "eta_mode": "fixed",
            "eta": float(eta),
            "n_steps": int(len(df)),
            "sharpe": float(_compute_sharpe(df["net_return_lin"])),
            "cagr": float(_compute_cagr(df["equity_net_lin"])),
            "maxdd": float(_compute_max_drawdown(df["equity_net_lin"])),
            "turnover_exec_mean": float(pd.to_numeric(df["turnover_exec"], errors="coerce").mean()),
            "tracking_error_l2_mean": float(pd.to_numeric(df["tracking_error_l2"], errors="coerce").mean()),
            "misalignment_gap_mean": float(pd.to_numeric(misalignment_gap, errors="coerce").mean()),
            "metrics_sharpe_net_lin": float(metrics.sharpe_net_lin) if metrics.sharpe_net_lin is not None else float("nan"),
            "metrics_maxdd_net_lin": float(metrics.max_drawdown_net_lin)
            if metrics.max_drawdown_net_lin is not None
            else float("nan"),
            "metrics_cumret_net_lin": float(metrics.cumulative_return_net_lin)
            if metrics.cumulative_return_net_lin is not None
            else float("nan"),
        }
        rows.append(row)
        LOGGER.info(
            "eta=%.4f sharpe=%.6f cagr=%.6f maxdd=%.6f turnover_exec_mean=%.6f tracking_error_mean=%.6f gap=%.6f",
            row["eta"],
            row["sharpe"],
            row["cagr"],
            row["maxdd"],
            row["turnover_exec_mean"],
            row["tracking_error_l2_mean"],
            row["misalignment_gap_mean"],
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_df = pd.DataFrame(rows)
    frontier_df.to_csv(out_path, index=False)
    print(f"Saved eta frontier: {out_path}")


if __name__ == "__main__":
    main()
