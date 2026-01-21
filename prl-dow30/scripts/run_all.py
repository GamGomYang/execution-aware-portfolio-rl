import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prl.baselines import run_all_baselines_detailed
from prl.data import slice_frame
from prl.eval import assert_env_compatible, load_model, run_backtest_episode_detailed
from prl.features import load_vol_stats
from prl.metrics import compute_metrics
from prl.train import (
    build_env_for_range,
    create_scheduler,
    prepare_market_and_features,
    run_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run training + evaluation for multiple seeds/model types.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="YAML config.")
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["baseline", "prl"],
        default=["baseline", "prl"],
        help="Model variants to execute.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds to iterate (defaults to config).")
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    return parser.parse_args()


def write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "model_type",
        "seed",
        "period",
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "avg_turnover",
        "total_turnover",
        "sharpe",
        "max_drawdown",
        "steps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summary_rows = []
    metric_cols = [
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "avg_turnover",
        "total_turnover",
        "sharpe",
        "max_drawdown",
        "steps",
    ]
    for model_type, group in df.groupby("model_type"):
        row = {"model_type": model_type}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        summary_rows.append(row)
    return summary_rows


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = float(np.mean(sample))
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def summarize_seed_stats(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["model_type", "seed"])
    metric_cols = [
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "avg_turnover",
        "total_turnover",
        "sharpe",
        "max_drawdown",
        "steps",
    ]
    summary_rows: list[dict] = []
    for model_type, group in df.groupby("model_type"):
        row = {"model_type": model_type, "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            values = group[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(values)) if values.size else float("nan")
            row[f"{col}_std"] = float(np.std(values, ddof=0)) if values.size else float("nan")
            ci_low, ci_high = _bootstrap_ci(values)
            row[f"{col}_ci_low"] = ci_low
            row[f"{col}_ci_high"] = ci_high
        summary_rows.append(row)
    return summary_rows


def write_regime_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _align_returns_vol(returns: pd.DataFrame, volatility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vol_clean = volatility.dropna()
    idx = returns.index.intersection(vol_clean.index)
    return returns.loc[idx], vol_clean.loc[idx]


def _compute_vz_series(volatility: pd.DataFrame, stats_path: Path) -> pd.Series:
    mean, std = load_vol_stats(stats_path)
    portfolio_vol = volatility.mean(axis=1)
    vz = (portfolio_vol - mean) / (std + 1e-8)
    return vz.astype(np.float64)


def _assign_regime(vz: pd.Series, q33: float, q66: float) -> pd.Series:
    regimes = pd.Series(index=vz.index, dtype="object")
    regimes[vz < q33] = "low"
    regimes[(vz >= q33) & (vz < q66)] = "mid"
    regimes[vz >= q66] = "high"
    return regimes


def _build_trace_df(run_id: str, model_type: str, seed: int, trace: dict) -> pd.DataFrame:
    turnover_target_changes = trace.get("turnover_target_changes")
    dates = trace.get("dates", [])
    if turnover_target_changes is None or not turnover_target_changes:
        turnover_target_changes = [np.nan] * len(dates)
    df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": trace.get("portfolio_returns", []),
            "reward": trace.get("rewards", []),
            "turnover": trace.get("turnovers", []),
            "turnover_target_change": turnover_target_changes,
        }
    )
    df["run_id"] = run_id
    df["model_type"] = model_type
    df["seed"] = seed
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_run_metadata(run_id: str, reports_dir: Path) -> dict | None:
    meta_path = reports_dir / f"run_metadata_{run_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _update_run_metadata(path: Path, updates: dict) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data, indent=2))


def _regime_metrics_from_trace(trace_df: pd.DataFrame, run_id: str, seed: int, include_all: bool = True) -> list[dict]:
    rows: list[dict] = []
    for model_type, group in trace_df.groupby("model_type"):
        for regime, regime_group in group.groupby("regime"):
            metrics = compute_metrics(
                regime_group["reward"].tolist(),
                regime_group["portfolio_return"].tolist(),
                regime_group["turnover"].tolist(),
            )
            rows.append(
                {
                    "run_id": run_id,
                    "model_type": model_type,
                    "seed": seed,
                    "regime": regime,
                    **metrics.to_dict(),
                }
            )
        if include_all:
            metrics = compute_metrics(
                group["reward"].tolist(),
                group["portfolio_return"].tolist(),
                group["turnover"].tolist(),
            )
            rows.append(
                {
                    "run_id": run_id,
                    "model_type": model_type,
                    "seed": seed,
                    "regime": "all",
                    **metrics.to_dict(),
                }
            )
    return rows


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["config_path"] = args.config
    dates = cfg["dates"]
    env_cfg = cfg["env"]
    prl_cfg = cfg.get("prl", {})
    data_cfg = cfg.get("data", {})
    if data_cfg.get("paper_mode", False) and not data_cfg.get("require_cache", False):
        raise ValueError("paper_mode=true requires require_cache=true.")
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    offline_cfg = data_cfg.get("offline", False)
    offline = args.offline or offline_cfg or paper_mode or require_cache_cfg
    require_cache = require_cache_cfg or paper_mode or offline
    cache_only = paper_mode or require_cache_cfg or offline_cfg or args.offline
    session_opts = data_cfg.get("session_opts", None)

    market, features = prepare_market_and_features(
        config=cfg,
        lv=env_cfg["Lv"],
        force_refresh=data_cfg.get("force_refresh", True),
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=session_opts,
        cache_only=cache_only,
    )

    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required for training/evaluation.")

    seeds = args.seeds or cfg.get("seeds", [0, 1, 2])
    returns_slice = slice_frame(market.returns, dates["test_start"], dates["test_end"])
    vol_slice = slice_frame(features.volatility, dates["test_start"], dates["test_end"])
    returns_slice, vol_slice = _align_returns_vol(returns_slice, vol_slice)
    vz_series = _compute_vz_series(vol_slice, features.stats_path)
    q33, q66 = np.quantile(vz_series.values, [1.0 / 3.0, 2.0 / 3.0])
    regime_series = _assign_regime(vz_series, q33, q66)
    regime_df = pd.DataFrame({"date": vz_series.index, "vz": vz_series.values, "regime": regime_series.values})

    baseline_results = run_all_baselines_detailed(
        returns_slice,
        vol_slice,
        transaction_cost=env_cfg["c_tc"],
    )

    metrics_rows: list[dict] = []
    regime_rows: list[dict] = []
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        baseline_meta = None
        baseline_run_id = None
        meta_by_type: dict[str, dict] = {}
        for model_type in args.model_types:
            model_path = run_training(
                config=cfg,
                model_type=model_type,
                seed=seed,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                output_dir="outputs/models",
                force_refresh=data_cfg.get("force_refresh", True),
                offline=offline,
                cache_only=cache_only,
            )

            env = build_env_for_range(
                market=market,
                features=features,
                start=dates["test_start"],
                end=dates["test_end"],
                window_size=env_cfg["L"],
                c_tc=env_cfg["c_tc"],
                seed=seed,
                logit_scale=env_cfg["logit_scale"],
            )

            scheduler = None
            if model_type == "prl":
                num_assets = market.returns.shape[1]
                scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)

            label = f"{model_type}_sac"
            run_id = model_path.stem
            if run_id.endswith("_final"):
                run_id = run_id[: -len("_final")]
            meta = _load_run_metadata(run_id, reports_dir)
            if meta is None:
                raise ValueError(f"RUN_METADATA_NOT_FOUND for run_id={run_id}")
            assert_env_compatible(env, meta, Lv=env_cfg.get("Lv"))
            meta_by_type[model_type] = meta
            model = load_model(model_path, model_type, env, scheduler=scheduler)
            metrics, trace = run_backtest_episode_detailed(model, env)
            metrics_row_id = len(metrics_rows)
            metrics_rows.append(
                {
                    "run_id": run_id,
                    "model_type": label,
                    "seed": seed,
                    "period": "test",
                    **metrics.to_dict(),
                }
            )
            if model_type == "baseline":
                baseline_meta = meta
                baseline_run_id = run_id
            if "baseline" in meta_by_type and "prl" in meta_by_type:
                base_meta = meta_by_type["baseline"]
                prl_meta = meta_by_type["prl"]
                if base_meta.get("data_manifest_hash") != prl_meta.get("data_manifest_hash"):
                    raise ValueError("DATA_MANIFEST_HASH_MISMATCH")
                if base_meta.get("env_signature_hash") != prl_meta.get("env_signature_hash"):
                    raise ValueError("ENV_SIGNATURE_HASH_MISMATCH")

            trace_frames = [_build_trace_df(run_id, label, seed, trace)]
            for name, (_, base_trace) in baseline_results.items():
                trace_frames.append(_build_trace_df(run_id, name, seed, base_trace))

            trace_df = pd.concat(trace_frames, ignore_index=True)
            trace_df = trace_df.merge(regime_df, on="date", how="left")
            trace_df = trace_df.dropna(subset=["regime"])
            trace_df["regime_label"] = trace_df["regime"]

            trace_path = reports_dir / f"trace_{run_id}.parquet"
            trace_df.to_parquet(trace_path, index=False)

            thresholds = {
                "q33": float(q33),
                "q66": float(q66),
                "regime_policy": "Vz quantile split on test period",
            }
            thresholds_path = reports_dir / f"regime_thresholds_{run_id}.json"
            thresholds_path.write_text(json.dumps(thresholds, indent=2))

            regime_rows.extend(_regime_metrics_from_trace(trace_df, run_id, seed))
            _update_run_metadata(
                reports_dir / f"run_metadata_{run_id}.json",
                {
                    "evaluation": {
                        "trace_path": str(trace_path),
                        "regime_thresholds_path": str(thresholds_path),
                        "metrics_row_id": metrics_row_id,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )

        baseline_run_id = baseline_run_id or f"baseline_strategies_seed{seed}"
        for name, (base_metrics, _) in baseline_results.items():
            metrics_rows.append(
                {
                    "run_id": baseline_run_id,
                    "model_type": name,
                    "seed": seed,
                    "period": "test",
                    **base_metrics.to_dict(),
                }
            )

    write_metrics(reports_dir / "metrics.csv", metrics_rows)
    summary_rows = summarize_metrics(metrics_rows)
    write_summary(reports_dir / "summary.csv", summary_rows)
    write_regime_metrics(reports_dir / "regime_metrics.csv", regime_rows)
    summary_seed_rows = summarize_seed_stats(metrics_rows)
    write_summary(reports_dir / "summary_seed_stats.csv", summary_seed_rows)
    print("Completed run_all workflow.")


if __name__ == "__main__":
    main()
