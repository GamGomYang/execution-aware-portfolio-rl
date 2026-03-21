import argparse
import csv
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prl.data import slice_frame
from prl.eval import (
    assert_env_compatible,
    compute_regime_labels,
    eval_model_to_trace,
    eval_strategies_to_trace,
    load_model,
    summarize_regime_metrics,
)
from prl.features import load_vol_stats
from prl.train import (
    build_env_for_range,
    build_signal_features,
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
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Base directory for reports/models/logs (default: config.output.root or outputs).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Override sac.total_timesteps for quick smoke runs.",
    )
    return parser.parse_args()


def _resolve_output_root(cli_output_root: str | None, cfg: dict) -> Path:
    if cli_output_root:
        return Path(cli_output_root)
    output_cfg = cfg.get("output", {}) or {}
    cfg_root = output_cfg.get("root")
    if cfg_root:
        return Path(cfg_root)
    return Path("outputs")


def _mean_std_safe(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std <= 1e-8:
        std = 0.0
    return mean, std


def _archive_config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if k != "config_path"}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _archive_exp_id(config_path: str, cfg: dict) -> str:
    stem = Path(config_path).stem
    return f"{stem}__{_archive_config_hash(cfg)}"


def _next_archive_path(archive_dir: Path, prefix: str, exp_id: str) -> Path:
    candidate = archive_dir / f"{prefix}_{exp_id}.csv"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = archive_dir / f"{prefix}_{exp_id}__{idx}.csv"
        if not candidate.exists():
            return candidate
        idx += 1


def _archive_reports(
    reports_dir: Path,
    *,
    config_path: str,
    cfg: dict,
) -> dict[str, str]:
    archive_dir = reports_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    exp_id = _archive_exp_id(config_path, cfg)
    written: dict[str, str] = {}
    targets = {
        "metrics": reports_dir / "metrics.csv",
        "summary": reports_dir / "summary.csv",
        "regime_metrics": reports_dir / "regime_metrics.csv",
    }
    for prefix, src in targets.items():
        if not src.exists():
            continue
        dst = _next_archive_path(archive_dir, prefix, exp_id)
        shutil.copy2(src, dst)
        written[prefix] = str(dst)
    return written


def write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "eval_id",
        "eval_window",
        "model_type",
        "seed",
        "period",
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "cumulative_return_net_exp",
        "cumulative_return_net_lin",
        "avg_turnover",
        "total_turnover",
        "avg_turnover_exec",
        "total_turnover_exec",
        "avg_turnover_target",
        "total_turnover_target",
        "sharpe",
        "sharpe_net_exp",
        "sharpe_net_lin",
        "max_drawdown",
        "max_drawdown_net_exp",
        "max_drawdown_net_lin",
        "mean_daily_return_gross",
        "std_daily_return_gross",
        "mean_daily_net_return_exp",
        "std_daily_net_return_exp",
        "mean_daily_net_return_lin",
        "std_daily_net_return_lin",
        "mean_daily_return_gross_mid",
        "std_daily_return_gross_mid",
        "mean_daily_net_return_exp_mid",
        "std_daily_net_return_exp_mid",
        "mean_daily_net_return_lin_mid",
        "std_daily_net_return_lin_mid",
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
        "cumulative_return_net_exp",
        "cumulative_return_net_lin",
        "avg_turnover",
        "total_turnover",
        "avg_turnover_exec",
        "total_turnover_exec",
        "avg_turnover_target",
        "total_turnover_target",
        "sharpe",
        "sharpe_net_exp",
        "sharpe_net_lin",
        "max_drawdown",
        "max_drawdown_net_exp",
        "max_drawdown_net_lin",
        "mean_daily_return_gross",
        "std_daily_return_gross",
        "mean_daily_net_return_exp",
        "std_daily_net_return_exp",
        "mean_daily_net_return_lin",
        "std_daily_net_return_lin",
        "mean_daily_return_gross_mid",
        "std_daily_return_gross_mid",
        "mean_daily_net_return_exp_mid",
        "std_daily_net_return_exp_mid",
        "mean_daily_net_return_lin_mid",
        "std_daily_net_return_lin_mid",
        "steps",
    ]
    group_cols = ["model_type"]
    if "eval_window" in df.columns:
        group_cols = ["eval_window"] + group_cols
    for keys, group in df.groupby(group_cols):
        if len(group_cols) == 2:
            eval_window, model_type = keys
            row = {"eval_window": eval_window, "model_type": model_type}
        else:
            model_type = keys
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
    subset = ["model_type", "seed"]
    if "eval_window" in df.columns:
        subset = ["eval_window"] + subset
    df = df.drop_duplicates(subset=subset)
    metric_cols = [
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "cumulative_return_net_exp",
        "cumulative_return_net_lin",
        "avg_turnover",
        "total_turnover",
        "avg_turnover_exec",
        "total_turnover_exec",
        "avg_turnover_target",
        "total_turnover_target",
        "sharpe",
        "sharpe_net_exp",
        "sharpe_net_lin",
        "max_drawdown",
        "max_drawdown_net_exp",
        "max_drawdown_net_lin",
        "mean_daily_return_gross",
        "std_daily_return_gross",
        "mean_daily_net_return_exp",
        "std_daily_net_return_exp",
        "mean_daily_net_return_lin",
        "std_daily_net_return_lin",
        "mean_daily_return_gross_mid",
        "std_daily_return_gross_mid",
        "mean_daily_net_return_exp_mid",
        "std_daily_net_return_exp_mid",
        "mean_daily_net_return_lin_mid",
        "std_daily_net_return_lin_mid",
        "steps",
    ]
    summary_rows: list[dict] = []
    group_cols = ["model_type"]
    if "eval_window" in df.columns:
        group_cols = ["eval_window"] + group_cols
    for keys, group in df.groupby(group_cols):
        if len(group_cols) == 2:
            eval_window, model_type = keys
            row = {"eval_window": eval_window, "model_type": model_type, "n_seeds": int(group["seed"].nunique())}
        else:
            model_type = keys
            row = {"model_type": model_type, "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            values = group[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(values)) if values.size else float("nan")
            row[f"{col}_std"] = float(np.std(values, ddof=0)) if values.size else float("nan")
            if values.size:
                p25, p50, p75 = np.quantile(values, [0.25, 0.50, 0.75])
                row[f"{col}_p25"] = float(p25)
                row[f"{col}_median"] = float(p50)
                row[f"{col}_p75"] = float(p75)
                row[f"{col}_iqr"] = float(p75 - p25)
            else:
                row[f"{col}_p25"] = float("nan")
                row[f"{col}_median"] = float("nan")
                row[f"{col}_p75"] = float("nan")
                row[f"{col}_iqr"] = float("nan")
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


def _resolve_eval_windows(cfg: dict) -> list[dict]:
    eval_cfg = cfg.get("eval", {}) or {}
    windows = eval_cfg.get("windows")
    if windows:
        resolved = []
        for idx, win in enumerate(windows):
            name = win.get("name") or f"eval_{idx}"
            start = win.get("start") or win.get("eval_start")
            end = win.get("end") or win.get("eval_end")
            if not start or not end:
                raise ValueError("EVAL_WINDOW_MISSING_START_END")
            resolved.append({"name": name, "start": start, "end": end})
        return resolved
    dates = cfg["dates"]
    start = eval_cfg.get("eval_start") or dates["test_start"]
    end = eval_cfg.get("eval_end") or dates["test_end"]
    name = eval_cfg.get("name") or eval_cfg.get("eval_window") or "test"
    return [{"name": name, "start": start, "end": end}]


def _eval_window_from_trace(trace_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    if trace_df.empty or "date" not in trace_df.columns:
        raise ValueError("TRACE_DATES_MISSING")
    dates = pd.to_datetime(trace_df["date"])
    eval_start = dates.min()
    eval_end = dates.max()
    eval_num_days = int(dates.nunique())
    return eval_start, eval_end, eval_num_days


def _build_thresholds(
    vz_series: pd.Series,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    eval_num_days: int,
) -> dict:
    q33, q66 = np.quantile(vz_series.values, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "q33": float(q33),
        "q66": float(q66),
        "regime_policy": "Vz quantile split on aligned eval window",
        "eval_start_date": eval_start.date().isoformat(),
        "eval_end_date": eval_end.date().isoformat(),
        "eval_num_days": int(eval_num_days),
    }


def _load_run_metadata(run_id: str, reports_dir: Path) -> dict | None:
    meta_path = reports_dir / f"run_metadata_{run_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _update_run_metadata(path: Path, updates: dict) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    if "evaluations" in updates:
        merged = data.get("evaluations", {})
        merged.update(updates["evaluations"])
        data["evaluations"] = merged
        updates = {k: v for k, v in updates.items() if k != "evaluations"}
    if "evaluation" in updates and isinstance(data.get("evaluation"), dict) and isinstance(updates["evaluation"], dict):
        merged_eval = {**data["evaluation"], **updates["evaluation"]}
        data["evaluation"] = merged_eval
        updates = {k: v for k, v in updates.items() if k != "evaluation"}
    data.update(updates)
    report_paths = data.get("report_paths", {})
    eval_section = updates.get("evaluation") or updates.get("evaluations", {})
    if isinstance(eval_section, dict):
        last_eval = eval_section if "trace_path" in eval_section else None
        if "evaluations" in updates and isinstance(updates["evaluations"], dict):
            last_eval = list(updates["evaluations"].values())[-1]
        if last_eval:
            trace_path = last_eval.get("trace_path")
            thresholds_path = last_eval.get("regime_thresholds_path")
            if trace_path:
                report_paths["trace_path"] = trace_path
            if thresholds_path:
                report_paths["regime_thresholds_path"] = thresholds_path
            data["report_paths"] = report_paths
    path.write_text(json.dumps(data, indent=2))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["config_path"] = args.config
    output_root = _resolve_output_root(args.output_root, cfg)
    if args.total_timesteps:
        cfg.setdefault("sac", {})
        cfg["sac"]["total_timesteps"] = args.total_timesteps
    dates = cfg["dates"]
    env_cfg = cfg["env"]
    prl_cfg = cfg.get("prl", {})
    data_cfg = cfg.get("data", {})
    if prl_cfg:
        multiplier = prl_cfg.get("mid_plasticity_multiplier")
        if multiplier is not None:
            logging.info("PRL mid_plasticity_multiplier=%s", multiplier)
        else:
            logging.info("PRL mid_plasticity_multiplier not set")
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
    signal_features, _ = build_signal_features(market, config=cfg)

    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required for training/evaluation.")

    seeds = args.seeds or cfg.get("seeds", [0, 1, 2])
    eval_windows = _resolve_eval_windows(cfg)
    multi_window = len(eval_windows) > 1
    eval_cfg = cfg.get("eval", {}) or {}
    write_trace = eval_cfg.get("write_trace", True)
    trace_stride = int(eval_cfg.get("trace_stride", 1))
    if trace_stride < 1:
        raise ValueError("trace_stride must be >= 1")
    run_baselines_flag = eval_cfg.get("run_baselines", True)
    write_step4 = eval_cfg.get("write_step4", True) and write_trace

    metrics_rows: list[dict] = []
    regime_rows: list[dict] = []
    run_ids_this_session: list[str] = []
    step4_targets: set[str] = set()
    reports_dir = output_root / "reports"
    traces_dir = output_root / "traces"
    models_dir = output_root / "models"
    logs_dir = output_root / "logs"
    for path in (reports_dir, traces_dir, models_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        baseline_strategy_run_id = f"baseline_strategies_seed{seed}"
        baseline_emitted_windows: set[str] = set()
        meta_by_type: dict[str, dict] = {}
        for model_type in args.model_types:
            model_path = run_training(
                config=cfg,
                model_type=model_type,
                seed=seed,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                output_dir=models_dir,
                reports_dir=reports_dir,
                logs_dir=logs_dir,
                force_refresh=data_cfg.get("force_refresh", True),
                offline=offline,
                cache_only=cache_only,
            )

            scheduler = None
            if model_type == "prl":
                num_assets = market.returns.shape[1]
                scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)

            label = f"{model_type}_sac"
            run_id = model_path.stem
            if run_id.endswith("_final"):
                run_id = run_id[: -len("_final")]
            eval_id = run_id
            meta = _load_run_metadata(run_id, reports_dir)
            if meta is None:
                raise ValueError(f"RUN_METADATA_NOT_FOUND for run_id={run_id}")
            meta_by_type[model_type] = meta
            step4_targets.add(run_id)

            for window in eval_windows:
                env = build_env_for_range(
                    market=market,
                    features=features,
                    start=window["start"],
                    end=window["end"],
                    window_size=env_cfg["L"],
                    c_tc=env_cfg["c_tc"],
                    seed=seed,
                    logit_scale=env_cfg["logit_scale"],
                    risk_lambda=env_cfg.get("risk_lambda", 0.0),
                    risk_penalty_type=env_cfg.get("risk_penalty_type", "r2"),
                    rebalance_eta=env_cfg.get("rebalance_eta"),
                    signal_features=signal_features,
                )

                assert_env_compatible(env, meta, Lv=env_cfg.get("Lv"))
                model = load_model(model_path, model_type, env, scheduler=scheduler)
                metrics, trace_df = eval_model_to_trace(
                    model,
                    env,
                    eval_id=eval_id,
                    run_id=run_id,
                    model_type=label,
                    seed=seed,
                )
                trace_df["eval_window"] = window["name"]
                metrics_row_id = len(metrics_rows)
                metrics_rows.append(
                    {
                        "run_id": run_id,
                        "eval_id": eval_id,
                        "eval_window": window["name"],
                        "model_type": label,
                        "seed": seed,
                        "period": "test",
                        **metrics.to_dict(),
                    }
                )
                # mid-only daily return mean/std (uses regime labels computed below)
                metrics_row_ref = metrics_rows[metrics_row_id]
                returns_slice = slice_frame(market.returns, window["start"], window["end"])
                vol_slice = slice_frame(features.volatility, window["start"], window["end"])
                returns_slice, vol_slice = _align_returns_vol(returns_slice, vol_slice)
                eval_start, eval_end, eval_num_days = _eval_window_from_trace(trace_df)
                aligned_returns = slice_frame(returns_slice, eval_start, eval_end)
                aligned_vol = slice_frame(vol_slice, eval_start, eval_end)
                aligned_returns, aligned_vol = _align_returns_vol(aligned_returns, aligned_vol)
                vz_series = _compute_vz_series(aligned_vol, features.stats_path)
                thresholds = _build_thresholds(
                    vz_series,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    eval_num_days=eval_num_days,
                )
                vz_df = pd.DataFrame({"date": vz_series.index, "vz": vz_series.values})
                trace_df = trace_df.merge(vz_df, on="date", how="left")
                trace_df = compute_regime_labels(trace_df, thresholds)
                mid_df = trace_df[trace_df["regime"] == "mid"]
                mean_gross_mid, std_gross_mid = _mean_std_safe(mid_df["portfolio_return"].tolist())
                mean_exp_mid, std_exp_mid = _mean_std_safe(mid_df["net_return_exp"].tolist())
                mean_lin_mid, std_lin_mid = _mean_std_safe(mid_df["net_return_lin"].tolist())
                metrics_row_ref["mean_daily_return_gross_mid"] = mean_gross_mid
                metrics_row_ref["std_daily_return_gross_mid"] = std_gross_mid
                metrics_row_ref["mean_daily_net_return_exp_mid"] = mean_exp_mid
                metrics_row_ref["std_daily_net_return_exp_mid"] = std_exp_mid
                metrics_row_ref["mean_daily_net_return_lin_mid"] = mean_lin_mid
                metrics_row_ref["std_daily_net_return_lin_mid"] = std_lin_mid
                if "baseline" in meta_by_type and "prl" in meta_by_type:
                    base_meta = meta_by_type["baseline"]
                    prl_meta = meta_by_type["prl"]
                    if base_meta.get("data_manifest_hash") != prl_meta.get("data_manifest_hash"):
                        raise ValueError("DATA_MANIFEST_HASH_MISMATCH")
                    if base_meta.get("env_signature_hash") != prl_meta.get("env_signature_hash"):
                        raise ValueError("ENV_SIGNATURE_HASH_MISMATCH")

                include_baselines = False
                if window["name"] not in baseline_emitted_windows:
                    if model_type == "prl":
                        include_baselines = True
                    elif model_type == "baseline" and "prl" not in args.model_types:
                        include_baselines = True
                window_suffix = f"_{window['name']}" if multi_window else ""
                if include_baselines and run_baselines_flag:
                    baseline_emitted_windows.add(window["name"])
                    strategy_metrics, strategy_trace_df = eval_strategies_to_trace(
                        aligned_returns,
                        aligned_vol,
                        transaction_cost=env_cfg["c_tc"],
                        eval_id=eval_id,
                        run_id=baseline_strategy_run_id,
                        seed=seed,
                    )
                    strategy_trace_df["eval_window"] = window["name"]
                    strategy_trace_df = strategy_trace_df.merge(vz_df, on="date", how="left")
                    strategy_trace_df = compute_regime_labels(strategy_trace_df, thresholds)
                    if write_trace:
                        strategy_trace_path = reports_dir / f"trace_{baseline_strategy_run_id}{window_suffix}.parquet"
                        strategy_trace_df.iloc[::trace_stride].to_parquet(strategy_trace_path, index=False)
                    regime_rows.extend(summarize_regime_metrics(strategy_trace_df, period="test"))
                    for name, base_metrics in strategy_metrics.items():
                        metrics_rows.append(
                            {
                                "run_id": baseline_strategy_run_id,
                                "eval_id": eval_id,
                                "eval_window": window["name"],
                                "model_type": name,
                                "seed": seed,
                                "period": "test",
                                **base_metrics.to_dict(),
                            }
                        )
                    run_ids_this_session.append(baseline_strategy_run_id)

                trace_path = reports_dir / f"trace_{run_id}{window_suffix}.parquet"
                if write_trace:
                    trace_to_save = trace_df.iloc[::trace_stride].copy()
                    trace_path_to_write = trace_path
                    trace_to_save.to_parquet(trace_path_to_write, index=False)
                run_ids_this_session.append(run_id)

                thresholds_path = reports_dir / f"regime_thresholds_{run_id}{window_suffix}.json"
                thresholds_path.write_text(json.dumps(thresholds, indent=2))

                regime_rows.extend(summarize_regime_metrics(trace_df, period="test"))
                eval_info = {
                    "eval_start_date": thresholds["eval_start_date"],
                    "eval_end_date": thresholds["eval_end_date"],
                    "eval_num_days": thresholds["eval_num_days"],
                    "eval_window": window["name"],
                }
                evaluation_payload = {
                    **eval_info,
                    "trace_path": str(trace_path) if write_trace else None,
                    "regime_thresholds_path": str(thresholds_path),
                    "metrics_row_id": metrics_row_id,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                updates = {
                    **eval_info,
                    "evaluations": {window["name"]: evaluation_payload},
                }
                if not multi_window:
                    updates["evaluation"] = evaluation_payload
                _update_run_metadata(reports_dir / f"run_metadata_{run_id}.json", updates)

    write_metrics(reports_dir / "metrics.csv", metrics_rows)
    summary_rows = summarize_metrics(metrics_rows)
    write_summary(reports_dir / "summary.csv", summary_rows)
    write_regime_metrics(reports_dir / "regime_metrics.csv", regime_rows)
    summary_seed_rows = summarize_seed_stats(metrics_rows)
    write_summary(reports_dir / "summary_seed_stats.csv", summary_seed_rows)
    run_index_path = reports_dir / "run_index.json"
    unique_run_ids = list(dict.fromkeys(run_ids_this_session))
    run_index_path.write_text(
        json.dumps(
            {
                "exp_name": cfg.get("name") or cfg.get("exp_name") or Path(args.config).stem,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_path": args.config,
                "model_types": args.model_types,
                "seeds": list(seeds),
                "eval_windows": {
                    "train_start": dates.get("train_start"),
                    "train_end": dates.get("train_end"),
                    "test_start": dates.get("test_start"),
                    "test_end": dates.get("test_end"),
                    "windows": eval_windows,
                },
                "run_ids": unique_run_ids,
                "metrics_path": str(reports_dir / "metrics.csv"),
                "regime_metrics_path": str(reports_dir / "regime_metrics.csv"),
                "reports_dir": str(reports_dir),
                "traces_dir": str(traces_dir),
                "models_dir": str(models_dir),
                "logs_dir": str(logs_dir),
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    archived = _archive_reports(reports_dir, config_path=args.config, cfg=cfg)
    if archived:
        logging.info("Archived reports for exp_id=%s: %s", _archive_exp_id(args.config, cfg), archived)
    if write_step4 and step4_targets:
        from scripts import make_step4_report

        import sys

        for run_id in sorted(step4_targets):
            meta_path = reports_dir / f"run_metadata_{run_id}.json"
            if not meta_path.exists():
                continue
            if not write_trace:
                continue
            args_list = [
                "make_step4_report",
                "--run-id",
                run_id,
                "--metadata",
                str(meta_path),
                "--outputs-dir",
                str(output_root),
            ]
            prev_argv = sys.argv
            sys.argv = args_list
            try:
                make_step4_report.main()
            except Exception as exc:  # pragma: no cover - best-effort
                logging.warning("step4 report failed for %s: %s", run_id, exc)
            finally:
                sys.argv = prev_argv
    print("Completed run_all workflow.")


if __name__ == "__main__":
    main()
