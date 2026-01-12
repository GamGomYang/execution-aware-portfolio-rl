import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prl.analysis import compute_regime_metrics
from prl.baselines import buy_and_hold_equal_weight, daily_rebalanced_equal_weight, inverse_vol_risk_parity
from prl.eval import load_model, run_backtest_timeseries
from prl.metrics import compute_portfolio_metrics
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
        "model_type",
        "seed",
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


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
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
    kept_tickers = market.manifest.get("kept_tickers") if market.manifest else None

    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required for training/evaluation.")

    seeds = args.seeds or cfg.get("seeds", [0, 1, 2])
    metrics_rows = []
    regime_rows = []
    rl_timeseries = []
    for model_type in args.model_types:
        for seed in seeds:
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

            model = load_model(model_path, model_type, env, scheduler=scheduler)
            metrics, ts = run_backtest_timeseries(model, env)
            metrics_rows.append(
                {
                    "model_type": model_type,
                    "seed": seed,
                    **metrics.to_dict(),
                }
            )
            if seed == 0:
                rl_timeseries.append((model_type, ts))

    test_returns = market.returns.loc[dates["test_start"] : dates["test_end"]]
    test_vol = features.volatility.loc[dates["test_start"] : dates["test_end"]]
    if kept_tickers:
        test_returns = test_returns[kept_tickers]
        test_vol = test_vol[kept_tickers]
    vol_clean = test_vol.dropna()
    idx = test_returns.index.intersection(vol_clean.index)
    test_returns = test_returns.loc[idx]
    test_vol = vol_clean.loc[idx]
    lookback = env_cfg["L"]

    baselines = [
        buy_and_hold_equal_weight(returns=test_returns, volatility=test_vol, lookback=lookback),
        daily_rebalanced_equal_weight(returns=test_returns, volatility=test_vol, lookback=lookback),
        inverse_vol_risk_parity(returns=test_returns, volatility=test_vol, lookback=lookback),
    ]
    for ts in baselines:
        metrics = compute_portfolio_metrics(ts.portfolio_return, ts.turnover)
        metrics_rows.append(
            {
                "model_type": ts.model_type,
                "seed": -1,
                **metrics.to_dict(),
            }
        )

    for model_type, ts in rl_timeseries:
        regime_rows.append(
            compute_regime_metrics(
                model_type=model_type,
                dates=pd.DatetimeIndex(ts["dates"]),
                portfolio_return=np.array(ts["portfolio_return"], dtype=float),
                turnover=np.array(ts["turnover"], dtype=float),
                vol_portfolio=np.array(ts["vol_portfolio"], dtype=float),
            )
        )
    for ts in baselines:
        if ts.vol_portfolio is None:
            continue
        regime_rows.append(
            compute_regime_metrics(
                model_type=ts.model_type,
                dates=ts.dates,
                portfolio_return=ts.portfolio_return,
                turnover=ts.turnover,
                vol_portfolio=ts.vol_portfolio,
            )
        )

    reports_dir = Path("outputs/reports")
    write_metrics(reports_dir / "metrics.csv", metrics_rows)
    summary_rows = summarize_metrics(metrics_rows)
    write_summary(reports_dir / "summary.csv", summary_rows)
    if regime_rows:
        regime_df = pd.concat(regime_rows, ignore_index=True)
        regime_df.to_csv(reports_dir / "regime_metrics.csv", index=False)
    print("Completed run_all workflow.")


if __name__ == "__main__":
    main()
