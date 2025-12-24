import argparse
import csv
import logging
from pathlib import Path

import yaml

from prl.eval import load_model, run_backtest_episode
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


def write_summary(path: Path, rows: list[dict]):
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
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    min_history_days = data_cfg.get("min_history_days", 500)
    quality_params = data_cfg.get("quality_params", None)
    source = data_cfg.get("source", "yfinance_only")
    require_cache = data_cfg.get("require_cache", False) or data_cfg.get("paper_mode", False)
    paper_mode = data_cfg.get("paper_mode", False)
    offline = args.offline or data_cfg.get("offline", False) or paper_mode or require_cache
    cache_only = paper_mode or require_cache
    require_cache = require_cache or offline
    session_opts = data_cfg.get("session_opts", None)

    market, features = prepare_market_and_features(
        start_date=dates["train_start"],
        end_date=dates["test_end"],
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        lv=env_cfg["Lv"],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        force_refresh=data_cfg.get("force_refresh", True),
        min_history_days=min_history_days,
        quality_params=quality_params,
        source=source,
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=session_opts,
        cache_only=cache_only,
    )

    seeds = args.seeds or cfg.get("seeds", [0, 1, 2])
    summary_rows = []
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
            )

            scheduler = None
            if model_type == "prl":
                num_assets = market.returns.shape[1]
                scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)

            model = load_model(model_path, model_type, env, scheduler=scheduler)
            metrics = run_backtest_episode(model, env)
            row = {
                "model_type": model_type,
                "seed": seed,
                "model_path": str(model_path),
                **metrics.to_dict(),
            }
            summary_rows.append(row)

    write_summary(Path("outputs/reports/summary.csv"), summary_rows)
    print("Completed run_all workflow.")


if __name__ == "__main__":
    main()
