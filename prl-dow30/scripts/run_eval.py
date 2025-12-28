import argparse
import csv
import logging
from pathlib import Path

import yaml

from prl.eval import load_model, run_backtest_episode
from prl.train import build_env_for_range, create_scheduler, prepare_market_and_features


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SAC/PRL models on 2022-2025 backtest.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model-type", choices=["baseline", "prl"], required=True, help="Model variant to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed identifier used during training.")
    parser.add_argument("--model-path", type=str, help="Optional explicit model path.")
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    return parser.parse_args()


def write_metrics(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_type",
        "seed",
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "avg_turnover",
        "sharpe",
        "max_drawdown",
        "steps",
    ]
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
        cache_only=cache_only,
        session_opts=session_opts,
    )

    env = build_env_for_range(
        market=market,
        features=features,
        start=dates["test_start"],
        end=dates["test_end"],
        window_size=env_cfg["L"],
        c_tc=env_cfg["c_tc"],
        seed=args.seed,
    )

    model_path = (
        Path(args.model_path)
        if args.model_path
        else Path("outputs/models") / f"{args.model_type}_seed{args.seed}_final.zip"
    )

    scheduler = None
    if args.model_type == "prl":
        num_assets = market.returns.shape[1]
        scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first or provide --model-path to an existing *_final.zip."
        )
    model = load_model(model_path, args.model_type, env, scheduler=scheduler)
    metrics = run_backtest_episode(model, env)
    row = {
        "model_type": args.model_type,
        "seed": args.seed,
        **metrics.to_dict(),
    }
    write_metrics(Path("outputs/reports/metrics.csv"), row)
    print(f"Backtest complete. Metrics saved to outputs/reports/metrics.csv")


if __name__ == "__main__":
    main()
