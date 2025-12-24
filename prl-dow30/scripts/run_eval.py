import argparse
import csv
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
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dates = cfg["dates"]
    env_cfg = cfg["env"]
    prl_cfg = cfg.get("prl", {})
    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")

    market, features = prepare_market_and_features(
        start_date=dates["train_start"],
        end_date=dates["test_end"],
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        lv=env_cfg["Lv"],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        force_refresh=False,
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
        else Path("outputs/models") / f"{args.model_type}_seed{args.seed}.zip"
    )

    scheduler = None
    if args.model_type == "prl":
        num_assets = market.returns.shape[1]
        scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)

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
