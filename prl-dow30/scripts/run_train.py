import argparse
from pathlib import Path

import yaml

from prl.train import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline or PRL SAC on Dow30 data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "prl"],
        default="baseline",
        help="Training mode to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the run.")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download raw data.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    model_path = run_training(
        config=cfg,
        model_type=args.model_type,
        seed=args.seed,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir="outputs/models",
        force_refresh=args.force_refresh,
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
