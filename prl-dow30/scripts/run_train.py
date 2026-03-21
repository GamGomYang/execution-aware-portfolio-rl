import argparse
import logging
from pathlib import Path

import yaml

from prl.train import resolve_signal_configuration, run_training


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
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Base directory for reports/models/logs (default: config.output.root or outputs).",
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


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["config_path"] = args.config
    resolve_signal_configuration(cfg)
    data_cfg = cfg.get("data", {})
    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    offline_cfg = data_cfg.get("offline", False)
    if paper_mode and not require_cache_cfg:
        raise ValueError("paper_mode=true requires require_cache=true.")
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    offline_flag = args.offline or offline_cfg or paper_mode
    cache_only = paper_mode or require_cache_cfg or offline_cfg or args.offline
    output_root = _resolve_output_root(args.output_root, cfg)
    model_path = run_training(
        config=cfg,
        model_type=args.model_type,
        seed=args.seed,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=output_root / "models",
        reports_dir=output_root / "reports",
        logs_dir=output_root / "logs",
        force_refresh=data_cfg.get("force_refresh", True),
        offline=offline_flag,
        cache_only=cache_only,
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
