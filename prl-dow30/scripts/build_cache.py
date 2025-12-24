import argparse
import logging
from pathlib import Path

import yaml

from prl.data import load_market_data


def parse_args():
    parser = argparse.ArgumentParser(description="Build reproducibility cache using yfinance Adj Close.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="YAML config path.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    dates = cfg["dates"]
    source = data_cfg.get("source", "yfinance_only")
    if source != "yfinance_only":
        raise ValueError("build_cache.py supports only yfinance_only source.")

    market = load_market_data(
        start_date=dates["train_start"],
        end_date=dates["test_end"],
        raw_dir=data_cfg.get("raw_dir", "data/raw"),
        processed_dir=data_cfg.get("processed_dir", "data/processed"),
        tickers=None,
        force_refresh=True,
        session_opts=data_cfg.get("session_opts", None),
        min_history_days=data_cfg.get("min_history_days", 500),
        quality_params=data_cfg.get("quality_params", None),
        source=source,
        offline=False,
        require_cache=False,
        paper_mode=False,
        cache_only=False,
        ticker_substitutions=data_cfg.get("ticker_substitutions"),
    )
    print(f"Cache built with {market.prices.shape[1]} tickers and {len(market.prices)} rows.")


if __name__ == "__main__":
    main()
