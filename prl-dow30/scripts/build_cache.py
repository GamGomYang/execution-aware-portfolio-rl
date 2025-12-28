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
    source = data_cfg.get("source", "yfinance_only")
    if source != "yfinance_only":
        raise ValueError("build_cache.py supports only yfinance_only source.")

    cfg_for_build = {
        **cfg,
        "data": {
            **data_cfg,
            "offline": False,
            "require_cache": False,
            "paper_mode": False,
        },
    }
    prices, _, manifest, _ = load_market_data(
        cfg_for_build,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )
    print(f"Cache built with {len(manifest.get('kept_tickers', []))} tickers and {len(prices)} rows.")


if __name__ == "__main__":
    main()
