import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import yaml

from prl.data import load_market_data


def parse_args():
    parser = argparse.ArgumentParser(description="Build reproducibility cache using yfinance Adj Close.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="YAML config path.")
    parser.add_argument(
        "--end-date",
        type=str,
        help="Optional override for dates.test_end when refreshing cache. Useful for forward OOS cache updates.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "yfinance_only")
    if source != "yfinance_only":
        raise ValueError("build_cache.py supports only yfinance_only source.")

    dates_cfg = dict(cfg.get("dates", {}) or {})
    if args.end_date:
        dates_cfg["test_end"] = str(args.end_date)
    elif "test_end" in dates_cfg:
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if str(dates_cfg["test_end"]) < today_utc:
            dates_cfg["test_end"] = today_utc

    cfg_for_build = {
        **cfg,
        "dates": dates_cfg,
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
