"""CLI entry point for running the MPRL training loop."""

from __future__ import annotations

import argparse

from mprl.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MPRL training")
    parser.add_argument("--env", choices=["stub", "real"], default="stub", help="Environment type")
    parser.add_argument("--steps", type=int, default=600, help="Total training steps")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Data start date (real env)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="Data end date (real env)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to cache downloaded data")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs and decision traces")
    parser.add_argument("--no-cache", action="store_true", help="Disable cached market data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        steps=args.steps,
        env_type=args.env,
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
