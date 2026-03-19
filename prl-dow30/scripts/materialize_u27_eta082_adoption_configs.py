#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize forward and operational configs for U27 eta082 adoption.")
    parser.add_argument(
        "--current-config",
        type=str,
        default="configs/prl_100k_signals_u27_eta082_current.yaml",
        help="Promoted current config path.",
    )
    parser.add_argument(
        "--step6-template",
        type=str,
        default="configs/step6_fixedeta_final_test_eta082_seed10.yaml",
        help="Step6 template config path.",
    )
    parser.add_argument(
        "--forward-config-out",
        type=str,
        default="configs/step6_fixedeta_forward_2026ytd_eta082_seed10.yaml",
        help="Forward Step6 config output path.",
    )
    parser.add_argument(
        "--operational-config-out",
        type=str,
        default="configs/prl_100k_signals_u27_eta082_operational_2026q1.yaml",
        help="Operational training config output path.",
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        default="outputs/reports/u27_eta082_adoption_materialization.json",
        help="Metadata JSON output path.",
    )
    parser.add_argument("--forward-start", type=str, default="2026-01-01", help="Forward OOS test start date.")
    parser.add_argument(
        "--operational-train-end",
        type=str,
        default="2025-12-31",
        help="Operational retrain train_end date.",
    )
    parser.add_argument(
        "--operational-output-root",
        type=str,
        default="outputs/operational_u27_eta082_2026q1",
        help="output.root to write into operational config.",
    )
    parser.add_argument(
        "--forward-output-root",
        type=str,
        default="outputs/step6_u27_eta082_forward_2026ytd",
        help="output.root to write into forward Step6 config.",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _read_yaml(path_str: str) -> dict[str, Any]:
    path = _resolve(path_str)
    return yaml.safe_load(path.read_text())


def _write_yaml(path_str: str, payload: dict[str, Any]) -> None:
    path = _resolve(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def main() -> None:
    args = parse_args()

    current_cfg = _read_yaml(args.current_config)
    step6_template = _read_yaml(args.step6_template)

    processed_dir = str((current_cfg.get("data", {}) or {}).get("processed_dir", "data/processed_u27"))
    returns_path = _resolve(processed_dir) / "returns.parquet"
    returns_df = pd.read_parquet(returns_path)
    cache_max_ts = pd.Timestamp(returns_df.index.max())
    cache_max_date = cache_max_ts.strftime("%Y-%m-%d")

    if cache_max_date < args.forward_start:
        raise ValueError(
            f"Cache max date {cache_max_date} is earlier than forward start {args.forward_start}. "
            "Refresh cache before forward OOS."
        )

    forward_cfg = step6_template
    forward_cfg["experiment_name"] = "step6_fixedeta_forward_2026ytd_eta082_seed10"
    forward_cfg.setdefault("dates", {})
    forward_cfg["dates"]["test_start"] = args.forward_start
    forward_cfg["dates"]["test_end"] = cache_max_date
    forward_cfg.setdefault("output", {})
    forward_cfg["output"]["root"] = args.forward_output_root

    operational_cfg = current_cfg
    operational_cfg.setdefault("dates", {})
    operational_cfg["dates"]["train_end"] = args.operational_train_end
    operational_cfg["dates"]["test_start"] = args.forward_start
    operational_cfg["dates"]["test_end"] = cache_max_date
    operational_cfg.setdefault("output", {})
    operational_cfg["output"]["root"] = args.operational_output_root

    _write_yaml(args.forward_config_out, forward_cfg)
    _write_yaml(args.operational_config_out, operational_cfg)

    meta = {
        "current_config": args.current_config,
        "step6_template": args.step6_template,
        "forward_config_out": args.forward_config_out,
        "operational_config_out": args.operational_config_out,
        "processed_dir": processed_dir,
        "returns_path": str(returns_path.relative_to(ROOT)),
        "cache_max_date": cache_max_date,
        "forward_start": args.forward_start,
        "operational_train_end": args.operational_train_end,
        "forward_output_root": args.forward_output_root,
        "operational_output_root": args.operational_output_root,
    }
    meta_path = _resolve(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
