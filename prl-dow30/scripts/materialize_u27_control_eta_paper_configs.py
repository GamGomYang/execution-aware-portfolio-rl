#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze current control config for eta-frontier paper runs.")
    parser.add_argument(
        "--current-config",
        type=str,
        default="configs/prl_100k_signals_u27_eta082_current.yaml",
        help="Current incumbent control config.",
    )
    parser.add_argument(
        "--snapshot-config-out",
        type=str,
        required=True,
        help="Frozen control training config output path.",
    )
    parser.add_argument(
        "--signal-snapshot-out",
        type=str,
        required=True,
        help="Frozen selected_signals JSON output path.",
    )
    parser.add_argument(
        "--validation-config-out",
        type=str,
        default="",
        help="Validation 2022~2023 eta frontier eval config output path.",
    )
    parser.add_argument(
        "--final-config-out",
        type=str,
        default="",
        help="Final 2024~2025 eta frontier eval config output path.",
    )
    parser.add_argument(
        "--forward-config-out",
        type=str,
        default="",
        help="Forward 2026 YTD eval config output path.",
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        required=True,
        help="Materialization metadata JSON output path.",
    )
    parser.add_argument("--job-ts", type=str, default="", help="Optional externally supplied UTC timestamp.")
    parser.add_argument("--validation-start", type=str, default="2022-01-01")
    parser.add_argument("--validation-end", type=str, default="2023-12-31")
    parser.add_argument("--final-start", type=str, default="2024-01-01")
    parser.add_argument("--final-end", type=str, default="2025-12-31")
    parser.add_argument("--forward-start", type=str, default="2026-01-01")
    parser.add_argument("--train-output-root", type=str, default="")
    parser.add_argument("--validation-output-root", type=str, default="")
    parser.add_argument("--final-output-root", type=str, default="")
    parser.add_argument("--forward-output-root", type=str, default="")
    parser.add_argument(
        "--sac-total-timesteps",
        type=int,
        default=0,
        help="Optional override for sac.total_timesteps across emitted configs.",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def _resolve_config_relative_path(config_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _relpath(target: Path, start: Path) -> str:
    return os.path.relpath(target.resolve(), start.resolve())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _selected_signals(config_path: Path, cfg: dict[str, Any]) -> list[str]:
    signals_cfg = cfg.get("signals", {}) or {}
    signal_names = list(signals_cfg.get("signal_names", []) or [])
    if signal_names:
        return signal_names
    selected_path = _resolve_config_relative_path(config_path, signals_cfg.get("selected_signals_path"))
    if selected_path is None or not selected_path.exists():
        raise ValueError(f"Could not resolve selected_signals_path from config: {config_path}")
    payload = json.loads(selected_path.read_text())
    out = list(payload.get("selected_signals", [])) if isinstance(payload, dict) else list(payload)
    if not out:
        raise ValueError(f"Resolved selected signal set is empty: {selected_path}")
    return out


def main() -> None:
    args = parse_args()
    ts = args.job_ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    current_config_path = _resolve(args.current_config)
    snapshot_config_path = _resolve(args.snapshot_config_out)
    signal_snapshot_path = _resolve(args.signal_snapshot_out)
    validation_config_path = _resolve(args.validation_config_out) if args.validation_config_out else None
    final_config_path = _resolve(args.final_config_out) if args.final_config_out else None
    forward_config_path = _resolve(args.forward_config_out) if args.forward_config_out else None
    meta_path = _resolve(args.meta_out)

    current_cfg = _read_yaml(current_config_path)
    selected_signals = _selected_signals(current_config_path, current_cfg)

    processed_dir = str((current_cfg.get("data", {}) or {}).get("processed_dir", "data/processed_u27"))
    returns_path = _resolve(processed_dir) / "returns.parquet"
    returns_df = pd.read_parquet(returns_path)
    cache_max_date = pd.Timestamp(returns_df.index.max()).strftime("%Y-%m-%d")
    if forward_config_path is not None and cache_max_date < args.forward_start:
        raise ValueError(
            f"Cache max date {cache_max_date} is earlier than requested forward_start {args.forward_start}."
        )

    signal_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    signal_payload = {
        "selected_signals": selected_signals,
        "source_current_config": _display_path(current_config_path),
        "generated_at": ts,
        "tag": f"u27_eta082_control_paper_{ts}",
    }
    signal_snapshot_path.write_text(json.dumps(signal_payload, indent=2))

    frozen_cfg = copy.deepcopy(current_cfg)
    frozen_cfg.setdefault("signals", {})
    frozen_cfg["signals"]["enabled"] = True
    frozen_cfg["signals"]["signal_state"] = True
    frozen_cfg["signals"]["signal_names"] = list(selected_signals)
    frozen_cfg["signals"]["selection_policy"] = "paper_control_frozen"
    frozen_cfg["signals"]["allow_nonfixed_selection"] = False
    frozen_cfg["signals"]["selected_signals_path"] = _relpath(signal_snapshot_path, snapshot_config_path.parent)
    if args.sac_total_timesteps and args.sac_total_timesteps > 0:
        frozen_cfg.setdefault("sac", {})
        frozen_cfg["sac"]["total_timesteps"] = int(args.sac_total_timesteps)
    if args.train_output_root:
        frozen_cfg.setdefault("output", {})
        frozen_cfg["output"]["root"] = args.train_output_root

    if validation_config_path is not None:
        validation_cfg = copy.deepcopy(frozen_cfg)
        validation_cfg.setdefault("dates", {})
        validation_cfg["dates"]["test_start"] = args.validation_start
        validation_cfg["dates"]["test_end"] = args.validation_end
        if args.validation_output_root:
            validation_cfg.setdefault("output", {})
            validation_cfg["output"]["root"] = args.validation_output_root
        validation_cfg["signals"]["selected_signals_path"] = _relpath(signal_snapshot_path, validation_config_path.parent)
        _write_yaml(validation_config_path, validation_cfg)

    if final_config_path is not None:
        final_cfg = copy.deepcopy(frozen_cfg)
        final_cfg.setdefault("dates", {})
        final_cfg["dates"]["test_start"] = args.final_start
        final_cfg["dates"]["test_end"] = args.final_end
        if args.final_output_root:
            final_cfg.setdefault("output", {})
            final_cfg["output"]["root"] = args.final_output_root
        final_cfg["signals"]["selected_signals_path"] = _relpath(signal_snapshot_path, final_config_path.parent)
        _write_yaml(final_config_path, final_cfg)

    if forward_config_path is not None:
        forward_cfg = copy.deepcopy(frozen_cfg)
        forward_cfg.setdefault("dates", {})
        forward_cfg["dates"]["test_start"] = args.forward_start
        forward_cfg["dates"]["test_end"] = cache_max_date
        if args.forward_output_root:
            forward_cfg.setdefault("output", {})
            forward_cfg["output"]["root"] = args.forward_output_root
        forward_cfg["signals"]["selected_signals_path"] = _relpath(signal_snapshot_path, forward_config_path.parent)
        _write_yaml(forward_config_path, forward_cfg)

    _write_yaml(snapshot_config_path, frozen_cfg)

    payload = {
        "generated_at": ts,
        "current_config": _display_path(current_config_path),
        "snapshot_config_out": _display_path(snapshot_config_path),
        "signal_snapshot_out": _display_path(signal_snapshot_path),
        "validation_config_out": _display_path(validation_config_path) if validation_config_path is not None else "",
        "final_config_out": _display_path(final_config_path) if final_config_path is not None else "",
        "forward_config_out": _display_path(forward_config_path) if forward_config_path is not None else "",
        "processed_dir": processed_dir,
        "returns_path": _display_path(returns_path),
        "cache_max_date": cache_max_date,
        "validation_start": args.validation_start,
        "validation_end": args.validation_end,
        "final_start": args.final_start,
        "final_end": args.final_end,
        "forward_start": args.forward_start,
        "selected_signals": selected_signals,
        "sac_total_timesteps": int(args.sac_total_timesteps) if args.sac_total_timesteps > 0 else None,
        "train_output_root": args.train_output_root or frozen_cfg.get("output", {}).get("root"),
        "validation_output_root": args.validation_output_root,
        "final_output_root": args.final_output_root
        or (final_cfg.get("output", {}).get("root") if final_config_path is not None else None),
        "forward_output_root": args.forward_output_root
        or (forward_cfg.get("output", {}).get("root") if forward_config_path is not None else None),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
