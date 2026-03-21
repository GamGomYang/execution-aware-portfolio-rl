#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.materialize_u27_alpha_first_batch_configs import candidate_key_from_tag, candidate_tag


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_TAG_20K = "u27_eta082_alpha_ctrl_20k_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adopt the alpha-first winner into current/forward/operational configs."
    )
    parser.add_argument(
        "--winner-tag-20k",
        type=str,
        default="",
        help="Explicit winner 20k tag. If omitted, resolve from Phase B summary.",
    )
    parser.add_argument(
        "--winner-tag-100k",
        type=str,
        default="",
        help="Explicit winner 100k tag. If omitted, derive from winner 20k tag.",
    )
    parser.add_argument(
        "--phaseb-summary-csv",
        type=str,
        default="",
        help="Optional Phase B summary CSV. If omitted, latest matching report is used.",
    )
    parser.add_argument(
        "--phaseb-summary-glob",
        type=str,
        default="outputs/reports/u27_eta082_phaseB_summary_*.csv",
        help="Glob used to resolve the latest Phase B summary.",
    )
    parser.add_argument(
        "--baseline-tag-20k",
        type=str,
        default=DEFAULT_BASELINE_TAG_20K,
        help="Reserved control tag excluded from winner selection.",
    )
    parser.add_argument(
        "--current-config-in",
        type=str,
        default="configs/prl_100k_signals_u27_eta082_current.yaml",
        help="Current alias config to back up and replace.",
    )
    parser.add_argument(
        "--current-config-out",
        type=str,
        default="configs/prl_100k_signals_u27_eta082_current.yaml",
        help="Path to write the adopted current alias config.",
    )
    parser.add_argument(
        "--current-backup-out",
        type=str,
        default="",
        help="Optional backup path for the pre-adoption current config.",
    )
    parser.add_argument(
        "--snapshot-config-out",
        type=str,
        default="",
        help="Optional immutable snapshot config path.",
    )
    parser.add_argument(
        "--snapshot-signals-out",
        type=str,
        default="",
        help="Optional immutable signal snapshot JSON path.",
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
        help="Operational config output path.",
    )
    parser.add_argument(
        "--materialize-meta-out",
        type=str,
        default="outputs/reports/u27_eta082_adoption_materialization.json",
        help="Metadata JSON path consumed by forward/post-adoption pipelines.",
    )
    parser.add_argument(
        "--adoption-meta-out",
        type=str,
        default="",
        help="Extended adoption metadata JSON path.",
    )
    parser.add_argument(
        "--adoption-md-out",
        type=str,
        default="",
        help="Extended adoption markdown path.",
    )
    parser.add_argument(
        "--step6-template",
        type=str,
        default="configs/step6_fixedeta_final_test_eta082_seed10.yaml",
        help="Forward materialization template.",
    )
    parser.add_argument("--forward-start", type=str, default="2026-01-01")
    parser.add_argument("--operational-train-end", type=str, default="2025-12-31")
    parser.add_argument(
        "--forward-output-root",
        type=str,
        default="outputs/step6_u27_eta082_forward_2026ytd",
    )
    parser.add_argument(
        "--operational-output-root",
        type=str,
        default="outputs/operational_u27_eta082_2026q1",
    )
    parser.add_argument(
        "--print-shell",
        action="store_true",
        help="Print shell-friendly KEY=VALUE lines for wrapper scripts.",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _latest_matching(glob_pattern: str) -> Path:
    matches = sorted(ROOT.glob(glob_pattern), key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No Phase B summary matched pattern: {glob_pattern}")
    return matches[0]


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


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _base_tag_20k(tag: str) -> str:
    raw = str(tag)
    return raw[:-7] if raw.endswith("_full10") else raw


def _resolve_winner_tags(args: argparse.Namespace) -> tuple[str, str, Path]:
    summary_path = _resolve(args.phaseb_summary_csv) if args.phaseb_summary_csv else _latest_matching(args.phaseb_summary_glob)

    winner_tag_20k = args.winner_tag_20k.strip()
    winner_tag_100k = args.winner_tag_100k.strip()
    if winner_tag_100k and not winner_tag_20k:
        winner_tag_20k = winner_tag_100k.replace("_100k_r1", "_20k_r1")

    if not winner_tag_20k:
        df = pd.read_csv(summary_path)
        if df.empty:
            raise ValueError(f"Phase B summary is empty: {summary_path}")
        df = df.copy()
        df["hard_pass"] = _as_bool(df["hard_pass"])
        df["soft_pass"] = _as_bool(df["soft_pass"])
        for col in ["k001_mean_delta_sharpe", "k001_main_median_sharpe", "k001_collapse_rate"]:
            if col in df.columns:
                df[col] = _as_float(df[col])
        df["base_tag_20k"] = df["tag"].map(_base_tag_20k)
        rank_df = df.sort_values(
            ["hard_pass", "soft_pass", "k001_mean_delta_sharpe", "k001_main_median_sharpe"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        challengers = rank_df.loc[rank_df["base_tag_20k"] != args.baseline_tag_20k].reset_index(drop=True)
        if challengers.empty:
            raise ValueError("No challenger winner resolved from Phase B summary.")
        winner_tag_20k = str(challengers.iloc[0]["base_tag_20k"])

    if not winner_tag_100k:
        winner_key = candidate_key_from_tag(winner_tag_20k)
        winner_tag_100k = candidate_tag(winner_key, "100k_r1")

    return winner_tag_20k, winner_tag_100k, summary_path


def _snapshot_defaults(winner_key: str, ts: str) -> tuple[Path, Path, Path]:
    backup = ROOT / "configs" / "backups" / f"prl_100k_signals_u27_eta082_current_pre_{winner_key}_{ts}.yaml"
    snapshot_cfg = ROOT / "configs" / "snapshots" / f"prl_100k_signals_u27_eta082_current_{winner_key}_{ts}.yaml"
    snapshot_signals = ROOT / "configs" / "signal_sets" / "adopted" / f"u27_eta082_current_{winner_key}_{ts}.json"
    return backup, snapshot_cfg, snapshot_signals


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    winner_tag_20k, winner_tag_100k, summary_path = _resolve_winner_tags(args)
    winner_key = candidate_key_from_tag(winner_tag_20k)

    current_in_path = _resolve(args.current_config_in)
    current_out_path = _resolve(args.current_config_out)
    winner_config_path = ROOT / "configs" / "exp" / f"{winner_tag_100k}.yaml"
    if not winner_config_path.exists():
        raise FileNotFoundError(f"Winner config not found: {winner_config_path}")

    default_backup, default_snapshot_cfg, default_snapshot_signals = _snapshot_defaults(winner_key, ts)
    backup_path = _resolve(args.current_backup_out) if args.current_backup_out else default_backup
    snapshot_config_path = _resolve(args.snapshot_config_out) if args.snapshot_config_out else default_snapshot_cfg
    snapshot_signals_path = _resolve(args.snapshot_signals_out) if args.snapshot_signals_out else default_snapshot_signals
    adoption_meta_path = (
        _resolve(args.adoption_meta_out)
        if args.adoption_meta_out
        else ROOT / "outputs" / "reports" / f"u27_alpha_first_batch_current_adoption_{ts}.json"
    )
    adoption_md_path = (
        _resolve(args.adoption_md_out)
        if args.adoption_md_out
        else ROOT / "outputs" / "reports" / f"u27_alpha_first_batch_current_adoption_{ts}.md"
    )
    materialize_meta_path = _resolve(args.materialize_meta_out)

    current_cfg = _read_yaml(current_in_path)
    winner_cfg = _read_yaml(winner_config_path)

    winner_signals = list((winner_cfg.get("signals", {}) or {}).get("signal_names", []))
    if not winner_signals:
        selected_path = _resolve_config_relative_path(
            winner_config_path,
            (winner_cfg.get("signals", {}) or {}).get("selected_signals_path"),
        )
        if selected_path is None or not selected_path.exists():
            raise ValueError(f"Winner config has no signal_names and no valid selected_signals_path: {winner_config_path}")
        payload = json.loads(selected_path.read_text())
        winner_signals = list(payload.get("selected_signals", [])) if isinstance(payload, dict) else list(payload)
    if not winner_signals:
        raise ValueError(f"Winner signal set is empty: {winner_config_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(current_in_path.read_text())

    snapshot_signals_path.parent.mkdir(parents=True, exist_ok=True)
    signal_payload = {
        "selected_signals": winner_signals,
        "winner_tag_20k": winner_tag_20k,
        "winner_tag_100k": winner_tag_100k,
        "winner_config": str(winner_config_path.relative_to(ROOT)),
        "generated_at": ts,
    }
    snapshot_signals_path.write_text(json.dumps(signal_payload, indent=2))

    adopted_cfg = copy.deepcopy(current_cfg)
    adopted_cfg["signals"] = copy.deepcopy(winner_cfg.get("signals", {}) or {})
    adopted_cfg["signals"]["enabled"] = True
    adopted_cfg["signals"]["signal_state"] = True
    adopted_cfg["signals"]["signal_names"] = list(winner_signals)
    adopted_cfg["signals"]["selection_policy"] = "alpha_research"
    adopted_cfg["signals"]["allow_nonfixed_selection"] = True
    adopted_cfg["env"] = copy.deepcopy(winner_cfg.get("env", {}) or {})
    adopted_cfg["prl"] = copy.deepcopy(winner_cfg.get("prl", {}) or {})
    adopted_cfg["sac"] = copy.deepcopy(winner_cfg.get("sac", {}) or {})
    if winner_cfg.get("seeds"):
        adopted_cfg["seeds"] = copy.deepcopy(winner_cfg["seeds"])
    adopted_cfg["mode"] = current_cfg.get("mode", "paper")

    snapshot_cfg = copy.deepcopy(adopted_cfg)
    snapshot_cfg["signals"]["selected_signals_path"] = _relpath(snapshot_signals_path, snapshot_config_path.parent)
    current_cfg_out = copy.deepcopy(adopted_cfg)
    current_cfg_out["signals"]["selected_signals_path"] = _relpath(snapshot_signals_path, current_out_path.parent)

    _write_yaml(snapshot_config_path, snapshot_cfg)
    _write_yaml(current_out_path, current_cfg_out)

    materialize_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_u27_eta082_adoption_configs.py"),
        "--current-config",
        str(current_out_path.relative_to(ROOT)),
        "--step6-template",
        args.step6_template,
        "--forward-config-out",
        args.forward_config_out,
        "--operational-config-out",
        args.operational_config_out,
        "--meta-out",
        args.materialize_meta_out,
        "--forward-start",
        args.forward_start,
        "--operational-train-end",
        args.operational_train_end,
        "--forward-output-root",
        args.forward_output_root,
        "--operational-output-root",
        args.operational_output_root,
    ]
    subprocess.run(materialize_cmd, cwd=ROOT, check=True, text=True)

    payload = {
        "phaseb_summary_csv": str(summary_path.relative_to(ROOT)),
        "winner_tag_20k": winner_tag_20k,
        "winner_tag_100k": winner_tag_100k,
        "winner_config": str(winner_config_path.relative_to(ROOT)),
        "model_root": f"outputs/modelswap_{winner_tag_100k}",
        "current_config_in": str(current_in_path.relative_to(ROOT)),
        "current_config_out": str(current_out_path.relative_to(ROOT)),
        "current_config_backup": str(backup_path.relative_to(ROOT)),
        "snapshot_config": str(snapshot_config_path.relative_to(ROOT)),
        "snapshot_signals": str(snapshot_signals_path.relative_to(ROOT)),
        "forward_config_out": args.forward_config_out,
        "operational_config_out": args.operational_config_out,
        "materialize_meta_out": str(materialize_meta_path.relative_to(ROOT)),
        "winner_signals": winner_signals,
    }
    adoption_meta_path.parent.mkdir(parents=True, exist_ok=True)
    adoption_meta_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# U27 Alpha First Batch Current Adoption",
        "",
        f"- phaseb_summary_csv: {payload['phaseb_summary_csv']}",
        f"- winner_tag_20k: {winner_tag_20k}",
        f"- winner_tag_100k: {winner_tag_100k}",
        f"- winner_config: {payload['winner_config']}",
        f"- model_root: {payload['model_root']}",
        f"- current_config_backup: {payload['current_config_backup']}",
        f"- current_config_out: {payload['current_config_out']}",
        f"- snapshot_config: {payload['snapshot_config']}",
        f"- snapshot_signals: {payload['snapshot_signals']}",
        f"- forward_config_out: {payload['forward_config_out']}",
        f"- operational_config_out: {payload['operational_config_out']}",
        f"- winner_signals: {', '.join(winner_signals)}",
    ]
    adoption_md_path.parent.mkdir(parents=True, exist_ok=True)
    adoption_md_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2))
    if args.print_shell:
        print(f"PHASEB_SUMMARY_CSV={summary_path.relative_to(ROOT)}")
        print(f"WINNER_TAG_20K={winner_tag_20k}")
        print(f"WINNER_TAG_100K={winner_tag_100k}")
        print(f"MODEL_ROOT=outputs/modelswap_{winner_tag_100k}")
        print(f"CURRENT_CONFIG={current_out_path.relative_to(ROOT)}")
        print(f"CURRENT_CONFIG_BACKUP={backup_path.relative_to(ROOT)}")
        print(f"CURRENT_SNAPSHOT_CONFIG={snapshot_config_path.relative_to(ROOT)}")
        print(f"CURRENT_SNAPSHOT_SIGNALS={snapshot_signals_path.relative_to(ROOT)}")
        print(f"FORWARD_CONFIG={args.forward_config_out}")
        print(f"OPERATIONAL_CONFIG={args.operational_config_out}")
        print(f"MATERIALIZE_META={materialize_meta_path.relative_to(ROOT)}")
        print(f"ADOPTION_META={adoption_meta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
