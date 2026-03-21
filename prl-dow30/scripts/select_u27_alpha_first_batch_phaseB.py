#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.materialize_u27_alpha_first_batch_configs import candidate_key_from_tag, candidate_tag


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_TAG = "u27_eta082_alpha_ctrl_20k_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the promoted alpha-first winner from Phase B.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional Phase B summary CSV. If omitted, the latest matching report is used.",
    )
    parser.add_argument(
        "--summary-glob",
        type=str,
        default="outputs/reports/u27_eta082_phaseB_summary_*.csv",
        help="Glob used to resolve the latest summary when --summary-csv is omitted.",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default=DEFAULT_BASELINE_TAG,
        help="20k baseline/control tag reserved outside winner selection.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional JSON output path. Defaults to a timestamped file under outputs/reports.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Optional Markdown output path. Defaults to a timestamped file under outputs/reports.",
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
        raise FileNotFoundError(f"No summary CSV matched pattern: {glob_pattern}")
    return matches[0]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _base_tag(tag: str) -> str:
    raw = str(tag)
    return raw[:-7] if raw.endswith("_full10") else raw


def main() -> None:
    args = parse_args()
    summary_path = _resolve(args.summary_csv) if args.summary_csv else _latest_matching(args.summary_glob)
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_path}")

    df = df.copy()
    df["hard_pass"] = _as_bool(df["hard_pass"])
    df["soft_pass"] = _as_bool(df["soft_pass"])
    for col in ["k001_mean_delta_sharpe", "k001_main_median_sharpe", "k001_collapse_rate"]:
        if col in df.columns:
            df[col] = _as_float(df[col])
    df["base_tag_20k"] = df["tag"].map(_base_tag)

    rank_df = df.sort_values(
        ["hard_pass", "soft_pass", "k001_mean_delta_sharpe", "k001_main_median_sharpe"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    challengers = rank_df.loc[rank_df["base_tag_20k"] != args.baseline_tag].reset_index(drop=True)
    winner_row: dict[str, Any] | None = challengers.iloc[0].to_dict() if not challengers.empty else None
    winner_tag_20k = str(winner_row["base_tag_20k"]) if winner_row is not None else ""
    winner_key = candidate_key_from_tag(winner_tag_20k) if winner_tag_20k else ""
    winner_tag_100k = candidate_tag(winner_key, "100k_r1") if winner_key else ""

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = _resolve(args.out_json) if args.out_json else ROOT / "outputs" / "reports" / f"u27_alpha_first_batch_phaseB_decision_{ts}.json"
    out_md = _resolve(args.out_md) if args.out_md else ROOT / "outputs" / "reports" / f"u27_alpha_first_batch_phaseB_decision_{ts}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "baseline_tag_20k": args.baseline_tag,
        "winner": {
            "tag_20k": winner_tag_20k,
            "tag_100k": winner_tag_100k,
            "hard_pass": bool(winner_row["hard_pass"]) if winner_row is not None else False,
            "soft_pass": bool(winner_row["soft_pass"]) if winner_row is not None else False,
            "k001_mean_delta_sharpe": winner_row.get("k001_mean_delta_sharpe") if winner_row is not None else None,
            "k001_main_median_sharpe": winner_row.get("k001_main_median_sharpe") if winner_row is not None else None,
            "k001_collapse_rate": winner_row.get("k001_collapse_rate") if winner_row is not None else None,
        },
        "ranking": rank_df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines: list[str] = []
    lines.append("# U27 Alpha First Batch Phase B Decision")
    lines.append("")
    lines.append(f"- summary_csv: {summary_path.relative_to(ROOT)}")
    lines.append(f"- baseline_tag_20k: {args.baseline_tag}")
    lines.append(f"- winner_tag_20k: {winner_tag_20k or 'none'}")
    lines.append(f"- winner_tag_100k: {winner_tag_100k or 'none'}")
    if winner_row is not None:
        lines.append(f"- hard_pass: {bool(winner_row['hard_pass'])}")
        lines.append(f"- soft_pass: {bool(winner_row['soft_pass'])}")
        lines.append(f"- k001_mean_delta_sharpe: {winner_row.get('k001_mean_delta_sharpe')}")
        lines.append(f"- k001_main_median_sharpe: {winner_row.get('k001_main_median_sharpe')}")
        lines.append(f"- k001_collapse_rate: {winner_row.get('k001_collapse_rate')}")
    out_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2))
    if args.print_shell:
        print(f"PHASEB_SUMMARY_CSV={summary_path.relative_to(ROOT)}")
        print(f"WINNER_TAG_20K={winner_tag_20k}")
        print(f"WINNER_TAG_100K={winner_tag_100k}")
        print(f"DECISION_JSON={out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
