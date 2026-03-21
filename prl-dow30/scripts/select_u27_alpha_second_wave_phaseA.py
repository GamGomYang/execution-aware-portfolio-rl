#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_TAG = "u27_eta082_alpha2_ctrl_20k_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select second-wave Phase A challengers for Phase B.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional Phase A summary CSV. If omitted, the latest matching report is used.",
    )
    parser.add_argument(
        "--summary-glob",
        type=str,
        default="outputs/reports/u27_alpha_second_wave_phaseA_summary_*.csv",
        help="Glob used to resolve the latest summary when --summary-csv is omitted.",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default=DEFAULT_BASELINE_TAG,
        help="Baseline/control tag reserved outside challenger selection.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of challengers to select.")
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


def _selection_warning(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    collapse = row.get("k001_collapse_rate")
    k001_positive = row.get("k001_positive")
    k001_delta = row.get("k001_mean_delta_sharpe")
    k0005_positive = row.get("k0005_positive")
    if pd.notna(collapse) and float(collapse) > 0.0:
        warnings.append("collapse_rate_gt_0")
    if pd.notna(k001_positive) and int(k001_positive) <= 1:
        warnings.append("k001_positive_le_1")
    if pd.notna(k001_delta) and float(k001_delta) <= 0.0:
        warnings.append("k001_mean_delta_nonpositive")
    if pd.notna(k0005_positive) and int(k0005_positive) < 4:
        warnings.append("k0005_not_full_sweep")
    return warnings


def main() -> None:
    args = parse_args()
    summary_path = _resolve(args.summary_csv) if args.summary_csv else _latest_matching(args.summary_glob)
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_path}")

    df = df.copy()
    df["hard_pass"] = _as_bool(df["hard_pass"])
    df["soft_pass"] = _as_bool(df["soft_pass"])
    for col in [
        "k001_positive",
        "k001_mean_delta_sharpe",
        "k001_median_delta_sharpe",
        "k001_collapse_rate",
        "k0005_positive",
        "k0005_mean_delta_sharpe",
        "k001_main_median_sharpe",
    ]:
        if col in df.columns:
            df[col] = _as_float(df[col])

    rank_df = df.sort_values(
        ["hard_pass", "k001_positive", "k001_mean_delta_sharpe", "k0005_positive", "k0005_mean_delta_sharpe"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    challengers = rank_df.loc[rank_df["tag"] != args.baseline_tag].reset_index(drop=True)
    selected = challengers.head(args.top_k).copy()
    rejected = challengers.iloc[args.top_k :].copy()

    selected_records: list[dict[str, Any]] = []
    for idx, row in enumerate(selected.to_dict(orient="records"), start=1):
        selected_records.append(
            {
                "rank": idx,
                "tag": row["tag"],
                "signals": row.get("signals", ""),
                "hard_pass": bool(row.get("hard_pass")),
                "soft_pass": bool(row.get("soft_pass")),
                "k001_positive": row.get("k001_positive"),
                "k001_mean_delta_sharpe": row.get("k001_mean_delta_sharpe"),
                "k001_main_median_sharpe": row.get("k001_main_median_sharpe"),
                "k0005_positive": row.get("k0005_positive"),
                "k001_negative_seeds": row.get("k001_negative_seeds"),
                "k001_collapse_rate": row.get("k001_collapse_rate"),
                "warnings": _selection_warning(row),
            }
        )

    rejected_records: list[dict[str, Any]] = []
    for row in rejected.to_dict(orient="records"):
        reasons = ["lower_rank_than_selected"]
        reasons.extend(_selection_warning(row))
        rejected_records.append(
            {
                "tag": row["tag"],
                "signals": row.get("signals", ""),
                "hard_pass": bool(row.get("hard_pass")),
                "soft_pass": bool(row.get("soft_pass")),
                "k001_positive": row.get("k001_positive"),
                "k001_mean_delta_sharpe": row.get("k001_mean_delta_sharpe"),
                "k001_main_median_sharpe": row.get("k001_main_median_sharpe"),
                "k0005_positive": row.get("k0005_positive"),
                "k001_collapse_rate": row.get("k001_collapse_rate"),
                "reasons": reasons,
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = (
        _resolve(args.out_json)
        if args.out_json
        else ROOT / "outputs" / "reports" / f"u27_alpha_second_wave_phaseA_decision_{ts}.json"
    )
    out_md = (
        _resolve(args.out_md)
        if args.out_md
        else ROOT / "outputs" / "reports" / f"u27_alpha_second_wave_phaseA_decision_{ts}.md"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "baseline_tag": args.baseline_tag,
        "top_k": args.top_k,
        "selected_candidates": selected_records,
        "rejected_candidates": rejected_records,
        "ranking": rank_df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines: list[str] = []
    lines.append("# U27 Alpha Second Wave Phase A Decision")
    lines.append("")
    lines.append(f"- summary_csv: {summary_path.relative_to(ROOT)}")
    lines.append(f"- baseline_tag: {args.baseline_tag}")
    lines.append(f"- selected_count: {len(selected_records)}")
    lines.append("")
    lines.append("## Selected Challengers")
    lines.append("")
    if selected_records:
        for item in selected_records:
            lines.append(
                f"- rank={item['rank']} tag={item['tag']} hard_pass={item['hard_pass']} "
                f"soft_pass={item['soft_pass']} k001_positive={item['k001_positive']} "
                f"k001_mean_delta_sharpe={item['k001_mean_delta_sharpe']} "
                f"k001_main_median_sharpe={item['k001_main_median_sharpe']} "
                f"k0005_positive={item['k0005_positive']} warnings={item['warnings']}"
            )
    else:
        lines.append("- no challengers selected")
    lines.append("")
    lines.append("## Rejected Challengers")
    lines.append("")
    if rejected_records:
        for item in rejected_records:
            lines.append(
                f"- tag={item['tag']} hard_pass={item['hard_pass']} soft_pass={item['soft_pass']} "
                f"k001_positive={item['k001_positive']} k001_mean_delta_sharpe={item['k001_mean_delta_sharpe']} "
                f"k001_main_median_sharpe={item['k001_main_median_sharpe']} "
                f"k0005_positive={item['k0005_positive']} reasons={item['reasons']}"
            )
    else:
        lines.append("- none")
    out_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2))
    if args.print_shell:
        print(f"BASELINE_TAG={args.baseline_tag}")
        print(f"SELECTED_TAGS={' '.join(item['tag'] for item in selected_records)}")
        print(f"PHASEA_SUMMARY_CSV={summary_path.relative_to(ROOT)}")
        print(f"DECISION_JSON={out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
