#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trace-based misalignment diagnostics for the selected eta.")
    parser.add_argument("--final-root", type=str, required=True, help="Final/test step6 root.")
    parser.add_argument("--selection-json", type=str, required=True, help="Validation eta selection JSON.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write diagnostic artifacts into.")
    parser.add_argument("--selected-eta", type=float, default=np.nan, help="Optional selected eta override.")
    parser.add_argument(
        "--representative-kappa",
        type=float,
        default=0.001,
        help="Cost regime used to choose the representative selected-eta seed for figures.",
    )
    return parser.parse_args()


def _selected_eta(args_eta: float, selection_path: Path) -> float:
    if np.isfinite(args_eta):
        return float(args_eta)
    payload = json.loads(selection_path.read_text())
    value = payload.get("selected_eta")
    if value is None:
        raise ValueError(f"selected_eta missing from {selection_path}")
    return float(value)


def _load_runs(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("kappa_*/*/seed_*/metrics.csv")) + sorted(root.glob("kappa_*/seed_*/metrics.csv"))
    rows: list[dict] = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["metrics_path"] = str(path)
        row["trace_path"] = str(path.parent / "trace.parquet")
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {root}")

    out = pd.DataFrame(rows)
    for column in ["kappa", "seed", "eta", "eta_requested", "sharpe_net_lin", "cagr", "maxdd", "avg_turnover_exec", "avg_turnover_target"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["seed"] = out["seed"].astype(int)
    out["pair_eta"] = out["eta_requested"].where(~out["eta_requested"].isna(), out["eta"])
    return out


def _iqr(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def _to_markdown_fallback(df: pd.DataFrame) -> list[str]:
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return lines


def _write_csv_md(df: pd.DataFrame, *, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("- no rows")
    else:
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            lines.extend(_to_markdown_fallback(df))
    md_path.write_text("\n".join(lines) + "\n")


def _trace_metrics(trace_path: Path) -> dict[str, float]:
    trace = pd.read_parquet(trace_path)
    exec_ret = pd.to_numeric(trace["net_return_lin"], errors="coerce")
    tgt_ret = pd.to_numeric(trace["net_return_lin_target"], errors="coerce")
    ret_gap = exec_ret - tgt_ret

    turnover_exec = pd.to_numeric(trace["turnover_exec"], errors="coerce")
    turnover_target = pd.to_numeric(trace["turnover_target"], errors="coerce")
    tracking = pd.to_numeric(trace["tracking_error_l2"], errors="coerce")
    equity_exec = pd.to_numeric(trace["equity_net_lin"], errors="coerce")
    equity_target = pd.to_numeric(trace["equity_net_lin_target"], errors="coerce")

    exec_turn_mean = float(turnover_exec.mean())
    target_turn_mean = float(turnover_target.mean())
    return {
        "avg_turnover_exec": exec_turn_mean,
        "avg_turnover_target": target_turn_mean,
        "avg_turnover_ratio_target_over_exec": float(target_turn_mean / exec_turn_mean) if exec_turn_mean > 0 else float("nan"),
        "mean_abs_return_gap": float(ret_gap.abs().mean()),
        "final_equity_gap": float(abs(equity_exec.iloc[-1] - equity_target.iloc[-1])),
        "max_abs_daily_gap": float(ret_gap.abs().max()),
        "avg_tracking_error_l2": float(tracking.mean()),
        "return_gap_std": float(ret_gap.std(ddof=0)),
    }


def _build_selected_summary(runs: pd.DataFrame, *, selected_eta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = runs[np.isclose(runs["pair_eta"], selected_eta, atol=1e-12)].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    seed_rows: list[dict] = []
    for _, row in subset.iterrows():
        trace_path = Path(row["trace_path"])
        metrics = _trace_metrics(trace_path)
        seed_rows.append(
            {
                "kappa": float(row["kappa"]),
                "seed": int(row["seed"]),
                "selected_eta": float(selected_eta),
                "trace_path": str(trace_path),
                "sharpe_net_lin": float(row["sharpe_net_lin"]),
                "cagr": float(row["cagr"]),
                "maxdd": float(row["maxdd"]),
                **metrics,
            }
        )
    seed_df = pd.DataFrame(seed_rows).sort_values(["kappa", "seed"]).reset_index(drop=True)

    summary_rows: list[dict] = []
    for kappa, grp in seed_df.groupby("kappa"):
        summary_rows.append(
            {
                "kappa": float(kappa),
                "selected_eta": float(selected_eta),
                "n_seeds": int(len(grp)),
                "median_sharpe_net_lin": float(grp["sharpe_net_lin"].median()),
                "iqr_sharpe_net_lin": _iqr(grp["sharpe_net_lin"]),
                "median_cagr": float(grp["cagr"].median()),
                "iqr_cagr": _iqr(grp["cagr"]),
                "median_maxdd": float(grp["maxdd"].median()),
                "median_turnover_exec": float(grp["avg_turnover_exec"].median()),
                "median_turnover_target": float(grp["avg_turnover_target"].median()),
                "median_turnover_ratio_target_over_exec": float(grp["avg_turnover_ratio_target_over_exec"].median()),
                "median_tracking_error_l2": float(grp["avg_tracking_error_l2"].median()),
                "median_mean_abs_return_gap": float(grp["mean_abs_return_gap"].median()),
                "median_final_equity_gap": float(grp["final_equity_gap"].median()),
                "median_max_abs_daily_gap": float(grp["max_abs_daily_gap"].median()),
                "median_return_gap_std": float(grp["return_gap_std"].median()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("kappa").reset_index(drop=True)
    return summary_df, seed_df


def _pick_representative_seed(seed_df: pd.DataFrame, *, representative_kappa: float) -> dict:
    subset = seed_df[np.isclose(seed_df["kappa"], representative_kappa, atol=1e-15)].copy()
    if subset.empty:
        subset = seed_df.copy()
    if subset.empty:
        raise ValueError("No selected-eta seed diagnostics available to choose a representative seed.")
    median_sharpe = float(subset["sharpe_net_lin"].median())
    subset["median_distance"] = (subset["sharpe_net_lin"] - median_sharpe).abs()
    subset = subset.sort_values(["median_distance", "seed"]).reset_index(drop=True)
    row = subset.iloc[0]
    return {
        "selected_eta": float(row["selected_eta"]),
        "representative_kappa": float(row["kappa"]),
        "representative_seed": int(row["seed"]),
        "representative_trace_path": str(row["trace_path"]),
        "representative_sharpe_net_lin": float(row["sharpe_net_lin"]),
        "median_sharpe_net_lin_at_kappa": median_sharpe,
        "mean_abs_return_gap": float(row["mean_abs_return_gap"]),
        "final_equity_gap": float(row["final_equity_gap"]),
        "max_abs_daily_gap": float(row["max_abs_daily_gap"]),
    }


def main() -> None:
    args = parse_args()
    final_root = Path(args.final_root)
    selection_json = Path(args.selection_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_eta = _selected_eta(args.selected_eta, selection_json)
    runs = _load_runs(final_root)
    summary_df, seed_df = _build_selected_summary(runs, selected_eta=selected_eta)
    representative = _pick_representative_seed(seed_df, representative_kappa=float(args.representative_kappa))

    _write_csv_md(
        summary_df,
        csv_path=output_dir / "diagnostic_selected_eta_v2.csv",
        md_path=output_dir / "diagnostic_selected_eta_v2.md",
        title="Diagnostic Selected Eta V2",
    )
    _write_csv_md(
        seed_df,
        csv_path=output_dir / "diagnostic_selected_eta_v2_seedwise.csv",
        md_path=output_dir / "diagnostic_selected_eta_v2_seedwise.md",
        title="Diagnostic Selected Eta V2 Seedwise",
    )
    (output_dir / "representative_seed_metrics.json").write_text(json.dumps(representative, indent=2))

    print(f"WROTE_DIAGNOSTIC={output_dir / 'diagnostic_selected_eta_v2.csv'}")
    print(f"WROTE_SEEDWISE={output_dir / 'diagnostic_selected_eta_v2_seedwise.csv'}")
    print(f"WROTE_REPRESENTATIVE={output_dir / 'representative_seed_metrics.json'}")


if __name__ == "__main__":
    main()
