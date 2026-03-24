#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "buy_and_hold_equal_weight": "Buy-and-hold EW",
    "daily_rebalanced_equal_weight": "Daily-rebalanced EW",
    "inverse_vol_risk_parity": "Inverse-vol RP",
    "minimum_variance": "Minimum-variance",
    "mean_variance_long_only": "Mean-variance (long-only)",
}

STRATEGY_ORDER = {
    "buy_and_hold_equal_weight": 0,
    "daily_rebalanced_equal_weight": 1,
    "inverse_vol_risk_parity": 2,
    "minimum_variance": 3,
    "mean_variance_long_only": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validation-first paper tables for the control-eta rebuild.")
    parser.add_argument("--validation-root", type=str, required=True, help="Validation step6 root.")
    parser.add_argument("--selection-json", type=str, required=True, help="Validation eta selection JSON.")
    parser.add_argument("--final-root", type=str, default="", help="Final/test step6 root.")
    parser.add_argument("--baselines-root", type=str, default="", help="External baseline root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write tables into.")
    parser.add_argument("--baseline-eta", type=float, default=1.0, help="Immediate-execution reference eta.")
    parser.add_argument("--selected-eta", type=float, default=np.nan, help="Optional override for selected eta.")
    parser.add_argument("--selected-stats-csv", type=str, default="", help="Optional selected-vs-eta1 stats CSV.")
    parser.add_argument("--diagnostic-v2-csv", type=str, default="", help="Optional trace-based diagnostic v2 CSV.")
    return parser.parse_args()


def _load_step6_runs(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("kappa_*/*/seed_*/metrics.csv")) + sorted(root.glob("kappa_*/seed_*/metrics.csv"))
    rows: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["metrics_path"] = str(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["kappa"] = pd.to_numeric(out["kappa"], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
    out["eta"] = pd.to_numeric(out["eta"], errors="coerce")
    out["eta_requested"] = pd.to_numeric(out.get("eta_requested", out["eta"]), errors="coerce")
    out["pair_eta"] = out["eta_requested"].where(~out["eta_requested"].isna(), out["eta"])
    out["sharpe_net_lin"] = pd.to_numeric(out["sharpe_net_lin"], errors="coerce")
    out["cagr"] = pd.to_numeric(out["cagr"], errors="coerce")
    out["maxdd"] = pd.to_numeric(out["maxdd"], errors="coerce")
    out["avg_turnover_exec"] = pd.to_numeric(out["avg_turnover_exec"], errors="coerce")
    out["avg_turnover_target"] = pd.to_numeric(out.get("avg_turnover_target", np.nan), errors="coerce")
    out["tracking_error_l2_mean"] = pd.to_numeric(out.get("tracking_error_l2_mean", np.nan), errors="coerce")
    out["misalignment_gap_mean"] = pd.to_numeric(out.get("misalignment_gap_mean", np.nan), errors="coerce")
    out["collapse_flag_any"] = out.get("collapse_flag_any", False).astype(bool)
    return out


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
            headers = [str(column) for column in df.columns]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def _selected_eta(args_eta: float, selection_path: Path) -> float:
    if np.isfinite(args_eta):
        return float(args_eta)
    payload = json.loads(selection_path.read_text())
    value = payload.get("selected_eta")
    if value is None:
        raise ValueError(f"Selected eta missing from {selection_path}")
    return float(value)


def _build_validation_frontier(root: Path) -> pd.DataFrame:
    aggregate_path = root / "aggregate.csv"
    if not aggregate_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(aggregate_path)
    cols = [col for col in ["kappa", "eta", "median_sharpe", "iqr_sharpe", "median_turnover_exec", "collapse_rate"] if col in df.columns]
    return df[cols].sort_values(["kappa", "eta"], ascending=[True, False]).reset_index(drop=True)


def _build_validation_selection(selection_csv: Path) -> pd.DataFrame:
    if not selection_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(selection_csv)
    cols = [
        "eta",
        "n_positive_kappas",
        "n_pairs_vs_eta1",
        "score_mean_median_sharpe_pos_kappa",
        "score_mean_median_delta_sharpe_vs_eta1_pos_kappa",
        "median_turnover_exec_pos_kappa_mean",
        "qualifies",
        "selected",
    ]
    return df[[col for col in cols if col in df.columns]].sort_values(["selected", "qualifies", "eta"], ascending=[False, False, False]).reset_index(drop=True)


def _median_summary(grp: pd.DataFrame) -> dict[str, float]:
    return {
        "median_sharpe_net_lin": float(grp["sharpe_net_lin"].median()),
        "median_cagr": float(grp["cagr"].median()),
        "median_maxdd": float(grp["maxdd"].median()),
        "median_turnover_exec": float(grp["avg_turnover_exec"].median()),
    }


def _build_test_selected_vs_eta1(runs: pd.DataFrame, *, selected_eta: float, baseline_eta: float) -> pd.DataFrame:
    selected = runs[np.isclose(runs["pair_eta"], selected_eta, atol=1e-12)].copy()
    baseline = runs[np.isclose(runs["pair_eta"], baseline_eta, atol=1e-12)].copy()
    if selected.empty or baseline.empty:
        return pd.DataFrame()
    baseline_by_key = baseline.set_index(["kappa", "seed"])
    rows: list[dict[str, Any]] = []
    for kappa, grp in selected.groupby("kappa"):
        summary_selected = _median_summary(grp)
        base_grp = baseline[baseline["kappa"] == kappa]
        if base_grp.empty:
            continue
        summary_base = _median_summary(base_grp)
        deltas: list[dict[str, float]] = []
        for _, row in grp.iterrows():
            key = (float(row["kappa"]), int(row["seed"]))
            if key not in baseline_by_key.index:
                continue
            base = baseline_by_key.loc[key]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            deltas.append(
                {
                    "delta_sharpe": float(row["sharpe_net_lin"]) - float(base["sharpe_net_lin"]),
                    "delta_cagr": float(row["cagr"]) - float(base["cagr"]),
                    "delta_turnover_exec": float(row["avg_turnover_exec"]) - float(base["avg_turnover_exec"]),
                }
            )
        delta_df = pd.DataFrame(deltas) if deltas else pd.DataFrame()
        rows.append(
            {
                "kappa": float(kappa),
                "selected_eta": float(selected_eta),
                "baseline_eta": float(baseline_eta),
                "selected_median_sharpe_net_lin": summary_selected["median_sharpe_net_lin"],
                "baseline_median_sharpe_net_lin": summary_base["median_sharpe_net_lin"],
                "median_delta_sharpe_net_lin": float(delta_df["delta_sharpe"].median()) if not delta_df.empty else float("nan"),
                "win_rate_sharpe": float((delta_df["delta_sharpe"] > 0.0).mean()) if not delta_df.empty else float("nan"),
                "selected_median_cagr": summary_selected["median_cagr"],
                "baseline_median_cagr": summary_base["median_cagr"],
                "median_delta_cagr": float(delta_df["delta_cagr"].median()) if not delta_df.empty else float("nan"),
                "selected_median_turnover_exec": summary_selected["median_turnover_exec"],
                "baseline_median_turnover_exec": summary_base["median_turnover_exec"],
                "median_delta_turnover_exec": float(delta_df["delta_turnover_exec"].median()) if not delta_df.empty else float("nan"),
                "n_pairs": int(len(delta_df)),
            }
        )
    return pd.DataFrame(rows).sort_values("kappa").reset_index(drop=True)


def _build_selected_summary(runs: pd.DataFrame, *, selected_eta: float) -> pd.DataFrame:
    subset = runs[np.isclose(runs["pair_eta"], selected_eta, atol=1e-12)].copy()
    if subset.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for kappa, grp in subset.groupby("kappa"):
        row = _median_summary(grp)
        row.update(
            {
                "kappa": float(kappa),
                "selected_eta": float(selected_eta),
                "median_turnover_target": float(grp["avg_turnover_target"].median()),
                "median_turnover_ratio_target_over_exec": float(
                    (grp["avg_turnover_target"] / grp["avg_turnover_exec"].replace(0.0, np.nan)).median()
                ),
                "median_tracking_error_l2": float(grp["tracking_error_l2_mean"].median()),
                "median_misalignment_gap": float(grp["misalignment_gap_mean"].median()),
                "collapse_rate": float(grp["collapse_flag_any"].astype(bool).mean()),
                "n_seeds": int(grp["seed"].nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("kappa").reset_index(drop=True)


def _build_selected_vs_external(selected_summary: pd.DataFrame, baselines_root: Path) -> pd.DataFrame:
    aggregate_path = baselines_root / "aggregate.csv"
    if selected_summary.empty or not aggregate_path.exists():
        return pd.DataFrame()
    base = pd.read_csv(aggregate_path)
    base["kappa"] = pd.to_numeric(base["kappa"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for _, sel in selected_summary.iterrows():
        kappa = float(sel["kappa"])
        for _, row in base[base["kappa"] == kappa].iterrows():
            strategy = str(row["strategy"])
            rows.append(
                {
                    "kappa": kappa,
                    "selected_eta": float(sel["selected_eta"]),
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                    "selected_sharpe_net_lin": float(sel["median_sharpe_net_lin"]),
                    "baseline_sharpe_net_lin": float(row["sharpe_net_lin"]),
                    "delta_sharpe_net_lin": float(sel["median_sharpe_net_lin"]) - float(row["sharpe_net_lin"]),
                    "selected_cagr": float(sel["median_cagr"]),
                    "baseline_cagr": float(row["cagr"]),
                    "delta_cagr": float(sel["median_cagr"]) - float(row["cagr"]),
                    "selected_turnover_exec": float(sel["median_turnover_exec"]),
                    "baseline_turnover_exec": float(row["avg_turnover_exec"]),
                    "delta_turnover_exec": float(sel["median_turnover_exec"]) - float(row["avg_turnover_exec"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_strategy_rank"] = out["strategy"].map(lambda s: STRATEGY_ORDER.get(str(s), 999))
    return out.sort_values(["kappa", "_strategy_rank", "strategy"]).drop(columns=["_strategy_rank"]).reset_index(drop=True)


def _load_optional_csv(path_raw: str) -> pd.DataFrame:
    if not path_raw:
        return pd.DataFrame()
    path = Path(path_raw)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    validation_root = Path(args.validation_root)
    selection_json = Path(args.selection_json)
    final_root = Path(args.final_root) if args.final_root else None
    baselines_root = Path(args.baselines_root) if args.baselines_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_eta = _selected_eta(args.selected_eta, selection_json)
    validation_frontier = _build_validation_frontier(validation_root)
    validation_selection = _build_validation_selection(selection_json.with_name("validation_eta_selection.csv"))

    final_runs = _load_step6_runs(final_root) if final_root is not None and final_root.exists() else pd.DataFrame()
    selected_stats = _load_optional_csv(args.selected_stats_csv)
    diagnostic_v2 = _load_optional_csv(args.diagnostic_v2_csv)

    test_selected_vs_eta1 = selected_stats
    if test_selected_vs_eta1.empty and not final_runs.empty:
        test_selected_vs_eta1 = _build_test_selected_vs_eta1(
            final_runs,
            selected_eta=selected_eta,
            baseline_eta=float(args.baseline_eta),
        )

    diagnostic_selected = diagnostic_v2
    if diagnostic_selected.empty and not final_runs.empty:
        diagnostic_selected = _build_selected_summary(final_runs, selected_eta=selected_eta)
    test_selected_vs_external = (
        _build_selected_vs_external(diagnostic_selected, baselines_root)
        if baselines_root is not None and baselines_root.exists()
        else pd.DataFrame()
    )

    tables_dir = output_dir / "tables"
    _write_csv_md(
        validation_frontier,
        csv_path=tables_dir / "validation_frontier.csv",
        md_path=tables_dir / "validation_frontier.md",
        title="Validation Frontier",
    )
    _write_csv_md(
        validation_selection,
        csv_path=tables_dir / "validation_selection.csv",
        md_path=tables_dir / "validation_selection.md",
        title="Validation Selection",
    )
    _write_csv_md(
        test_selected_vs_eta1,
        csv_path=tables_dir / "test_selected_vs_eta1.csv",
        md_path=tables_dir / "test_selected_vs_eta1.md",
        title="Test Selected Eta Vs Immediate Execution",
    )
    _write_csv_md(
        test_selected_vs_external,
        csv_path=tables_dir / "test_selected_vs_external_baselines.csv",
        md_path=tables_dir / "test_selected_vs_external_baselines.md",
        title="Test Selected Eta Vs External Baselines",
    )
    _write_csv_md(
        diagnostic_selected,
        csv_path=tables_dir / "diagnostic_selected_eta.csv",
        md_path=tables_dir / "diagnostic_selected_eta.md",
        title="Diagnostic Selected Eta",
    )

    protocol_lines = [
        "# Validation-First Paper Protocol",
        "",
        f"- selected_eta: {selected_eta}",
        f"- baseline_eta: {float(args.baseline_eta)}",
        "- eta grid fixed a priori.",
        "- eta selected on validation only.",
        "- test used only for final held-out evaluation of the selected operating point.",
        "- heuristic baselines matched on window, kappa, annualization, rf, and executed-path metrics.",
        "",
        "## Table Layout",
        "",
        "- Validation table: frontier plus selection report.",
        "- Test table A: selected eta vs immediate-execution baseline with paired dispersion, bootstrap CI, sign test, and Wilcoxon reporting.",
        "- Test table B: selected eta vs external heuristic baselines.",
        "- Diagnostic table: turnover, tracking, and trace-based gap summaries at the selected eta.",
    ]
    (output_dir / "protocol_lock.md").write_text("\n".join(protocol_lines) + "\n")
    (output_dir / "protocol_lock.json").write_text(
        json.dumps(
            {
                "selected_eta": selected_eta,
                "baseline_eta": float(args.baseline_eta),
                "rules": {
                    "eta_grid_fixed_a_priori": True,
                    "validation_only_selection": True,
                    "test_not_used_for_selection": True,
                    "matched_external_baselines": True,
                },
            },
            indent=2,
        )
    )
    print(f"WROTE_TABLES={tables_dir}")


if __name__ == "__main__":
    main()
