#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready pack for frozen control eta-frontier runs.")
    parser.add_argument("--final-root", type=str, required=True, help="Final 2024~2025 eta frontier step6 root.")
    parser.add_argument("--forward-root", type=str, required=True, help="Forward 2026 YTD eta082 step6 root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Pack output directory.")
    parser.add_argument("--selected-eta", type=float, default=0.082, help="Selected eta used for forward sanity.")
    parser.add_argument("--baseline-eta", type=float, default=1.0, help="Eta used as paired baseline in tables.")
    parser.add_argument("--meta-json", type=str, default="", help="Optional materialization metadata JSON.")
    return parser.parse_args()


def _load_metrics(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("kappa_*/*/seed_*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv found under: {root}")
    rows: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["metrics_path"] = str(path)
        rows.append(df)
    if not rows:
        raise ValueError(f"All metrics.csv files were empty under: {root}")
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
    out["collapse_flag_any"] = out["collapse_flag_any"].astype(bool)
    return out


def _eta_key(value: float) -> str:
    return f"{float(value):.6g}"


def _summarize_frontier(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (kappa, eta), grp in runs.groupby(["kappa", "pair_eta"]):
        turnover = pd.to_numeric(grp["avg_turnover_exec"], errors="coerce")
        rows.append(
            {
                "kappa": float(kappa),
                "eta": float(eta),
                "n_seeds": int(grp["seed"].nunique()),
                "median_sharpe_net_lin": float(pd.to_numeric(grp["sharpe_net_lin"], errors="coerce").median()),
                "median_cagr": float(pd.to_numeric(grp["cagr"], errors="coerce").median()),
                "median_maxdd": float(pd.to_numeric(grp["maxdd"], errors="coerce").median()),
                "median_turnover_exec": float(turnover.median()),
                "avg_cost": float((float(kappa) * turnover).median()),
                "collapse_rate": float(grp["collapse_flag_any"].astype(bool).mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["kappa", "eta"], ascending=[True, False]).reset_index(drop=True)


def _paired_vs_eta1(runs: pd.DataFrame, baseline_eta: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kappa, grp in runs.groupby("kappa"):
        baseline = grp[np.isclose(grp["pair_eta"], baseline_eta, atol=1e-12)].copy()
        if baseline.empty:
            continue
        base_by_seed = baseline.set_index("seed")
        for eta, eta_grp in grp.groupby("pair_eta"):
            if np.isclose(float(eta), baseline_eta, atol=1e-12):
                continue
            deltas: list[dict[str, float]] = []
            for _, row in eta_grp.iterrows():
                seed = int(row["seed"])
                if seed not in base_by_seed.index:
                    continue
                base = base_by_seed.loc[seed]
                if isinstance(base, pd.DataFrame):
                    base = base.iloc[0]
                deltas.append(
                    {
                        "delta_sharpe": float(row["sharpe_net_lin"]) - float(base["sharpe_net_lin"]),
                        "delta_cagr": float(row["cagr"]) - float(base["cagr"]),
                        "delta_turnover_exec": float(row["avg_turnover_exec"]) - float(base["avg_turnover_exec"]),
                    }
                )
            if not deltas:
                continue
            delta_df = pd.DataFrame(deltas)
            rows.append(
                {
                    "kappa": float(kappa),
                    "eta": float(eta),
                    "baseline_eta": float(baseline_eta),
                    "n_pairs": int(len(delta_df)),
                    "median_delta_sharpe": float(delta_df["delta_sharpe"].median()),
                    "median_delta_cagr": float(delta_df["delta_cagr"].median()),
                    "median_delta_turnover_exec": float(delta_df["delta_turnover_exec"].median()),
                    "win_rate_sharpe": float((delta_df["delta_sharpe"] > 0.0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["kappa", "eta"], ascending=[True, False]).reset_index(drop=True)


def _selected_eta_summary(runs: pd.DataFrame, *, selected_eta: float, window_label: str) -> pd.DataFrame:
    subset = runs[np.isclose(runs["pair_eta"], selected_eta, atol=1e-12)].copy()
    if subset.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for kappa, grp in subset.groupby("kappa"):
        sharpe = pd.to_numeric(grp["sharpe_net_lin"], errors="coerce")
        rows.append(
            {
                "window": window_label,
                "kappa": float(kappa),
                "eta": float(selected_eta),
                "n_seeds": int(grp["seed"].nunique()),
                "median_sharpe_net_lin": float(sharpe.median()),
                "median_cagr": float(pd.to_numeric(grp["cagr"], errors="coerce").median()),
                "median_maxdd": float(pd.to_numeric(grp["maxdd"], errors="coerce").median()),
                "median_turnover_exec": float(pd.to_numeric(grp["avg_turnover_exec"], errors="coerce").median()),
                "n_positive_sharpe": int((sharpe > 0.0).sum()),
                "collapse_rate": float(grp["collapse_flag_any"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["window", "kappa"]).reset_index(drop=True)


def _write_md_summary(path: Path, title: str, df: pd.DataFrame) -> None:
    lines: list[str] = [f"# {title}", ""]
    if df.empty:
        lines.append("- no rows")
        path.write_text("\n".join(lines) + "\n")
        return
    lines.append(f"- rows: {len(df)}")
    lines.append("")
    try:
        lines.append(df.to_markdown(index=False))
    except ImportError:
        render_df = df.copy()
        for column in render_df.columns:
            series = render_df[column]
            if pd.api.types.is_float_dtype(series):
                render_df[column] = series.map(lambda value: f"{float(value):.6g}")
        headers = [str(column) for column in render_df.columns]
        rows = render_df.astype(str).values.tolist()
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_latex_rows(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    lines: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                parts.append(f"{float(value):.4f}")
            else:
                parts.append(str(value))
        lines.append(" & ".join(parts) + r" \\")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    final_root = Path(args.final_root)
    forward_root = Path(args.forward_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_runs = _load_metrics(final_root)
    forward_runs = _load_metrics(forward_root)

    final_frontier = _summarize_frontier(final_runs)
    paired_vs_eta1 = _paired_vs_eta1(final_runs, baseline_eta=float(args.baseline_eta))
    selected_summary = pd.concat(
        [
            _selected_eta_summary(final_runs, selected_eta=float(args.selected_eta), window_label="final_2024_2025"),
            _selected_eta_summary(forward_runs, selected_eta=float(args.selected_eta), window_label="forward_2026_ytd"),
        ],
        ignore_index=True,
    )

    final_frontier_csv = output_dir / "eta_frontier_final_summary.csv"
    paired_vs_eta1_csv = output_dir / "eta_frontier_paired_vs_eta1.csv"
    selected_summary_csv = output_dir / "selected_eta_final_forward_summary.csv"
    final_frontier.to_csv(final_frontier_csv, index=False)
    paired_vs_eta1.to_csv(paired_vs_eta1_csv, index=False)
    selected_summary.to_csv(selected_summary_csv, index=False)

    _write_md_summary(output_dir / "eta_frontier_final_summary.md", "Eta Frontier Final Summary", final_frontier)
    _write_md_summary(output_dir / "eta_frontier_paired_vs_eta1.md", "Eta Frontier Paired Vs Eta1", paired_vs_eta1)
    _write_md_summary(
        output_dir / "selected_eta_final_forward_summary.md",
        "Selected Eta Final Forward Summary",
        selected_summary,
    )

    _write_latex_rows(
        output_dir / "table_eta_frontier_rows.tex",
        final_frontier[["kappa", "eta", "median_sharpe_net_lin", "median_cagr", "median_turnover_exec", "avg_cost"]],
        ["kappa", "eta", "median_sharpe_net_lin", "median_cagr", "median_turnover_exec", "avg_cost"],
    )
    _write_latex_rows(
        output_dir / "table_paired_vs_eta1_rows.tex",
        paired_vs_eta1[["eta", "kappa", "median_delta_sharpe", "median_delta_cagr", "median_delta_turnover_exec", "win_rate_sharpe"]],
        ["eta", "kappa", "median_delta_sharpe", "median_delta_cagr", "median_delta_turnover_exec", "win_rate_sharpe"],
    )

    _copy_if_exists(final_root / "aggregate.csv", output_dir / "final_eta_frontier" / "aggregate.csv")
    _copy_if_exists(final_root / "paired_delta.csv", output_dir / "final_eta_frontier" / "paired_delta.csv")
    _copy_if_exists(final_root / "collapse_report.md", output_dir / "final_eta_frontier" / "collapse_report.md")
    _copy_if_exists(final_root / "fig_frontier.png", output_dir / "final_eta_frontier" / "fig_frontier.png")
    _copy_if_exists(final_root / "fig_misalignment.png", output_dir / "final_eta_frontier" / "fig_misalignment.png")

    _copy_if_exists(forward_root / "aggregate.csv", output_dir / "forward_eta082" / "aggregate.csv")
    _copy_if_exists(forward_root / "paired_delta.csv", output_dir / "forward_eta082" / "paired_delta.csv")
    _copy_if_exists(forward_root / "collapse_report.md", output_dir / "forward_eta082" / "collapse_report.md")
    _copy_if_exists(forward_root / "fig_frontier.png", output_dir / "forward_eta082" / "fig_frontier.png")
    _copy_if_exists(forward_root / "fig_misalignment.png", output_dir / "forward_eta082" / "fig_misalignment.png")

    manifest: dict[str, Any] = {
        "final_root": str(final_root),
        "forward_root": str(forward_root),
        "selected_eta": float(args.selected_eta),
        "baseline_eta": float(args.baseline_eta),
        "eta_grid": sorted({_eta_key(float(v)) for v in final_runs["pair_eta"].dropna().tolist()}, reverse=True),
        "final_frontier_csv": str(final_frontier_csv),
        "paired_vs_eta1_csv": str(paired_vs_eta1_csv),
        "selected_summary_csv": str(selected_summary_csv),
    }

    if args.meta_json:
        meta_path = Path(args.meta_json)
        manifest["meta_json"] = str(meta_path)
        _copy_if_exists(meta_path, output_dir / meta_path.name)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            for key in ["snapshot_config_out", "signal_snapshot_out", "final_config_out", "forward_config_out"]:
                value = meta.get(key)
                if value:
                    src = ROOT / value if not Path(value).is_absolute() else Path(value)
                    _copy_if_exists(src, output_dir / "configs" / Path(value).name)

    readme_lines = [
        "# U27 Control Eta Frontier Paper Pack",
        "",
        f"- selected_eta: {float(args.selected_eta)}",
        f"- baseline_eta: {float(args.baseline_eta)}",
        f"- final_root: {final_root}",
        f"- forward_root: {forward_root}",
        "",
        "## Main Artifacts",
        "",
        "- `eta_frontier_final_summary.csv`",
        "- `eta_frontier_paired_vs_eta1.csv`",
        "- `selected_eta_final_forward_summary.csv`",
        "- `table_eta_frontier_rows.tex`",
        "- `table_paired_vs_eta1_rows.tex`",
        "- `final_eta_frontier/fig_frontier.png`",
        "- `final_eta_frontier/fig_misalignment.png`",
        "",
        "## Notes",
        "",
        "- Final frontier summary is built on the 2024-01-01 ~ 2025-12-31 window.",
        "- Forward summary is built on the 2026-01-01 ~ cache_max_date window.",
        "- Paired delta table uses eta=1.0 as the baseline arm within each kappa.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n")

    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
