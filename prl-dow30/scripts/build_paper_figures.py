#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_STYLE = {
    0.0: {"color": "#1f4e79", "label": r"$\kappa=0$"},
    0.0005: {"color": "#c55a11", "label": r"$\kappa=5\times10^{-4}$"},
    0.001: {"color": "#5b8a3c", "label": r"$\kappa=10^{-3}$"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-oriented figures for the validation-first rebuild.")
    parser.add_argument("--validation-root", type=str, required=True, help="Validation step6 root.")
    parser.add_argument("--selection-json", type=str, required=True, help="Validation eta selection JSON.")
    parser.add_argument("--final-root", type=str, required=True, help="Final/test step6 root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write paper figures into.")
    parser.add_argument("--representative-json", type=str, default="", help="Representative seed metadata JSON.")
    parser.add_argument("--seedwise-stats-csv", type=str, default="", help="Seedwise selected-vs-eta1 deltas CSV.")
    parser.add_argument("--selected-eta", type=float, default=np.nan, help="Optional selected eta override.")
    parser.add_argument("--legacy-dir", type=str, default="", help="Optional directory to also write legacy figure filenames.")
    return parser.parse_args()


def _selected_eta(args_eta: float, selection_path: Path) -> float:
    if np.isfinite(args_eta):
        return float(args_eta)
    payload = json.loads(selection_path.read_text())
    value = payload.get("selected_eta")
    if value is None:
        raise ValueError(f"selected_eta missing from {selection_path}")
    return float(value)


def _load_representative_path(representative_json: Path) -> tuple[dict, Path]:
    payload = json.loads(representative_json.read_text())
    path = Path(payload["representative_trace_path"])
    if not path.exists():
        raise FileNotFoundError(f"Representative trace not found: {path}")
    return payload, path


def _build_selected_trace_figure(trace_path: Path, metadata: dict, output_path: Path) -> None:
    trace = pd.read_parquet(trace_path).copy()
    trace["date"] = pd.to_datetime(trace["date"])

    turnover_exec_cum = pd.to_numeric(trace["turnover_exec"], errors="coerce").fillna(0.0).cumsum()
    turnover_target_cum = pd.to_numeric(trace["turnover_target"], errors="coerce").fillna(0.0).cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, constrained_layout=True)

    axes[0].plot(trace["date"], trace["equity_net_lin"], color="#1f4e79", linewidth=2.2, label="Executed equity")
    axes[0].plot(
        trace["date"],
        trace["equity_net_lin_target"],
        color="#c55a11",
        linewidth=1.9,
        linestyle="--",
        label="Hypothetical target equity",
    )
    axes[0].set_ylabel("Net equity")
    axes[0].set_title(
        "Selected operating point on held-out test: "
        rf"$\eta={metadata['selected_eta']}$, $\kappa={metadata['representative_kappa']}$, seed={metadata['representative_seed']}"
    )
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.2, linewidth=0.5)

    axes[1].plot(trace["date"], turnover_exec_cum, color="#1f4e79", linewidth=2.2, label="Cumulative executed turnover")
    axes[1].plot(
        trace["date"],
        turnover_target_cum,
        color="#c55a11",
        linewidth=1.9,
        linestyle="--",
        label="Cumulative target turnover",
    )
    axes[1].set_ylabel("Cumulative turnover")
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(alpha=0.2, linewidth=0.5)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_validation_frontier_figure(validation_root: Path, *, selected_eta: float, output_path: Path) -> None:
    aggregate = pd.read_csv(validation_root / "aggregate.csv")
    aggregate["kappa"] = pd.to_numeric(aggregate["kappa"], errors="coerce")
    aggregate["eta"] = pd.to_numeric(aggregate["eta"], errors="coerce")
    aggregate["median_sharpe"] = pd.to_numeric(aggregate["median_sharpe"], errors="coerce")
    aggregate["iqr_sharpe"] = pd.to_numeric(aggregate["iqr_sharpe"], errors="coerce")
    aggregate["median_turnover_exec"] = pd.to_numeric(aggregate["median_turnover_exec"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8.8, 6.1), constrained_layout=True)

    selected_legend_drawn = False

    for kappa, grp in aggregate.groupby("kappa"):
        style = FIGURE_STYLE.get(float(kappa), {"color": "#444444", "label": f"kappa={kappa:g}"})
        grp = grp.sort_values("median_turnover_exec", ascending=False)
        yerr = grp["iqr_sharpe"].fillna(0.0) / 2.0
        ax.errorbar(
            grp["median_turnover_exec"],
            grp["median_sharpe"],
            yerr=yerr,
            color=style["color"],
            marker="o",
            markersize=5.5,
            linewidth=1.6,
            capsize=2.5,
            label=style["label"],
        )
        selected_rows = grp[np.isclose(grp["eta"], selected_eta, atol=1e-12)].copy()
        if not selected_rows.empty:
            ax.scatter(
                selected_rows["median_turnover_exec"],
                selected_rows["median_sharpe"],
                s=105,
                facecolors="none",
                edgecolors="#111111",
                linewidths=1.4,
                zorder=5,
                label=rf"selected $\eta={selected_eta:g}$" if not selected_legend_drawn else None,
            )
            selected_legend_drawn = True
        for _, row in grp.iterrows():
            alpha = 0.95 if np.isclose(row["eta"], selected_eta, atol=1e-12) else 0.55
            weight = "bold" if np.isclose(row["eta"], selected_eta, atol=1e-12) else "normal"
            ax.annotate(
                f"{row['eta']:g}",
                (row["median_turnover_exec"], row["median_sharpe"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8.2,
                color=style["color"],
                alpha=alpha,
                fontweight=weight,
            )

    ax.set_xlabel("Average executed turnover")
    ax.set_ylabel("Validation net Sharpe")
    ax.set_title("Validation frontier with selected operating point highlighted")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_seed_scatter(seedwise_csv: Path, output_path: Path) -> None:
    seedwise = pd.read_csv(seedwise_csv)
    seedwise["kappa"] = pd.to_numeric(seedwise["kappa"], errors="coerce")
    seedwise["seed"] = pd.to_numeric(seedwise["seed"], errors="coerce").astype(int)
    seedwise["delta_sharpe_net_lin"] = pd.to_numeric(seedwise["delta_sharpe_net_lin"], errors="coerce")

    kappas = list(seedwise["kappa"].dropna().sort_values().unique())
    fig, axes = plt.subplots(1, len(kappas), figsize=(4.7 * len(kappas), 4.2), constrained_layout=True, sharey=True)
    if len(kappas) == 1:
        axes = [axes]

    for ax, kappa in zip(axes, kappas):
        sub = seedwise[np.isclose(seedwise["kappa"], kappa, atol=1e-15)].sort_values("seed")
        style = FIGURE_STYLE.get(float(kappa), {"color": "#444444", "label": f"kappa={kappa:g}"})
        ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
        ax.scatter(sub["seed"], sub["delta_sharpe_net_lin"], color=style["color"], s=40)
        ax.plot(sub["seed"], sub["delta_sharpe_net_lin"], color=style["color"], linewidth=1.1, alpha=0.7)
        ax.set_title(style["label"])
        ax.set_xlabel("Seed")
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel(r"Paired $\Delta$ net Sharpe vs $\eta=1.0$")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _copy_legacy(output_path: Path, legacy_dir: Path, legacy_name: str) -> None:
    legacy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, legacy_dir / legacy_name)


def main() -> None:
    args = parse_args()
    validation_root = Path(args.validation_root)
    selection_json = Path(args.selection_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = Path(args.legacy_dir) if args.legacy_dir else None

    selected_eta = _selected_eta(args.selected_eta, selection_json)
    representative_json = Path(args.representative_json) if args.representative_json else None
    if representative_json is None or not representative_json.exists():
        raise FileNotFoundError("Representative seed JSON is required to build the selected-trace figure.")
    representative_meta, trace_path = _load_representative_path(representative_json)

    fig_selected_trace = output_dir / "fig_selected_trace.png"
    fig_validation_frontier = output_dir / "fig_validation_frontier.png"
    fig_seed_scatter = output_dir / "fig_seed_scatter.png"

    _build_selected_trace_figure(trace_path, representative_meta, fig_selected_trace)
    _build_validation_frontier_figure(validation_root, selected_eta=selected_eta, output_path=fig_validation_frontier)

    if args.seedwise_stats_csv:
        _build_seed_scatter(Path(args.seedwise_stats_csv), fig_seed_scatter)

    payload = {
        "selected_eta": selected_eta,
        "representative_seed": representative_meta["representative_seed"],
        "representative_kappa": representative_meta["representative_kappa"],
        "fig_selected_trace": str(fig_selected_trace),
        "fig_validation_frontier": str(fig_validation_frontier),
        "fig_seed_scatter": str(fig_seed_scatter) if fig_seed_scatter.exists() else "",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(payload, indent=2))

    if legacy_dir is not None:
        _copy_legacy(fig_selected_trace, legacy_dir, "fig_misalignment.png")
        _copy_legacy(fig_validation_frontier, legacy_dir, "fig_frontier.png")
        if fig_seed_scatter.exists():
            _copy_legacy(fig_seed_scatter, legacy_dir, "fig_seed_scatter.png")

    print(f"WROTE_FIGURE={fig_selected_trace}")
    print(f"WROTE_FIGURE={fig_validation_frontier}")
    if fig_seed_scatter.exists():
        print(f"WROTE_FIGURE={fig_seed_scatter}")


if __name__ == "__main__":
    main()
