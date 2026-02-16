from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step6 production reports.")
    parser.add_argument("--root", type=str, default="outputs/step6", help="Step6 run root.")
    return parser.parse_args()


def _parse_kappa_from_dir(path: Path) -> float:
    name = path.name
    if not name.startswith("kappa_"):
        raise ValueError(f"Invalid kappa dir name: {name}")
    return float(name.split("kappa_", 1)[1])


def _parse_seed_from_dir(path: Path) -> int:
    name = path.name
    if not name.startswith("seed_"):
        raise ValueError(f"Invalid seed dir name: {name}")
    return int(name.split("seed_", 1)[1])


def _parse_eta_from_dir(path: Path) -> float:
    name = path.name
    if not name.startswith("eta_"):
        raise ValueError(f"Invalid eta dir name: {name}")
    return float(name.split("eta_", 1)[1])


def _collect_runs(root: Path) -> pd.DataFrame:
    rows = []
    metrics_paths = sorted(root.glob("kappa_*/eta_*/seed_*/metrics.csv"))
    has_eta_dir_structure = len(metrics_paths) > 0
    if not has_eta_dir_structure:
        metrics_paths = sorted(root.glob("kappa_*/seed_*/metrics.csv"))

    for metrics_path in metrics_paths:
        seed_dir = metrics_path.parent
        seed = _parse_seed_from_dir(seed_dir)

        if has_eta_dir_structure:
            eta_dir = seed_dir.parent
            kappa_dir = eta_dir.parent
            eta = _parse_eta_from_dir(eta_dir)
        else:
            eta_dir = None
            kappa_dir = seed_dir.parent
            eta = np.nan

        kappa = _parse_kappa_from_dir(kappa_dir)
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["kappa"] = float(kappa)
        row["seed"] = int(seed)
        row["eta"] = float(row.get("eta", eta))
        row["eta_requested"] = float(row.get("eta_requested", eta))
        row["run_dir"] = str(seed_dir)
        row["eta_dir"] = str(eta_dir) if eta_dir is not None else None
        row["metrics_path"] = str(metrics_path)
        row["trace_path"] = row.get("trace_path", str(seed_dir / "trace.parquet"))
        rows.append(row)
    if not rows:
        raise ValueError(f"No run metrics found under: {root}")
    out = pd.DataFrame(rows)
    out["collapse_flag_any"] = out.get("collapse_flag_any", False).astype(bool)
    out["kappa"] = pd.to_numeric(out["kappa"], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
    out["eta"] = pd.to_numeric(out.get("eta", np.nan), errors="coerce")
    out["eta_requested"] = pd.to_numeric(out.get("eta_requested", np.nan), errors="coerce")
    out["pair_eta"] = out["eta_requested"].where(~out["eta_requested"].isna(), out["eta"])
    return out


def _build_aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kappa, eta), group in runs.groupby(["kappa", "pair_eta"]):
        sharpe = pd.to_numeric(group["sharpe_net_lin"], errors="coerce")
        turnover = pd.to_numeric(group["avg_turnover_exec"], errors="coerce")
        collapse = group["collapse_flag_any"].astype(bool)
        q75 = float(sharpe.quantile(0.75))
        q25 = float(sharpe.quantile(0.25))
        rows.append(
            {
                "kappa": float(kappa),
                "eta": float(eta),
                "n_runs": int(len(group)),
                "median_sharpe": float(sharpe.median()),
                "iqr_sharpe": float(q75 - q25),
                "median_turnover_exec": float(turnover.median()),
                "collapse_rate": float(collapse.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["kappa", "eta"]).reset_index(drop=True)


def _build_paired_delta(runs: pd.DataFrame) -> pd.DataFrame:
    baseline = runs[np.isclose(pd.to_numeric(runs["kappa"], errors="coerce"), 0.0, atol=1e-15)]
    if baseline.empty:
        raise ValueError("Baseline (kappa=0) runs are required for paired deltas.")
    base_by_seed_eta = baseline.set_index(["seed", "pair_eta"])

    pair_keys = sorted({(int(row.seed), float(row.pair_eta)) for row in runs.itertuples()})
    missing_baseline = [key for key in pair_keys if key not in base_by_seed_eta.index]
    if missing_baseline:
        raise ValueError(f"Missing kappa=0 baseline for (seed, eta): {missing_baseline}")

    rows = []
    for _, row in runs.iterrows():
        kappa = float(row["kappa"])
        seed = int(row["seed"])
        eta = float(row["pair_eta"])
        if np.isclose(kappa, 0.0, atol=1e-15):
            continue
        base = base_by_seed_eta.loc[(seed, eta)]
        rows.append(
            {
                "seed": seed,
                "kappa": kappa,
                "eta": eta,
                "run_id": row.get("run_id"),
                "baseline_run_id": base.get("run_id"),
                "delta_sharpe": float(row.get("sharpe_net_lin", np.nan)) - float(base.get("sharpe_net_lin", np.nan)),
                "delta_cagr": float(row.get("cagr", np.nan)) - float(base.get("cagr", np.nan)),
                "delta_maxdd": float(row.get("maxdd", np.nan)) - float(base.get("maxdd", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "seed",
                "kappa",
                "eta",
                "run_id",
                "baseline_run_id",
                "delta_sharpe",
                "delta_cagr",
                "delta_maxdd",
            ]
        )
    return pd.DataFrame(rows).sort_values(["seed", "eta", "kappa"]).reset_index(drop=True)


def _build_collapse_report(root: Path, runs: pd.DataFrame) -> None:
    total = int(len(runs))
    collapsed = runs[runs["collapse_flag_any"].astype(bool)]

    lines = [
        "# Collapse Report",
        f"- total_runs: {total}",
        f"- collapsed_runs: {int(len(collapsed))}",
        f"- collapse_rate: {float(len(collapsed) / max(total, 1)):.6f}",
        "",
        "## By Kappa",
    ]
    for kappa, group in runs.groupby("kappa"):
        rate = float(group["collapse_flag_any"].astype(bool).mean())
        lines.append(f"- kappa={kappa:g}: collapse_rate={rate:.6f} ({int(group['collapse_flag_any'].sum())}/{len(group)})")

    lines.append("")
    lines.append("## Collapsed Runs")
    if collapsed.empty:
        lines.append("- None")
    else:
        for _, row in collapsed.iterrows():
            lines.append(
                f"- kappa={float(row['kappa']):g}, eta={float(row['pair_eta']):g}, "
                f"seed={int(row['seed'])}, run_dir={row['run_dir']}"
            )

    (root / "collapse_report.md").write_text("\n".join(lines) + "\n")


def _pick_trace_for_misalignment(runs: pd.DataFrame) -> Path:
    preferred = runs[
        np.isclose(pd.to_numeric(runs["kappa"], errors="coerce"), 0.001, atol=1e-12)
        & (pd.to_numeric(runs["seed"], errors="coerce") == 0)
    ]
    if not preferred.empty:
        # Use the smallest eta for clear misalignment visualization.
        candidate = preferred.sort_values("pair_eta").iloc[0]
        return Path(str(candidate["trace_path"]))

    baseline = runs[np.isclose(pd.to_numeric(runs["kappa"], errors="coerce"), 0.001, atol=1e-12)]
    if not baseline.empty:
        candidate = baseline.sort_values(["seed", "pair_eta"]).iloc[0]
        return Path(str(candidate["trace_path"]))

    baseline = runs[np.isclose(pd.to_numeric(runs["kappa"], errors="coerce"), 0.0, atol=1e-15)]
    if not baseline.empty:
        candidate = baseline.sort_values(["seed", "pair_eta"]).iloc[0]
        return Path(str(candidate["trace_path"]))
    return Path(str(runs.iloc[0]["trace_path"]))


def _build_fig_misalignment(root: Path, runs: pd.DataFrame) -> None:
    trace_path = _pick_trace_for_misalignment(runs)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found for misalignment figure: {trace_path}")
    df = pd.read_parquet(trace_path)

    required = {"equity_net_lin", "equity_net_lin_target", "turnover_exec", "turnover_target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in trace for fig_misalignment: {missing}")

    x = pd.to_datetime(df["date"]) if "date" in df.columns else np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(x, df["equity_net_lin"], label="equity_net_lin")
    ax1.plot(x, df["equity_net_lin_target"], label="equity_net_lin_target")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, df["turnover_exec"], label="turnover_exec")
    ax2.plot(x, df["turnover_target"], label="turnover_target")
    ax2.set_ylabel("Turnover")
    ax2.set_xlabel("Date")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(root / "fig_misalignment.png", dpi=150)
    plt.close(fig)


def _build_fig_frontier(root: Path, runs: pd.DataFrame) -> None:
    frontier_runs = runs[np.isclose(pd.to_numeric(runs["kappa"], errors="coerce"), 0.001, atol=1e-12)]
    if frontier_runs.empty:
        frontier_runs = runs

    grouped = (
        frontier_runs.groupby("pair_eta", as_index=False)
        .agg(
            avg_turnover_exec=("avg_turnover_exec", "mean"),
            sharpe_net_lin=("sharpe_net_lin", "median"),
            eta=("pair_eta", "median"),
        )
        .sort_values("eta", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    x = pd.to_numeric(grouped["avg_turnover_exec"], errors="coerce")
    y = pd.to_numeric(grouped["sharpe_net_lin"], errors="coerce")
    ax.plot(x, y, alpha=0.4)
    ax.scatter(x, y)
    for _, row in grouped.iterrows():
        label = f"eta={float(row['eta']):.3g}"
        ax.annotate(label, (float(row["avg_turnover_exec"]), float(row["sharpe_net_lin"])))
    ax.set_xlabel("avg_turnover_exec")
    ax.set_ylabel("sharpe_net_lin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "fig_frontier.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(root)
    aggregate = _build_aggregate(runs)
    paired = _build_paired_delta(runs)

    aggregate.to_csv(root / "aggregate.csv", index=False)
    paired.to_csv(root / "paired_delta.csv", index=False)
    _build_collapse_report(root, runs)
    _build_fig_misalignment(root, runs)
    _build_fig_frontier(root, runs)


if __name__ == "__main__":
    main()
