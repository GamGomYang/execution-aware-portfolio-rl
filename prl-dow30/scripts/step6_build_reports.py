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
    metrics_paths = sorted(set(root.glob("kappa_*/*/seed_*/metrics.csv")) | set(root.glob("kappa_*/seed_*/metrics.csv")))
    for metrics_path in metrics_paths:
        seed_dir = metrics_path.parent
        seed = _parse_seed_from_dir(seed_dir)

        parent_dir = seed_dir.parent
        arm = None
        rule_vol_a = np.nan
        eta = np.nan
        if parent_dir.name.startswith("kappa_"):
            # Legacy kappa/seed layout.
            eta_dir = None
            kappa_dir = parent_dir
        else:
            eta_dir = parent_dir
            kappa_dir = eta_dir.parent
            label = eta_dir.name
            if label.startswith("eta_"):
                eta = _parse_eta_from_dir(eta_dir)
                arm = "eta_sweep"
            elif label in {"main", "baseline"}:
                arm = label
            elif label.startswith("rule_vol_a_"):
                arm = "rule_vol"
                rule_vol_a = float(label.split("rule_vol_a_", 1)[1])
            elif label.startswith("fixed_eta_"):
                arm = "fixed_comparison"
                eta = float(label.split("fixed_eta_", 1)[1])
            else:
                arm = label

        kappa = _parse_kappa_from_dir(kappa_dir)
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["kappa"] = float(kappa)
        row["seed"] = int(seed)
        row["eta"] = float(row.get("eta", eta))
        row["eta_requested"] = float(row.get("eta_requested", eta))
        row["arm"] = row.get("arm", arm)
        row["rule_vol_a"] = float(row.get("rule_vol_a", rule_vol_a))
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
    out["rule_vol_a"] = pd.to_numeric(out.get("rule_vol_a", np.nan), errors="coerce")
    if "arm" in out.columns:
        out["arm"] = out["arm"].astype("string")
    out["pair_eta"] = out["eta_requested"].where(~out["eta_requested"].isna(), out["eta"])
    return out


def _build_aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_arm = "arm" in runs.columns and runs["arm"].notna().any()
    group_cols = ["kappa", "pair_eta"]
    if has_arm:
        group_cols = ["kappa", "arm", "pair_eta"]
    for group_key, group in runs.groupby(group_cols):
        if has_arm:
            kappa, arm, eta = group_key
        else:
            kappa, eta = group_key
            arm = None
        sharpe = pd.to_numeric(group["sharpe_net_lin"], errors="coerce")
        turnover = pd.to_numeric(group["avg_turnover_exec"], errors="coerce")
        collapse = group["collapse_flag_any"].astype(bool)
        q75 = float(sharpe.quantile(0.75))
        q25 = float(sharpe.quantile(0.25))
        row = {
            "kappa": float(kappa),
            "eta": float(eta) if np.isfinite(float(eta)) else np.nan,
            "n_runs": int(len(group)),
            "median_sharpe": float(sharpe.median()),
            "iqr_sharpe": float(q75 - q25),
            "median_turnover_exec": float(turnover.median()),
            "collapse_rate": float(collapse.mean()),
        }
        if arm is not None:
            row["arm"] = str(arm)
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = ["kappa", "eta"]
    if "arm" in out.columns:
        sort_cols = ["kappa", "arm", "eta"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _build_paired_delta(runs: pd.DataFrame) -> pd.DataFrame:
    # EXP-1: paired baseline vs main within each kappa/seed.
    if "arm" in runs.columns and {"main", "baseline"} <= set(runs["arm"].dropna().unique().tolist()):
        main_rows = runs[runs["arm"] == "main"].copy()
        baseline_rows = runs[runs["arm"] == "baseline"].copy()
        if not main_rows.empty and not baseline_rows.empty:
            baseline_index = baseline_rows.set_index(["kappa", "seed"])
            rows = []
            for _, row in main_rows.iterrows():
                key = (float(row["kappa"]), int(row["seed"]))
                if key not in baseline_index.index:
                    continue
                base = baseline_index.loc[key]
                rows.append(
                    {
                        "seed": int(row["seed"]),
                        "kappa": float(row["kappa"]),
                        "eta_main": float(row.get("pair_eta", np.nan)),
                        "eta_baseline": float(base.get("pair_eta", np.nan)),
                        "run_id": row.get("run_id"),
                        "baseline_run_id": base.get("run_id"),
                        "delta_sharpe": float(row.get("sharpe_net_lin", np.nan)) - float(base.get("sharpe_net_lin", np.nan)),
                        "delta_cagr": float(row.get("cagr", np.nan)) - float(base.get("cagr", np.nan)),
                        "delta_maxdd": float(row.get("maxdd", np.nan)) - float(base.get("maxdd", np.nan)),
                    }
                )
            if rows:
                return pd.DataFrame(rows).sort_values(["seed", "kappa"]).reset_index(drop=True)

    # Existing behavior: paired against kappa=0 baseline by seed+eta.
    baseline = runs[np.isclose(pd.to_numeric(runs["kappa"], errors="coerce"), 0.0, atol=1e-15)]
    if baseline.empty:
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
    base_by_seed_eta = baseline.set_index(["seed", "pair_eta"])

    pair_keys = sorted({(int(row.seed), float(row.pair_eta)) for row in runs.itertuples()})
    missing_baseline = [key for key in pair_keys if key not in base_by_seed_eta.index]
    if missing_baseline:
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
        return
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
    grouped = grouped.dropna(subset=["avg_turnover_exec", "sharpe_net_lin", "eta"])
    if grouped.empty:
        return

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
