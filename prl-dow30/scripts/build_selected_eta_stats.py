#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired selected-eta statistics against eta=1.0.")
    parser.add_argument("--final-root", type=str, required=True, help="Final/test step6 root.")
    parser.add_argument("--selection-json", type=str, required=True, help="Validation eta selection JSON.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write stats artifacts into.")
    parser.add_argument("--baseline-eta", type=float, default=1.0, help="Immediate-execution baseline eta.")
    parser.add_argument("--selected-eta", type=float, default=np.nan, help="Optional selected eta override.")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap resamples for CI estimation.")
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05, help="Bootstrap CI alpha level.")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed.")
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
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {root}")

    out = pd.DataFrame(rows)
    numeric_cols = [
        "kappa",
        "seed",
        "eta",
        "eta_requested",
        "sharpe_net_lin",
        "cagr",
        "maxdd",
        "avg_turnover_exec",
    ]
    for column in numeric_cols:
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


def _sign_test_one_sided(num_positive: int, num_pairs: int) -> float:
    if num_pairs <= 0:
        return float("nan")
    return float(sum(comb(num_pairs, k) for k in range(num_positive, num_pairs + 1)) / (2**num_pairs))


def _try_stats():
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None
    return stats


def _bootstrap_stat_ci(
    values: np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        resample = rng.choice(values, size=values.size, replace=True)
        samples[i] = float(stat_fn(resample))
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high


def _wilcoxon_signed_rank(values: np.ndarray) -> tuple[float, str]:
    if values.size < 2:
        return float("nan"), "n_lt_2"
    if np.allclose(values, 0.0):
        return 1.0, "all_zero"
    stats = _try_stats()
    if stats is None:
        return float("nan"), "scipy_unavailable"
    try:
        return float(stats.wilcoxon(values).pvalue), ""
    except Exception as exc:
        return float("nan"), f"wilcoxon_error:{exc}"


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


def _build_summary(
    runs: pd.DataFrame,
    *,
    selected_eta: float,
    baseline_eta: float,
    n_boot: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = runs[np.isclose(runs["pair_eta"], selected_eta, atol=1e-12)].copy()
    baseline = runs[np.isclose(runs["pair_eta"], baseline_eta, atol=1e-12)].copy()
    if selected.empty or baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    baseline_index = baseline.set_index(["kappa", "seed"])
    seedwise_rows: list[dict] = []
    summary_rows: list[dict] = []

    for kappa, grp in selected.groupby("kappa"):
        base_grp = baseline[np.isclose(baseline["kappa"], float(kappa), atol=1e-15)].copy()
        if base_grp.empty:
            continue
        deltas: list[dict] = []
        for _, row in grp.iterrows():
            key = (float(row["kappa"]), int(row["seed"]))
            if key not in baseline_index.index:
                continue
            base = baseline_index.loc[key]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            delta_row = {
                "kappa": float(kappa),
                "seed": int(row["seed"]),
                "selected_eta": float(selected_eta),
                "baseline_eta": float(baseline_eta),
                "selected_sharpe_net_lin": float(row["sharpe_net_lin"]),
                "baseline_sharpe_net_lin": float(base["sharpe_net_lin"]),
                "delta_sharpe_net_lin": float(row["sharpe_net_lin"]) - float(base["sharpe_net_lin"]),
                "selected_cagr": float(row["cagr"]),
                "baseline_cagr": float(base["cagr"]),
                "delta_cagr": float(row["cagr"]) - float(base["cagr"]),
                "selected_turnover_exec": float(row["avg_turnover_exec"]),
                "baseline_turnover_exec": float(base["avg_turnover_exec"]),
                "delta_turnover_exec": float(row["avg_turnover_exec"]) - float(base["avg_turnover_exec"]),
                "selected_maxdd": float(row["maxdd"]),
                "baseline_maxdd": float(base["maxdd"]),
                "delta_maxdd": float(row["maxdd"]) - float(base["maxdd"]),
            }
            deltas.append(delta_row)
            seedwise_rows.append(delta_row)

        delta_df = pd.DataFrame(deltas)
        if delta_df.empty:
            continue
        num_pairs = int(len(delta_df))
        num_positive = int((delta_df["delta_sharpe_net_lin"] > 0.0).sum())
        wilcoxon_p_sharpe, wilcoxon_skip_sharpe = _wilcoxon_signed_rank(delta_df["delta_sharpe_net_lin"].to_numpy(dtype=np.float64))
        wilcoxon_p_cagr, wilcoxon_skip_cagr = _wilcoxon_signed_rank(delta_df["delta_cagr"].to_numpy(dtype=np.float64))
        ci_low_delta_sharpe, ci_high_delta_sharpe = _bootstrap_stat_ci(
            delta_df["delta_sharpe_net_lin"].to_numpy(dtype=np.float64),
            stat_fn=lambda arr: float(np.median(arr)),
            n_boot=n_boot,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + int(round(float(kappa) * 1_000_000.0)),
        )
        ci_low_delta_cagr, ci_high_delta_cagr = _bootstrap_stat_ci(
            delta_df["delta_cagr"].to_numpy(dtype=np.float64),
            stat_fn=lambda arr: float(np.median(arr)),
            n_boot=n_boot,
            alpha=bootstrap_alpha,
            seed=bootstrap_seed + 10_000 + int(round(float(kappa) * 1_000_000.0)),
        )
        summary_rows.append(
            {
                "kappa": float(kappa),
                "selected_eta": float(selected_eta),
                "baseline_eta": float(baseline_eta),
                "n_pairs": num_pairs,
                "n_wins_sharpe": num_positive,
                "win_rate_sharpe": float(num_positive / num_pairs),
                "sign_test_one_sided_p": _sign_test_one_sided(num_positive, num_pairs),
                "wilcoxon_two_sided_p_delta_sharpe": wilcoxon_p_sharpe,
                "wilcoxon_skipped_reason_delta_sharpe": wilcoxon_skip_sharpe,
                "selected_median_sharpe_net_lin": float(grp["sharpe_net_lin"].median()),
                "selected_iqr_sharpe_net_lin": _iqr(grp["sharpe_net_lin"]),
                "baseline_median_sharpe_net_lin": float(base_grp["sharpe_net_lin"].median()),
                "baseline_iqr_sharpe_net_lin": _iqr(base_grp["sharpe_net_lin"]),
                "median_delta_sharpe_net_lin": float(delta_df["delta_sharpe_net_lin"].median()),
                "iqr_delta_sharpe_net_lin": _iqr(delta_df["delta_sharpe_net_lin"]),
                "bootstrap_ci_low_median_delta_sharpe_net_lin": ci_low_delta_sharpe,
                "bootstrap_ci_high_median_delta_sharpe_net_lin": ci_high_delta_sharpe,
                "selected_median_cagr": float(grp["cagr"].median()),
                "selected_iqr_cagr": _iqr(grp["cagr"]),
                "baseline_median_cagr": float(base_grp["cagr"].median()),
                "baseline_iqr_cagr": _iqr(base_grp["cagr"]),
                "median_delta_cagr": float(delta_df["delta_cagr"].median()),
                "iqr_delta_cagr": _iqr(delta_df["delta_cagr"]),
                "wilcoxon_two_sided_p_delta_cagr": wilcoxon_p_cagr,
                "wilcoxon_skipped_reason_delta_cagr": wilcoxon_skip_cagr,
                "bootstrap_ci_low_median_delta_cagr": ci_low_delta_cagr,
                "bootstrap_ci_high_median_delta_cagr": ci_high_delta_cagr,
                "selected_median_turnover_exec": float(grp["avg_turnover_exec"].median()),
                "baseline_median_turnover_exec": float(base_grp["avg_turnover_exec"].median()),
                "median_delta_turnover_exec": float(delta_df["delta_turnover_exec"].median()),
                "selected_median_maxdd": float(grp["maxdd"].median()),
                "baseline_median_maxdd": float(base_grp["maxdd"].median()),
                "median_delta_maxdd": float(delta_df["delta_maxdd"].median()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("kappa").reset_index(drop=True)
    seedwise_df = pd.DataFrame(seedwise_rows).sort_values(["kappa", "seed"]).reset_index(drop=True)
    return summary_df, seedwise_df


def main() -> None:
    args = parse_args()
    final_root = Path(args.final_root)
    selection_json = Path(args.selection_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_eta = _selected_eta(args.selected_eta, selection_json)
    runs = _load_runs(final_root)
    summary_df, seedwise_df = _build_summary(
        runs,
        selected_eta=selected_eta,
        baseline_eta=float(args.baseline_eta),
        n_boot=int(args.bootstrap),
        bootstrap_alpha=float(args.bootstrap_alpha),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    _write_csv_md(
        summary_df,
        csv_path=output_dir / "selected_eta_vs_eta1_stats.csv",
        md_path=output_dir / "selected_eta_vs_eta1_stats.md",
        title="Selected Eta Vs Immediate Execution Stats",
    )
    _write_csv_md(
        seedwise_df,
        csv_path=output_dir / "selected_eta_seedwise_deltas.csv",
        md_path=output_dir / "selected_eta_seedwise_deltas.md",
        title="Selected Eta Seedwise Deltas",
    )

    payload = {
        "selected_eta": float(selected_eta),
        "baseline_eta": float(args.baseline_eta),
        "bootstrap": int(args.bootstrap),
        "bootstrap_alpha": float(args.bootstrap_alpha),
        "bootstrap_seed": int(args.bootstrap_seed),
        "summary_csv": str(output_dir / "selected_eta_vs_eta1_stats.csv"),
        "seedwise_csv": str(output_dir / "selected_eta_seedwise_deltas.csv"),
    }
    (output_dir / "selected_eta_stats_meta.json").write_text(json.dumps(payload, indent=2))
    print(f"WROTE_STATS={output_dir / 'selected_eta_vs_eta1_stats.csv'}")
    print(f"WROTE_SEEDWISE={output_dir / 'selected_eta_seedwise_deltas.csv'}")


if __name__ == "__main__":
    main()
