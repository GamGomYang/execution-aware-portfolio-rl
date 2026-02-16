from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from analysis.step5_common import (
    STEP5_EXPERIMENT_LABELS,
    STEP5_MAIN_METRICS,
    describe,
    load_latest_archive_frames,
    pair_seed_values,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)


def _metric_label(metric: str) -> str:
    mapping = {
        "sharpe_net_exp": "Sharpe (net_exp)",
        "cumulative_return_net_exp": "Cumulative Return (net_exp)",
        "max_drawdown_net_exp": "Max Drawdown (net_exp)",
        "avg_turnover_exec": "Avg Turnover (exec)",
        "avg_turnover_target": "Avg Turnover (target)",
        "std_daily_net_return_exp": "Std Daily Net Return (exp)",
    }
    return mapping.get(metric, metric)


def _normalize_turnover_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "avg_turnover_exec" not in out.columns and "avg_turnover" in out.columns:
        out["avg_turnover_exec"] = pd.to_numeric(out["avg_turnover"], errors="coerce")
    if "total_turnover_exec" not in out.columns and "total_turnover" in out.columns:
        out["total_turnover_exec"] = pd.to_numeric(out["total_turnover"], errors="coerce")
    return out


def _paired_metric_summary(base_df: pd.DataFrame, comp_df: pd.DataFrame, metric: str) -> dict:
    seeds, base_vals, comp_vals, delta = pair_seed_values(base_df, comp_df, metric)
    b = describe(base_vals)
    c = describe(comp_vals)
    d = describe(delta)
    return {
        "metric": metric,
        "n_pairs": len(seeds),
        "baseline_mean": b["mean"],
        "baseline_std": b["std"],
        "baseline_p25": b["p25"],
        "baseline_median": b["median"],
        "baseline_p75": b["p75"],
        "baseline_iqr": b["iqr"],
        "prl_mean": c["mean"],
        "prl_std": c["std"],
        "prl_p25": c["p25"],
        "prl_median": c["median"],
        "prl_p75": c["p75"],
        "prl_iqr": c["iqr"],
        "delta_mean": d["mean"],
        "delta_std": d["std"],
        "delta_p25": d["p25"],
        "delta_median": d["median"],
        "delta_p75": d["p75"],
        "delta_iqr": d["iqr"],
    }


def build_main_table(base_df: pd.DataFrame, prl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in STEP5_MAIN_METRICS:
        if metric not in base_df.columns or metric not in prl_df.columns:
            continue
        rows.append(_paired_metric_summary(base_df, prl_df, metric))
    return pd.DataFrame(rows)


def build_stats_tests(base_df: pd.DataFrame, prl_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        if metric not in base_df.columns or metric not in prl_df.columns:
            continue
        _, base_vals, prl_vals, delta = pair_seed_values(base_df, prl_df, metric)
        n_pairs = int(delta.size)
        median_delta = float(np.median(delta)) if n_pairs else float("nan")
        if n_pairs >= 2:
            t_res = stats.ttest_rel(prl_vals, base_vals)
            t_stat = float(t_res.statistic)
            t_p = float(t_res.pvalue)
            d_std = float(np.std(delta, ddof=1))
            effect_t = float(np.mean(delta) / d_std) if d_std > 1e-12 else float("nan")
        else:
            t_stat = float("nan")
            t_p = float("nan")
            effect_t = float("nan")

        rows.append(
            {
                "metric": metric,
                "test": "ttest_rel",
                "statistic": t_stat,
                "p_value": t_p,
                "effect_size": effect_t,
                "median_delta": median_delta,
                "n_pairs": n_pairs,
            }
        )

        if n_pairs >= 2:
            if np.allclose(delta, 0.0):
                w_stat = 0.0
                w_p = 1.0
            else:
                try:
                    w = stats.wilcoxon(delta)
                    w_stat = float(w.statistic)
                    w_p = float(w.pvalue)
                except Exception:
                    w_stat = float("nan")
                    w_p = float("nan")
        else:
            w_stat = float("nan")
            w_p = float("nan")

        rows.append(
            {
                "metric": metric,
                "test": "wilcoxon",
                "statistic": w_stat,
                "p_value": w_p,
                "effect_size": median_delta,
                "median_delta": median_delta,
                "n_pairs": n_pairs,
            }
        )
    return pd.DataFrame(rows)


def build_regime_table(base_df: pd.DataFrame, prl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["low", "mid", "high"]:
        base_r = base_df[base_df["regime"] == regime]
        prl_r = prl_df[prl_df["regime"] == regime]
        for metric in STEP5_MAIN_METRICS:
            if metric not in base_r.columns or metric not in prl_r.columns:
                continue
            summary = _paired_metric_summary(base_r, prl_r, metric)
            summary["regime"] = regime
            rows.append(summary)
    return pd.DataFrame(rows)


def _delta_stats(base_df: pd.DataFrame, comp_df: pd.DataFrame, metric: str) -> tuple[dict, dict[int, float]]:
    seeds, _, _, delta = pair_seed_values(base_df, comp_df, metric)
    by_seed = {int(seed): float(delta[idx]) for idx, seed in enumerate(seeds)}
    d = describe(delta)
    return {
        "mean": d["mean"],
        "std": d["std"],
        "p25": d["p25"],
        "median": d["median"],
        "p75": d["p75"],
        "iqr": d["iqr"],
        "n": len(seeds),
    }, by_seed


def build_ablation_table(metrics_by_key: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = {"A", "B", "C", "D"}
    if not required.issubset(set(metrics_by_key.keys())):
        return pd.DataFrame()

    a_df = metrics_by_key["A"]
    b_df = metrics_by_key["B"]
    c_df = metrics_by_key["C"]
    d_df = metrics_by_key["D"]

    rows = []
    for metric in STEP5_MAIN_METRICS:
        if any(metric not in frame.columns for frame in [a_df, b_df, c_df, d_df]):
            continue

        prl_effect, ba_map = _delta_stats(a_df, b_df, metric)  # B - A
        eta_effect_baseline, ac_map = _delta_stats(c_df, a_df, metric)  # A - C
        eta_effect_prl, db_map = _delta_stats(d_df, b_df, metric)  # B - D
        total_effect, bc_map = _delta_stats(c_df, b_df, metric)  # B - C

        dc_effect, dc_map = _delta_stats(c_df, d_df, metric)  # D - C
        synergy_seeds = sorted(set(ba_map.keys()) & set(dc_map.keys()))
        synergy_values = np.asarray([ba_map[s] - dc_map[s] for s in synergy_seeds], dtype=np.float64)
        synergy_stats = describe(synergy_values)

        rows.append(
            {
                "metric": metric,
                "prl_effect_eta010_mean": prl_effect["mean"],
                "prl_effect_eta010_median": prl_effect["median"],
                "prl_effect_eta010_p25": prl_effect["p25"],
                "prl_effect_eta010_p75": prl_effect["p75"],
                "prl_effect_eta010_iqr": prl_effect["iqr"],
                "eta_effect_baseline_mean": eta_effect_baseline["mean"],
                "eta_effect_baseline_median": eta_effect_baseline["median"],
                "eta_effect_baseline_p25": eta_effect_baseline["p25"],
                "eta_effect_baseline_p75": eta_effect_baseline["p75"],
                "eta_effect_baseline_iqr": eta_effect_baseline["iqr"],
                "eta_effect_prl_mean": eta_effect_prl["mean"],
                "eta_effect_prl_median": eta_effect_prl["median"],
                "eta_effect_prl_p25": eta_effect_prl["p25"],
                "eta_effect_prl_p75": eta_effect_prl["p75"],
                "eta_effect_prl_iqr": eta_effect_prl["iqr"],
                "total_effect_B_minus_C_mean": total_effect["mean"],
                "total_effect_B_minus_C_median": total_effect["median"],
                "total_effect_B_minus_C_p25": total_effect["p25"],
                "total_effect_B_minus_C_p75": total_effect["p75"],
                "total_effect_B_minus_C_iqr": total_effect["iqr"],
                "synergy_mean": synergy_stats["mean"],
                "synergy_median": synergy_stats["median"],
                "synergy_p25": synergy_stats["p25"],
                "synergy_p75": synergy_stats["p75"],
                "synergy_iqr": synergy_stats["iqr"],
            }
        )
    return pd.DataFrame(rows)


def _format_median_iqr(row: pd.Series, prefix: str) -> str:
    return f"{row[f'{prefix}_median']:.3f} [{row[f'{prefix}_p25']:.3f}, {row[f'{prefix}_p75']:.3f}]"


def build_main_robust_tex(table_main: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Metric & Baseline (median [P25, P75]) & PRL (median [P25, P75]) & $\\Delta$ (median [P25, P75]) \\\\",
        "\\hline",
    ]
    for _, row in table_main.iterrows():
        label = _metric_label(str(row["metric"]))
        lines.append(
            f"{label} & {_format_median_iqr(row, 'baseline')} & {_format_median_iqr(row, 'prl')} & {_format_median_iqr(row, 'delta')} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _seed_from_run_id(run_id: str) -> int | None:
    match = re.search(r"_seed(\d+)_", run_id)
    if not match:
        return None
    return int(match.group(1))


def _load_trace_frames(input_root: Path, run_ids: list[str]) -> pd.DataFrame:
    reports_dir = input_root / "reports"
    frames = []
    for run_id in run_ids:
        trace_paths = sorted(reports_dir.glob(f"trace_{run_id}*.parquet"))
        for trace_path in trace_paths:
            try:
                trace_df = pd.read_parquet(trace_path)
            except Exception:
                continue
            if trace_df.empty or "date" not in trace_df.columns:
                continue
            trace_df = trace_df.copy()
            trace_df["run_id"] = run_id
            if "seed" not in trace_df.columns:
                seed = _seed_from_run_id(run_id)
                trace_df["seed"] = seed if seed is not None else -1
            trace_df["date"] = pd.to_datetime(trace_df["date"])
            if "eval_window" in trace_df.columns:
                windows = sorted(trace_df["eval_window"].dropna().unique().tolist())
                if windows:
                    target = "W1" if "W1" in windows else windows[0]
                    trace_df = trace_df[trace_df["eval_window"] == target].copy()
            frames.append(trace_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _curve_from_trace(trace_df: pd.DataFrame, *, value_col: str) -> dict[int, pd.Series]:
    curves: dict[int, pd.Series] = {}
    if trace_df.empty:
        return curves
    for seed, group in trace_df.groupby("seed"):
        group = group.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if value_col in group.columns:
            series = pd.to_numeric(group[value_col], errors="coerce")
            values = series.to_numpy(dtype=np.float64)
            if value_col.startswith("equity"):
                curve = values
            else:
                curve = np.cumprod(1.0 + np.nan_to_num(values, nan=0.0))
        elif "net_return_exp" in group.columns:
            values = pd.to_numeric(group["net_return_exp"], errors="coerce").to_numpy(dtype=np.float64)
            curve = np.cumprod(1.0 + np.nan_to_num(values, nan=0.0))
        else:
            continue
        curves[int(seed)] = pd.Series(curve, index=pd.to_datetime(group["date"]))
    return curves


def _placeholder_figure(path: Path, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.4, reason, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_equity_curve(path: Path, traces: dict[str, pd.DataFrame]) -> None:
    colors = {"A": "tab:blue", "B": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for key in ["A", "B"]:
        df = traces.get(key)
        if df is None or df.empty:
            continue
        curves = _curve_from_trace(df, value_col="equity_net_exp")
        if not curves:
            continue
        panel = pd.concat(curves.values(), axis=1).sort_index()
        mean = panel.mean(axis=1)
        q25 = panel.quantile(0.25, axis=1)
        q75 = panel.quantile(0.75, axis=1)
        label = STEP5_EXPERIMENT_LABELS.get(key, key)
        color = colors.get(key)
        ax.plot(mean.index, mean.values, label=label, color=color)
        ax.fill_between(mean.index, q25.values, q75.values, alpha=0.2, color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        _placeholder_figure(path, "Net Equity Curve", "trace files not found")
        return
    ax.set_title("Net Equity Curve (net_exp, mean with IQR band)")
    ax.set_ylabel("equity")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_drawdown_curve(path: Path, traces: dict[str, pd.DataFrame]) -> None:
    colors = {"A": "tab:blue", "B": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for key in ["A", "B"]:
        df = traces.get(key)
        if df is None or df.empty:
            continue
        curves = _curve_from_trace(df, value_col="equity_net_exp")
        if not curves:
            continue
        dd_curves = []
        for series in curves.values():
            dd = series / series.cummax() - 1.0
            dd_curves.append(dd)
        panel = pd.concat(dd_curves, axis=1).sort_index()
        mean = panel.mean(axis=1)
        label = STEP5_EXPERIMENT_LABELS.get(key, key)
        ax.plot(mean.index, mean.values, label=label, color=colors.get(key))
        plotted = True
    if not plotted:
        plt.close(fig)
        _placeholder_figure(path, "Net Drawdown Curve", "trace files not found")
        return
    ax.set_title("Net Drawdown Curve (net_exp mean)")
    ax.set_ylabel("drawdown")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_turnover_summary(path: Path, base_df: pd.DataFrame, prl_df: pd.DataFrame) -> None:
    labels = []
    data = []

    base_exec = pd.to_numeric(base_df.get("avg_turnover_exec"), errors="coerce").dropna().to_numpy(dtype=np.float64)
    prl_exec = pd.to_numeric(prl_df.get("avg_turnover_exec"), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if base_exec.size:
        labels.append("Base exec")
        data.append(base_exec)
    if prl_exec.size:
        labels.append("PRL exec")
        data.append(prl_exec)

    if "avg_turnover_target" in base_df.columns and "avg_turnover_target" in prl_df.columns:
        base_t = pd.to_numeric(base_df["avg_turnover_target"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        prl_t = pd.to_numeric(prl_df["avg_turnover_target"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if base_t.size:
            labels.append("Base target")
            data.append(base_t)
        if prl_t.size:
            labels.append("PRL target")
            data.append(prl_t)

    if not data:
        _placeholder_figure(path, "Turnover Summary", "turnover columns not found")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:  # pragma: no cover - matplotlib < 3.9
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Turnover: exec vs target")
    ax.set_ylabel("turnover")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_summary_md(path: Path, table_main: pd.DataFrame) -> None:
    lines = ["# Step5 Summary"]
    if table_main.empty:
        lines.append("- No paired metrics available.")
    else:
        for metric in ["sharpe_net_exp", "cumulative_return_net_exp", "max_drawdown_net_exp"]:
            block = table_main[table_main["metric"] == metric]
            if block.empty:
                continue
            row = block.iloc[0]
            trend = "improved" if row["delta_median"] > 0 else "degraded" if row["delta_median"] < 0 else "unchanged"
            lines.append(
                f"- {metric}: median delta (PRL - baseline, eta=0.10) = {row['delta_median']:.4f} ({trend})."
            )
    path.write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Step5 final paper artifacts.")
    parser.add_argument("--input-root", required=True, help="Step5 run root (e.g., outputs/exp_runs/step5_final/<timestamp>).")
    parser.add_argument("--out-dir", help="Output directory. Default: <input-root>/reports/paper/step5")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir) if args.out_dir else input_root / "reports" / "paper" / "step5"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_loaded = load_latest_archive_frames(input_root, prefix="metrics")
    if "A" not in metrics_loaded or "B" not in metrics_loaded:
        raise ValueError("STEP5_MAIN_METRICS_MISSING: archive metrics for A/B were not found")

    metrics_by_key = {key: _normalize_turnover_cols(df) for key, (_, df) in metrics_loaded.items()}
    base_main = metrics_by_key["A"]
    prl_main = metrics_by_key["B"]

    table_main = build_main_table(base_main, prl_main)
    table_main.to_csv(out_dir / "table_main.csv", index=False)

    robust_cols = [
        "metric",
        "baseline_p25",
        "baseline_median",
        "baseline_p75",
        "baseline_iqr",
        "prl_p25",
        "prl_median",
        "prl_p75",
        "prl_iqr",
        "delta_p25",
        "delta_median",
        "delta_p75",
        "delta_iqr",
    ]
    robust_stats = table_main[robust_cols].copy() if not table_main.empty else pd.DataFrame(columns=robust_cols)
    robust_stats.to_csv(out_dir / "robust_stats_summary.csv", index=False)

    robust_delta = (
        table_main[["metric", "n_pairs", "delta_mean", "delta_std", "delta_p25", "delta_median", "delta_p75", "delta_iqr"]].copy()
        if not table_main.empty
        else pd.DataFrame(columns=["metric", "n_pairs", "delta_mean", "delta_std", "delta_p25", "delta_median", "delta_p75", "delta_iqr"])
    )
    robust_delta.to_csv(out_dir / "robust_delta_prl_minus_base.csv", index=False)

    tex = build_main_robust_tex(table_main)
    (out_dir / "table_main_robust.tex").write_text(tex)

    stats_tests = build_stats_tests(base_main, prl_main, STEP5_MAIN_METRICS)
    stats_tests.to_csv(out_dir / "stats_tests.csv", index=False)

    regime_loaded = load_latest_archive_frames(input_root, prefix="regime_metrics")
    if "A" in regime_loaded and "B" in regime_loaded:
        base_reg = _normalize_turnover_cols(regime_loaded["A"][1])
        prl_reg = _normalize_turnover_cols(regime_loaded["B"][1])
        table_regime = build_regime_table(base_reg, prl_reg)
    else:
        table_regime = pd.DataFrame()
    table_regime.to_csv(out_dir / "table_regime.csv", index=False)

    table_ablation = build_ablation_table(metrics_by_key)
    if not table_ablation.empty:
        table_ablation.to_csv(out_dir / "table_ablation.csv", index=False)

    traces = {}
    for key in ["A", "B"]:
        run_ids = sorted(metrics_by_key[key]["run_id"].dropna().astype(str).unique().tolist())
        traces[key] = _load_trace_frames(input_root, run_ids)

    plot_equity_curve(out_dir / "fig_equity_curve_net_exp.png", traces)
    plot_drawdown_curve(out_dir / "fig_drawdown_net_exp.png", traces)
    plot_turnover_summary(out_dir / "fig_turnover_exec_vs_target.png", base_main, prl_main)

    _write_summary_md(out_dir / "summary_step5.md", table_main)

    LOGGER.info("Wrote Step5 artifacts to %s", out_dir)
    print(f"Step5 analysis complete. Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
