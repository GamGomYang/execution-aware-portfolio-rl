import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = float(np.mean(sample))
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def _try_stats():
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None
    return stats


def _collect_metric_columns(df: pd.DataFrame, *, include_turnover: bool = True) -> list[str]:
    base_cols = ["sharpe", "max_drawdown", "cumulative_return"]
    if include_turnover:
        if "avg_turnover_exec" in df.columns:
            base_cols.append("avg_turnover_exec")
        elif "avg_turnover" in df.columns:
            base_cols.append("avg_turnover")
        if "avg_turnover_target" in df.columns:
            base_cols.append("avg_turnover_target")
    metric_cols = [col for col in base_cols if col in df.columns]
    for col in sorted(df.columns):
        if col.startswith(("sharpe_net_", "max_drawdown_net_", "cumulative_return_net_")):
            metric_cols.append(col)
        if col.startswith(("mean_daily_", "std_daily_")):
            metric_cols.append(col)
    return metric_cols


def _metric_label(col: str) -> str:
    if col.startswith("sharpe_net_"):
        return f"Sharpe ({col.replace('sharpe_', '')})"
    if col.startswith("cumulative_return_net_"):
        return f"Cumulative Return ({col.replace('cumulative_return_', '')})"
    if col.startswith("max_drawdown_net_"):
        return f"Max Drawdown ({col.replace('max_drawdown_', '')})"
    label_map = {
        "sharpe": "Sharpe",
        "max_drawdown": "Max Drawdown",
        "avg_turnover": "Avg Turnover",
        "avg_turnover_exec": "Avg Turnover (Exec)",
        "avg_turnover_target": "Avg Turnover (Target)",
        "cumulative_return": "Cumulative Return",
    }
    return label_map.get(col, col)


def _delta_col_name(col: str) -> str:
    mapping = {
        "sharpe": "delta_sharpe",
        "max_drawdown": "delta_mdd",
        "avg_turnover": "delta_turnover",
        "avg_turnover_exec": "delta_turnover_exec",
        "avg_turnover_target": "delta_turnover_target",
        "cumulative_return": "delta_cumret",
    }
    return mapping.get(col, f"delta_{col}")


def _fallback_pvalues(values: np.ndarray, *, seed: int = 0, n_perm: int = 4000) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    obs_mean = float(np.mean(values))
    flips = rng.choice([-1.0, 1.0], size=(n_perm, values.size))
    perm_means = (flips * values).mean(axis=1)
    p_perm = float(min(1.0, 2.0 * (np.abs(perm_means) >= abs(obs_mean)).mean()))

    wins = int((values > 0).sum())
    losses = int((values < 0).sum())
    n = wins + losses
    if n == 0:
        p_sign = float("nan")
    else:
        from math import comb

        tail = sum(comb(n, i) for i in range(0, min(wins, losses) + 1))
        p_sign = float(min(1.0, 2.0 * tail / (2**n)))
    return p_perm, p_sign


def compute_paired_diffs(
    metrics_df: pd.DataFrame,
    *,
    baseline_model_type: str = "baseline_sac",
    prl_model_type: str = "prl_sac",
) -> pd.DataFrame:
    df = metrics_df.copy()
    if "period" in df.columns:
        df = df[df["period"] == "test"].copy()
    has_eval_window = "eval_window" in df.columns
    eval_windows = [None]
    if has_eval_window:
        eval_windows = sorted(df["eval_window"].dropna().unique().tolist())
    rows = []
    for eval_window in eval_windows:
        base_df = df[df["model_type"] == baseline_model_type]
        prl_df = df[df["model_type"] == prl_model_type]
        if eval_window is not None:
            base_df = base_df[base_df["eval_window"] == eval_window]
            prl_df = prl_df[prl_df["eval_window"] == eval_window]
        base = base_df.drop_duplicates(subset=["seed"]).set_index("seed")
        prl = prl_df.drop_duplicates(subset=["seed"]).set_index("seed")
        base_seeds = set(base.index.tolist())
        prl_seeds = set(prl.index.tolist())
        missing_in_prl = sorted(base_seeds - prl_seeds)
        missing_in_base = sorted(prl_seeds - base_seeds)
        if missing_in_prl or missing_in_base:
            raise ValueError(
                f"SEED_PAIRING_MISMATCH missing_in_prl={missing_in_prl} missing_in_baseline={missing_in_base}"
            )
        seeds = sorted(base_seeds)
        base = base.loc[seeds]
        prl = prl.loc[seeds]

        metric_cols = _collect_metric_columns(base)
        for seed in seeds:
            row = {"seed": seed}
            if eval_window is not None:
                row["eval_window"] = eval_window
            for col in metric_cols:
                delta_col = _delta_col_name(col)
                row[delta_col] = float(prl.loc[seed, col] - base.loc[seed, col])
            rows.append(row)
    return pd.DataFrame(rows)


def _format_mean_std(values: np.ndarray) -> str:
    if values.size == 0:
        return "nan"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return f"{mean:.3f}±{std:.3f}"


def _format_delta_ci(mean: float, ci_low: float, ci_high: float) -> str:
    return f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def summarize_seed_stats(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    if "period" in df.columns:
        df = df[df["period"] == "test"].copy()
    metric_cols = _collect_metric_columns(df, include_turnover=True)
    group_cols = ["model_type"]
    if "eval_window" in df.columns:
        group_cols = ["eval_window"] + group_cols
    rows = []
    for keys, group in df.groupby(group_cols):
        if len(group_cols) == 2:
            eval_window, model_type = keys
            row = {"eval_window": eval_window, "model_type": model_type, "n_seeds": int(group["seed"].nunique())}
        else:
            model_type = keys
            row = {"model_type": model_type, "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            vals = group[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{col}_std"] = float(np.std(vals, ddof=0)) if vals.size else float("nan")
            if vals.size:
                p25, p50, p75 = np.quantile(vals, [0.25, 0.50, 0.75])
                row[f"{col}_p25"] = float(p25)
                row[f"{col}_median"] = float(p50)
                row[f"{col}_p75"] = float(p75)
                row[f"{col}_iqr"] = float(p75 - p25)
            else:
                row[f"{col}_p25"] = float("nan")
                row[f"{col}_median"] = float("nan")
                row[f"{col}_p75"] = float("nan")
                row[f"{col}_iqr"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _qstats(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    return float(q25), float(q50), float(q75)


def _format_median_iqr(values: np.ndarray) -> str:
    q25, q50, q75 = _qstats(values)
    return f"{q50:.3f} [{q25:.3f}, {q75:.3f}]"


def _build_robust_table(base: pd.DataFrame, prl: pd.DataFrame, diffs: pd.DataFrame, metric_cols: list[str]) -> str:
    preferred = ["sharpe_net_exp", "cumulative_return_net_exp", "max_drawdown_net_exp"]
    selected = [m for m in preferred if m in metric_cols and m in base.columns and m in prl.columns]
    if not selected:
        return ""
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Metric & Baseline (median [P25, P75]) & PRL (median [P25, P75]) & $\\Delta$ (median [P25, P75]) \\\\",
        "\\hline",
    ]
    for col in selected:
        delta_col = _delta_col_name(col)
        if delta_col not in diffs.columns:
            continue
        label = _metric_label(col)
        base_vals = base[col].to_numpy(dtype=np.float64)
        prl_vals = prl[col].to_numpy(dtype=np.float64)
        delta_vals = diffs[delta_col].to_numpy(dtype=np.float64)
        lines.append(
            f"{label} & {_format_median_iqr(base_vals)} & {_format_median_iqr(prl_vals)} & {_format_median_iqr(delta_vals)} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _build_table(
    base: pd.DataFrame, prl: pd.DataFrame, diffs: pd.DataFrame, summary: pd.DataFrame, metric_cols: list[str]
) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Metric & Baseline (mean±std) & PRL (mean±std) & $\\Delta$ (mean, 95\\% CI) \\\\",
        "\\hline",
    ]
    for col in metric_cols:
        delta_col = _delta_col_name(col)
        if col not in base.columns or col not in prl.columns or delta_col not in diffs.columns:
            continue
        label = _metric_label(col)
        base_vals = base[col].to_numpy(dtype=np.float64)
        prl_vals = prl[col].to_numpy(dtype=np.float64)
        delta_match = summary[summary["metric"] == delta_col]
        if delta_match.empty:
            continue
        delta_row = delta_match.iloc[0]
        delta_fmt = _format_delta_ci(
            float(delta_row["mean"]),
            float(delta_row["ci_low"]),
            float(delta_row["ci_high"]),
        )
        lines.append(
            f"{label} & {_format_mean_std(base_vals)} & {_format_mean_std(prl_vals)} & {delta_fmt} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _validate_regime_labels(regime_df: pd.DataFrame) -> None:
    regimes_required = {"low", "mid", "high"}
    missing = []
    for (model_type, seed), group in regime_df.groupby(["model_type", "seed"]):
        regimes = set(group["regime"].tolist())
        if not regimes_required.issubset(regimes):
            missing.append((model_type, seed, sorted(regimes_required - regimes)))
    if missing:
        raise ValueError(f"REGIME_LABELS_MISSING: {missing}")


def compute_regime_seed_summary(regime_df: pd.DataFrame, *, n_boot: int = 2000) -> pd.DataFrame:
    df = regime_df.copy()
    df = df[df["regime"].isin(["low", "mid", "high"])]
    _validate_regime_labels(df)
    turnover_exec_col = "avg_turnover_exec" if "avg_turnover_exec" in df.columns else "avg_turnover"
    metric_map = {
        "sharpe": "sharpe",
        "mdd": "max_drawdown",
        "turnover": turnover_exec_col,
        "turnover_exec": turnover_exec_col,
        "cumret": "cumulative_return",
    }
    if "avg_turnover_target" in df.columns:
        metric_map["turnover_target"] = "avg_turnover_target"
    for col in df.columns:
        if col.startswith(("sharpe_net_", "max_drawdown_net_", "cumulative_return_net_")):
            metric_map[col] = col
    rows = []
    has_eval_window = "eval_window" in df.columns
    group_cols = ["model_type", "regime"]
    if has_eval_window:
        group_cols = ["eval_window"] + group_cols
    for group_keys, group in df.groupby(group_cols):
        if has_eval_window:
            eval_window, model_type, regime = group_keys
            row = {"eval_window": eval_window, "model_type": model_type, "regime": regime, "n_seeds": int(group["seed"].nunique())}
        else:
            model_type, regime = group_keys
            row = {"model_type": model_type, "regime": regime, "n_seeds": int(group["seed"].nunique())}
        for label, col in metric_map.items():
            values = group[col].to_numpy(dtype=np.float64)
            row[f"{label}_mean"] = float(np.mean(values)) if values.size else float("nan")
            row[f"{label}_std"] = float(np.std(values, ddof=0)) if values.size else float("nan")
            ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot)
            row[f"{label}_ci_low"] = ci_low
            row[f"{label}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def compute_regime_paired_diffs(
    regime_df: pd.DataFrame,
    *,
    baseline_model_type: str,
    prl_model_type: str,
) -> pd.DataFrame:
    df = regime_df.copy()
    df = df[df["regime"].isin(["low", "mid", "high"])]
    _validate_regime_labels(df)
    base = df[df["model_type"] == baseline_model_type]
    prl = df[df["model_type"] == prl_model_type]
    metric_cols = _collect_metric_columns(df, include_turnover=True)
    rows = []
    has_eval_window = "eval_window" in df.columns
    eval_windows = [None]
    if has_eval_window:
        eval_windows = sorted(df["eval_window"].dropna().unique().tolist())
    for eval_window in eval_windows:
        base_subset = base if eval_window is None else base[base["eval_window"] == eval_window]
        prl_subset = prl if eval_window is None else prl[prl["eval_window"] == eval_window]
        for regime in ["low", "mid", "high"]:
            base_r = base_subset[base_subset["regime"] == regime].set_index("seed")
            prl_r = prl_subset[prl_subset["regime"] == regime].set_index("seed")
            seeds = sorted(set(base_r.index.tolist()) & set(prl_r.index.tolist()))
            if not seeds:
                continue
            for seed in seeds:
                row = {"seed": int(seed), "regime": regime}
                if eval_window is not None:
                    row["eval_window"] = eval_window
                for col in metric_cols:
                    if col not in base_r.columns or col not in prl_r.columns:
                        continue
                    delta_col = _delta_col_name(col)
                    row[delta_col] = float(prl_r.loc[seed, col] - base_r.loc[seed, col])
                rows.append(row)
    return pd.DataFrame(rows)


def _boxplot_by_regime(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    model_types: list[str],
) -> None:
    regimes = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        data = []
        labels = []
        for model_type in model_types:
            vals = df[(df["model_type"] == model_type) & (df["regime"] == regime)][metric].to_numpy(dtype=np.float64)
            if vals.size:
                data.append(vals)
                labels.append(model_type)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_title(regime)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_regime_delta_sharpe(diffs: pd.DataFrame, output_path: Path) -> None:
    regimes = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        data = diffs[diffs["regime"] == regime]
        ax.axhline(0.0, color="black", linewidth=0.5)
        if not data.empty:
            ax.scatter(data["seed"], data["delta_sharpe"], color="tab:blue")
        ax.set_title(regime)
        ax.set_xlabel("seed")
    axes[0].set_ylabel("delta_sharpe")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def analyze_metrics(
    metrics_path: Path,
    *,
    output_dir: Path,
    baseline_model_type: str,
    prl_model_type: str,
    n_boot: int,
    run_ids: set[str] | None = None,
) -> None:
    df = pd.read_csv(metrics_path)
    if run_ids:
        df = df[df["run_id"].isin(run_ids)].copy()
    # summary per model_type/eval_window (seed-level means/stds), keep mean/std columns
    summary_seed = summarize_seed_stats(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_seed.to_csv(output_dir / "summary_seed_stats.csv", index=False)
    diffs = compute_paired_diffs(df, baseline_model_type=baseline_model_type, prl_model_type=prl_model_type)

    diffs_path = output_dir / "paired_seed_diffs.csv"
    diffs.to_csv(diffs_path, index=False)

    stats_rows = []
    stats = _try_stats()
    group_field = "eval_window" if "eval_window" in diffs.columns else None
    delta_cols = [col for col in diffs.columns if col not in {"seed", "eval_window"}]
    grouped = [(None, diffs)] if group_field is None else diffs.groupby(group_field, dropna=False)
    for key, group in grouped:
        for col in delta_cols:
            values = group[col].to_numpy(dtype=np.float64)
            mean = float(np.mean(values)) if values.size else float("nan")
            std = float(np.std(values, ddof=0)) if values.size else float("nan")
            ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot)
            p_t = float("nan")
            p_w = float("nan")
            wilcoxon_skip = ""
            if stats is not None and values.size >= 2:
                try:
                    p_t = float(stats.ttest_rel(values, np.zeros_like(values)).pvalue)
                except Exception:
                    p_t = float("nan")
                try:
                    if np.allclose(values, 0.0):
                        p_w = 1.0
                        wilcoxon_skip = "all_zero"
                    else:
                        p_w = float(stats.wilcoxon(values).pvalue)
                except Exception as exc:
                    p_w = float("nan")
                    wilcoxon_skip = f"wilcoxon_error:{exc}"
            elif values.size >= 2:
                p_t, p_w = _fallback_pvalues(values, seed=0)
            else:
                wilcoxon_skip = "n_lt_2"
            row = {
                "metric": col,
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value_ttest": p_t,
                "p_value_wilcoxon": p_w,
                "wilcoxon_skipped_reason": wilcoxon_skip,
                "n_seeds": int(values.size),
            }
            if group_field is not None:
                row[group_field] = key
            stats_rows.append(row)

    summary_df = pd.DataFrame(stats_rows)
    summary_path = output_dir / "paired_stats_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if "eval_window" in df.columns and not df["eval_window"].empty:
        target_window = sorted(df["eval_window"].dropna().unique().tolist())[0]
        df = df[df["eval_window"] == target_window]
    base = df[df["model_type"] == baseline_model_type].drop_duplicates(subset=["seed"])
    prl = df[df["model_type"] == prl_model_type].drop_duplicates(subset=["seed"])
    metric_cols = _collect_metric_columns(base)
    table_tex = _build_table(base, prl, diffs, summary_df, metric_cols)
    (output_dir / "table_main.tex").write_text(table_tex)
    robust_table_tex = _build_robust_table(base, prl, diffs, metric_cols)
    if robust_table_tex:
        (output_dir / "table_robust.tex").write_text(robust_table_tex)

    robust_rows = []
    for col in ["sharpe_net_exp", "cumulative_return_net_exp", "max_drawdown_net_exp"]:
        if col not in base.columns or col not in prl.columns:
            continue
        delta_col = _delta_col_name(col)
        if delta_col not in diffs.columns:
            continue
        base_vals = base[col].to_numpy(dtype=np.float64)
        prl_vals = prl[col].to_numpy(dtype=np.float64)
        delta_vals = diffs[delta_col].to_numpy(dtype=np.float64)
        b25, b50, b75 = _qstats(base_vals)
        p25, p50, p75 = _qstats(prl_vals)
        d25, d50, d75 = _qstats(delta_vals)
        robust_rows.append(
            {
                "metric": col,
                "baseline_p25": b25,
                "baseline_median": b50,
                "baseline_p75": b75,
                "baseline_iqr": b75 - b25,
                "prl_p25": p25,
                "prl_median": p50,
                "prl_p75": p75,
                "prl_iqr": p75 - p25,
                "delta_p25": d25,
                "delta_median": d50,
                "delta_p75": d75,
                "delta_iqr": d75 - d25,
            }
        )
    if robust_rows:
        pd.DataFrame(robust_rows).to_csv(output_dir / "robust_stats_summary.csv", index=False)


def analyze_regimes(
    regime_metrics_path: Path,
    *,
    output_dir: Path,
    baseline_model_type: str,
    prl_model_type: str,
    n_boot: int,
    plot: bool,
    run_ids: set[str] | None = None,
) -> None:
    if not regime_metrics_path.exists():
        logging.warning("regime_metrics.csv not found; skipping regime analysis.")
        return
    df = pd.read_csv(regime_metrics_path)
    if run_ids:
        df = df[df["run_id"].isin(run_ids)].copy()
    try:
        summary = compute_regime_seed_summary(df, n_boot=n_boot)
    except ValueError as exc:
        logging.warning("Skipping regime analysis: %s", exc)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "regime_seed_summary.csv", index=False)

    diffs = compute_regime_paired_diffs(
        df,
        baseline_model_type=baseline_model_type,
        prl_model_type=prl_model_type,
    )
    if not diffs.empty:
        diffs.to_csv(output_dir / "regime_paired_diffs.csv", index=False)

    if plot:
        model_types = sorted(df["model_type"].unique().tolist())
        if baseline_model_type in model_types and prl_model_type in model_types:
            model_types = [baseline_model_type, prl_model_type]

        figures_dir = Path("outputs/figures/summary")
        _boxplot_by_regime(df, "sharpe", figures_dir / "regime_sharpe_boxplot.png", "Sharpe by Regime", model_types)
        _boxplot_by_regime(df, "max_drawdown", figures_dir / "regime_mdd_boxplot.png", "Max Drawdown by Regime", model_types)
        turnover_plot_col = "avg_turnover_exec" if "avg_turnover_exec" in df.columns else "avg_turnover"
        turnover_title = "Avg Turnover (Exec) by Regime" if turnover_plot_col == "avg_turnover_exec" else "Avg Turnover by Regime"
        _boxplot_by_regime(df, turnover_plot_col, figures_dir / "turnover_by_regime.png", turnover_title, model_types)
        if "avg_turnover_target" in df.columns:
            _boxplot_by_regime(
                df,
                "avg_turnover_target",
                figures_dir / "turnover_target_by_regime.png",
                "Avg Turnover (Target) by Regime",
                model_types,
            )
        if not diffs.empty:
            _plot_regime_delta_sharpe(diffs, figures_dir / "regime_delta_sharpe_by_seed.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze paper metrics and compute paired seed stats.")
    parser.add_argument("--metrics", type=str, default="outputs/reports/metrics.csv")
    parser.add_argument("--regime-metrics", type=str, default="outputs/reports/regime_metrics.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--baseline-model-type", type=str, default="baseline_sac")
    parser.add_argument("--prl-model-type", type=str, default="prl_sac")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--run-index", type=str, help="Path to run_index.json; filters metrics/regime_metrics to listed run_ids.")
    parser.add_argument(
        "--run-ids",
        type=str,
        help='Comma-separated run_ids to include (applied in conjunction with --run-index if provided).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_ids: set[str] | None = None
    if args.run_index:
        idx_path = Path(args.run_index)
        if not idx_path.exists():
            raise FileNotFoundError(f"run_index not found: {idx_path}")
        import json

        run_index = json.loads(idx_path.read_text())
        if "run_ids" not in run_index:
            raise ValueError("run_index.json missing run_ids")
        run_ids = set(run_index["run_ids"])
    if args.run_ids:
        provided = {rid.strip() for rid in args.run_ids.split(",") if rid.strip()}
        run_ids = provided if run_ids is None else run_ids & provided
    analyze_metrics(
        Path(args.metrics),
        output_dir=Path(args.output_dir),
        baseline_model_type=args.baseline_model_type,
        prl_model_type=args.prl_model_type,
        n_boot=args.bootstrap,
        run_ids=run_ids,
    )
    analyze_regimes(
        Path(args.regime_metrics),
        output_dir=Path(args.output_dir),
        baseline_model_type=args.baseline_model_type,
        prl_model_type=args.prl_model_type,
        n_boot=args.bootstrap,
        plot=not args.no_plots,
        run_ids=run_ids,
    )
    print("Analysis complete.")


if __name__ == "__main__":
    main()
