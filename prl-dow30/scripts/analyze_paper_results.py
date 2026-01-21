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


def compute_paired_diffs(
    metrics_df: pd.DataFrame,
    *,
    baseline_model_type: str = "baseline_sac",
    prl_model_type: str = "prl_sac",
) -> pd.DataFrame:
    df = metrics_df.copy()
    if "period" in df.columns:
        df = df[df["period"] == "test"].copy()
    base = df[df["model_type"] == baseline_model_type].drop_duplicates(subset=["seed"])
    prl = df[df["model_type"] == prl_model_type].drop_duplicates(subset=["seed"])

    base = base.set_index("seed")
    prl = prl.set_index("seed")
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

    diffs = pd.DataFrame(
        {
            "seed": seeds,
            "delta_sharpe": prl["sharpe"].values - base["sharpe"].values,
            "delta_mdd": prl["max_drawdown"].values - base["max_drawdown"].values,
            "delta_turnover": prl["avg_turnover"].values - base["avg_turnover"].values,
            "delta_cumret": prl["cumulative_return"].values - base["cumulative_return"].values,
        }
    )
    return diffs


def _format_mean_std(values: np.ndarray) -> str:
    if values.size == 0:
        return "nan"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return f"{mean:.3f}±{std:.3f}"


def _format_delta_ci(mean: float, ci_low: float, ci_high: float) -> str:
    return f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def _build_table(
    base: pd.DataFrame, prl: pd.DataFrame, diffs: pd.DataFrame, summary: pd.DataFrame
) -> str:
    metrics = [
        ("Sharpe", "sharpe", "delta_sharpe"),
        ("Max Drawdown", "max_drawdown", "delta_mdd"),
        ("Avg Turnover", "avg_turnover", "delta_turnover"),
        ("Cumulative Return", "cumulative_return", "delta_cumret"),
    ]
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Metric & Baseline (mean±std) & PRL (mean±std) & $\\Delta$ (mean, 95\\% CI) \\\\",
        "\\hline",
    ]
    for label, col, delta_col in metrics:
        base_vals = base[col].to_numpy(dtype=np.float64)
        prl_vals = prl[col].to_numpy(dtype=np.float64)
        delta_row = summary[summary["metric"] == delta_col].iloc[0]
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
    metric_map = {
        "sharpe": "sharpe",
        "mdd": "max_drawdown",
        "turnover": "avg_turnover",
        "cumret": "cumulative_return",
    }
    rows = []
    for (model_type, regime), group in df.groupby(["model_type", "regime"]):
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
    rows = []
    for regime in ["low", "mid", "high"]:
        base_r = base[base["regime"] == regime].set_index("seed")
        prl_r = prl[prl["regime"] == regime].set_index("seed")
        seeds = sorted(set(base_r.index.tolist()) & set(prl_r.index.tolist()))
        if not seeds:
            continue
        for seed in seeds:
            rows.append(
                {
                    "seed": int(seed),
                    "regime": regime,
                    "delta_sharpe": float(prl_r.loc[seed, "sharpe"] - base_r.loc[seed, "sharpe"]),
                    "delta_mdd": float(prl_r.loc[seed, "max_drawdown"] - base_r.loc[seed, "max_drawdown"]),
                    "delta_turnover": float(prl_r.loc[seed, "avg_turnover"] - base_r.loc[seed, "avg_turnover"]),
                    "delta_cumret": float(prl_r.loc[seed, "cumulative_return"] - base_r.loc[seed, "cumulative_return"]),
                }
            )
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
            ax.boxplot(data, labels=labels, showfliers=False)
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
) -> None:
    df = pd.read_csv(metrics_path)
    diffs = compute_paired_diffs(df, baseline_model_type=baseline_model_type, prl_model_type=prl_model_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    diffs_path = output_dir / "paired_seed_diffs.csv"
    diffs.to_csv(diffs_path, index=False)

    stats_rows = []
    stats = _try_stats()
    for col in ["delta_sharpe", "delta_mdd", "delta_turnover", "delta_cumret"]:
        values = diffs[col].to_numpy(dtype=np.float64)
        mean = float(np.mean(values)) if values.size else float("nan")
        std = float(np.std(values, ddof=0)) if values.size else float("nan")
        ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot)
        p_t = float("nan")
        p_w = float("nan")
        if stats is not None and values.size >= 2:
            try:
                p_t = float(stats.ttest_rel(values, np.zeros_like(values)).pvalue)
            except Exception:
                p_t = float("nan")
            try:
                p_w = float(stats.wilcoxon(values).pvalue)
            except Exception:
                p_w = float("nan")
        stats_rows.append(
            {
                "metric": col,
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value_ttest": p_t,
                "p_value_wilcoxon": p_w,
                "n_seeds": int(values.size),
            }
        )

    summary_df = pd.DataFrame(stats_rows)
    summary_path = output_dir / "paired_stats_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    base = df[df["model_type"] == baseline_model_type].drop_duplicates(subset=["seed"])
    prl = df[df["model_type"] == prl_model_type].drop_duplicates(subset=["seed"])
    table_tex = _build_table(base, prl, diffs, summary_df)
    (output_dir / "table_main.tex").write_text(table_tex)


def analyze_regimes(
    regime_metrics_path: Path,
    *,
    output_dir: Path,
    baseline_model_type: str,
    prl_model_type: str,
    n_boot: int,
    plot: bool,
) -> None:
    if not regime_metrics_path.exists():
        logging.warning("regime_metrics.csv not found; skipping regime analysis.")
        return
    df = pd.read_csv(regime_metrics_path)
    summary = compute_regime_seed_summary(df, n_boot=n_boot)
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
        _boxplot_by_regime(df, "avg_turnover", figures_dir / "turnover_by_regime.png", "Avg Turnover by Regime", model_types)
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
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_metrics(
        Path(args.metrics),
        output_dir=Path(args.output_dir),
        baseline_model_type=args.baseline_model_type,
        prl_model_type=args.prl_model_type,
        n_boot=args.bootstrap,
    )
    analyze_regimes(
        Path(args.regime_metrics),
        output_dir=Path(args.output_dir),
        baseline_model_type=args.baseline_model_type,
        prl_model_type=args.prl_model_type,
        n_boot=args.bootstrap,
        plot=not args.no_plots,
    )
    print("Analysis complete.")


if __name__ == "__main__":
    main()
