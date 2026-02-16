import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    return pd.read_parquet(path)


def _gross_to_net_drop(metrics_df: pd.DataFrame, model_types: Iterable[str]) -> pd.DataFrame:
    rows = []
    for model in model_types:
        subset = metrics_df[metrics_df["model_type"] == model]
        if subset.empty:
            continue
        row = subset.iloc[0]
        gross = row.get("cumulative_return", np.nan)
        net = row.get("cumulative_return_net_exp", np.nan)
        rows.append({"model_type": model, "gross_cumret": gross, "net_cumret": net, "drop_gross_to_net": gross - net})
    return pd.DataFrame(rows)


def _turnover_distribution(trace_df: pd.DataFrame, model_types: Iterable[str]) -> pd.DataFrame:
    records = []
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    for model in model_types:
        df = trace_df[trace_df["model_type"] == model]
        if df.empty or "turnover" not in df.columns:
            continue
        vals = pd.to_numeric(df["turnover"], errors="coerce").dropna()
        for q in quantiles:
            records.append({"model_type": model, "quantile": q, "turnover": float(vals.quantile(q)) if not vals.empty else np.nan})
    return pd.DataFrame(records)


def _regime_breakdown(regime_df: pd.DataFrame, model_types: Iterable[str]) -> pd.DataFrame:
    cols = [
        "cumulative_return",
        "cumulative_return_net_exp",
        "sharpe",
        "sharpe_net_exp",
        "avg_turnover",
        "total_turnover",
    ]
    rows = []
    for model in model_types:
        for regime in ["low", "mid", "high"]:
            subset = regime_df[(regime_df["model_type"] == model) & (regime_df["regime"] == regime)]
            if subset.empty:
                continue
            row = {"model_type": model, "regime": regime}
            for col in cols:
                if col in subset.columns:
                    row[col] = float(subset[col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _sharpe_improvement(metrics_df: pd.DataFrame, model_types: Iterable[str], target: float = 2.0) -> pd.DataFrame:
    rows = []
    for model in model_types:
        subset = metrics_df[metrics_df["model_type"] == model]
        if subset.empty:
            continue
        row = subset.iloc[0]
        sharpe = float(row.get("sharpe_net_exp", np.nan))
        steps = float(row.get("steps", np.nan))
        cumret_net = float(row.get("cumulative_return_net_exp", np.nan))
        if steps and not np.isnan(sharpe) and sharpe != 0:
            daily_mean = cumret_net / steps
            daily_std = (daily_mean * np.sqrt(252)) / sharpe
            needed_mean = (target * daily_std) / np.sqrt(252)
            improvement = needed_mean - daily_mean
        else:
            improvement = float("nan")
        rows.append({"model_type": model, "current_sharpe_net_exp": sharpe, "delta_mean_needed_for_sharpe_2": improvement})
    return pd.DataFrame(rows)


def render_report(output_dir: Path, gross_net_df: pd.DataFrame, regime_df: pd.DataFrame, sharpe_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Diagnosis Decomposition")
    lines.append("")
    if not gross_net_df.empty:
        lines.append("## Gross → Net_exp Drop")
        for _, row in gross_net_df.iterrows():
            lines.append(
                f"- {row['model_type']}: gross {row['gross_cumret']:.4f} -> net_exp {row['net_cumret']:.4f} (drop {row['drop_gross_to_net']:.4f})"
            )
        lines.append("")
    if not regime_df.empty:
        lines.append("## Regime Breakdown (net/gross)")
        for _, row in regime_df.iterrows():
            lines.append(
                f"- {row['model_type']} [{row['regime']}]: cumret {row.get('cumulative_return', np.nan):.4f}, net_exp {row.get('cumulative_return_net_exp', np.nan):.4f}, sharpe {row.get('sharpe', np.nan):.4f}, sharpe_net_exp {row.get('sharpe_net_exp', np.nan):.4f}, avg_turnover {row.get('avg_turnover', np.nan):.4f}"
            )
        lines.append("")
    if not sharpe_df.empty:
        lines.append("## Sharpe 2.0 Gap (net_exp)")
        for _, row in sharpe_df.iterrows():
            lines.append(
                f"- {row['model_type']}: current {row['current_sharpe_net_exp']:.3f}, daily mean improvement needed ~ {row['delta_mean_needed_for_sharpe_2']:.6f}"
            )
        lines.append("")
    if not lines:
        lines.append("No data available for diagnosis.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnosis_decomposition.md").write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate diagnosis decomposition report.")
    parser.add_argument("--metrics", type=str, default="outputs/reports/metrics.csv")
    parser.add_argument("--regime-metrics", type=str, default="outputs/reports/regime_metrics.csv")
    parser.add_argument("--trace", type=str, help="Optional trace parquet for turnover distribution.")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--baseline-model-type", type=str, default="baseline_sac")
    parser.add_argument("--prl-model-type", type=str, default="prl_sac")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_df = _safe_load_csv(Path(args.metrics))
    regime_df = _safe_load_csv(Path(args.regime_metrics))
    model_types = [args.baseline_model_type, args.prl_model_type]

    gross_net_df = _gross_to_net_drop(metrics_df, model_types) if not metrics_df.empty else pd.DataFrame()

    trace_df = pd.DataFrame()
    if args.trace:
        trace_df = _load_trace(Path(args.trace))
    if trace_df.empty:
        turnover_df = _turnover_distribution(regime_df, model_types)
    else:
        turnover_df = _turnover_distribution(trace_df, model_types)

    regime_breakdown_df = _regime_breakdown(regime_df, model_types) if not regime_df.empty else pd.DataFrame()
    sharpe_df = _sharpe_improvement(metrics_df, model_types) if not metrics_df.empty else pd.DataFrame()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not turnover_df.empty:
        turnover_df.to_csv(out_dir / "turnover_distribution.csv", index=False)
    if not regime_breakdown_df.empty:
        regime_breakdown_df.to_csv(out_dir / "regime_breakdown_net.csv", index=False)
    render_report(out_dir, gross_net_df, regime_breakdown_df, sharpe_df)


if __name__ == "__main__":
    main()
