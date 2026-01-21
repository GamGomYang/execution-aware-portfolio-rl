import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


FIG_NAMES = [
    "train_losses.png",
    "alpha_beta_emergency.png",
    "equity_curve.png",
    "equity_by_regime.png",
    "turnover_by_regime.png",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Step4 figures + report for a run_id.")
    parser.add_argument("--run-id", type=str, help="Run ID to render.")
    parser.add_argument("--metadata", type=str, help="Path to run_metadata_{run_id}.json.")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Base outputs directory.")
    return parser.parse_args()


def _load_metadata(run_id: str | None, metadata_path: str | None, reports_dir: Path) -> dict:
    if metadata_path:
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"metadata not found: {path}")
        data = json.loads(path.read_text())
        return data
    if not run_id:
        raise ValueError("Either --run-id or --metadata is required.")
    path = reports_dir / f"run_metadata_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"run_metadata not found: {path}")
    return json.loads(path.read_text())


def _resolve_paths(run_id: str, metadata: dict, outputs_dir: Path) -> dict:
    reports_dir = outputs_dir / "reports"
    logs_dir = outputs_dir / "logs"
    figs_dir = outputs_dir / "figures" / run_id

    artifacts = metadata.get("artifact_paths") or metadata.get("artifacts") or {}
    report_paths = metadata.get("report_paths", {})

    train_log_path = Path(artifacts.get("train_log_path", logs_dir / f"train_{run_id}.csv"))
    trace_path = Path(report_paths.get("trace_path", reports_dir / f"trace_{run_id}.parquet"))
    thresholds_path = Path(report_paths.get("regime_thresholds_path", reports_dir / f"regime_thresholds_{run_id}.json"))
    regime_metrics_path = Path(report_paths.get("regime_metrics_path", reports_dir / "regime_metrics.csv"))
    report_path = Path(report_paths.get("step4_report_path", reports_dir / f"step4_report_{run_id}.md"))
    figures_dir = Path(report_paths.get("figures_dir", figs_dir))

    return {
        "reports_dir": reports_dir,
        "figures_dir": figures_dir,
        "train_log_path": train_log_path,
        "trace_path": trace_path,
        "thresholds_path": thresholds_path,
        "regime_metrics_path": regime_metrics_path,
        "report_path": report_path,
    }


def _ensure_figure_dir(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)


def _save_placeholder(fig_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _plot_train_losses(train_df: pd.DataFrame, fig_path: Path) -> None:
    if train_df is None or train_df.empty:
        _save_placeholder(fig_path, "No train log data")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_df["timesteps"], train_df["actor_loss"], label="actor_loss")
    ax.plot(train_df["timesteps"], train_df["critic_loss"], label="critic_loss")
    if "entropy_loss" in train_df.columns:
        ax.plot(train_df["timesteps"], train_df["entropy_loss"], label="entropy_loss")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _plot_alpha_beta_emergency(train_df: pd.DataFrame, fig_path: Path) -> None:
    required = {"alpha_raw_mean", "alpha_clamped_mean", "beta_effective_mean", "emergency_rate"}
    if train_df is None or train_df.empty or not required.issubset(train_df.columns):
        _save_placeholder(fig_path, "PRL diagnostics not available")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_df["timesteps"], train_df["alpha_raw_mean"], label="alpha_raw_mean")
    ax.plot(train_df["timesteps"], train_df["alpha_clamped_mean"], label="alpha_clamped_mean")
    ax.plot(train_df["timesteps"], train_df["beta_effective_mean"], label="beta_effective_mean")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("alpha / beta_effective")
    ax2 = ax.twinx()
    ax2.plot(train_df["timesteps"], train_df["emergency_rate"], color="tab:red", label="emergency_rate")
    ax2.set_ylabel("emergency_rate")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _plot_equity_curve(trace_df: pd.DataFrame, fig_path: Path) -> None:
    if trace_df is None or trace_df.empty:
        _save_placeholder(fig_path, "No trace data")
        return
    returns = trace_df["portfolio_return"].fillna(0.0).to_numpy()
    equity = np.cumprod(1.0 + returns)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trace_df["date"], equity, label="equity")
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _plot_equity_by_regime(trace_df: pd.DataFrame, fig_path: Path) -> None:
    if trace_df is None or trace_df.empty or "regime" not in trace_df.columns:
        _save_placeholder(fig_path, "Regime data not available")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    returns = trace_df["portfolio_return"].fillna(0.0).to_numpy()
    for regime in ["low", "mid", "high"]:
        mask = trace_df["regime"] == regime
        masked_returns = np.where(mask, returns, 0.0)
        equity = np.cumprod(1.0 + masked_returns)
        ax.plot(trace_df["date"], equity, label=regime)
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _plot_turnover_by_regime(trace_df: pd.DataFrame, fig_path: Path) -> None:
    if trace_df is None or trace_df.empty or "regime" not in trace_df.columns:
        _save_placeholder(fig_path, "Regime data not available")
        return
    means = trace_df.groupby("regime")["turnover"].mean()
    regimes = ["low", "mid", "high"]
    values = [means.get(regime, 0.0) for regime in regimes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(regimes, values, color="tab:blue")
    ax.set_ylabel("avg_turnover")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _last_value(series: pd.Series) -> float | None:
    if series is None:
        return None
    series = series.dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _render_regime_table(regime_df: pd.DataFrame) -> str:
    if regime_df is None or regime_df.empty:
        return "No regime metrics available."
    cols = ["regime", "cumulative_return", "sharpe", "max_drawdown", "avg_turnover", "total_reward"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in regime_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("regime", "")),
                    f"{row.get('cumulative_return', 0.0):.6f}",
                    f"{row.get('sharpe', 0.0):.6f}",
                    f"{row.get('max_drawdown', 0.0):.6f}",
                    f"{row.get('avg_turnover', 0.0):.6f}",
                    f"{row.get('total_reward', 0.0):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    reports_dir = outputs_dir / "reports"

    metadata = _load_metadata(args.run_id, args.metadata, reports_dir)
    run_id = args.run_id or metadata.get("run_id")
    if not run_id:
        raise ValueError("run_id missing from metadata.")

    paths = _resolve_paths(run_id, metadata, outputs_dir)
    figures_dir = paths["figures_dir"]
    _ensure_figure_dir(figures_dir)

    train_df = None
    if paths["train_log_path"].exists():
        train_df = pd.read_csv(paths["train_log_path"])

    trace_df = None
    if paths["trace_path"].exists():
        trace_df = pd.read_parquet(paths["trace_path"])
        if "date" in trace_df.columns:
            trace_df["date"] = pd.to_datetime(trace_df["date"])

    thresholds = {}
    if paths["thresholds_path"].exists():
        thresholds = json.loads(paths["thresholds_path"].read_text())

    regime_df = None
    if paths["regime_metrics_path"].exists():
        raw = pd.read_csv(paths["regime_metrics_path"])
        model_type = metadata.get("model_type")
        label = f"{model_type}_sac" if model_type in {"baseline", "prl"} else model_type
        regime_df = raw[(raw["run_id"] == run_id) & (raw["model_type"] == label)]
        regime_df = regime_df[regime_df["regime"].isin(["low", "mid", "high"])]

    model_type = metadata.get("model_type")
    label = f"{model_type}_sac" if model_type in {"baseline", "prl"} else model_type
    if trace_df is not None:
        trace_df = trace_df[trace_df["model_type"] == label]

    _plot_train_losses(train_df, figures_dir / "train_losses.png")
    _plot_alpha_beta_emergency(train_df, figures_dir / "alpha_beta_emergency.png")
    _plot_equity_curve(trace_df, figures_dir / "equity_curve.png")
    _plot_equity_by_regime(trace_df, figures_dir / "equity_by_regime.png")
    _plot_turnover_by_regime(trace_df, figures_dir / "turnover_by_regime.png")

    final_actor_loss = _last_value(train_df["actor_loss"]) if train_df is not None else None
    final_critic_loss = _last_value(train_df["critic_loss"]) if train_df is not None else None
    final_entropy_loss = _last_value(train_df.get("entropy_loss")) if train_df is not None else None
    emergency_rate = train_df.get("emergency_rate") if train_df is not None else None
    emergency_mean = float(emergency_rate.mean()) if emergency_rate is not None else None
    emergency_max = float(emergency_rate.max()) if emergency_rate is not None else None

    lines = []
    lines.append(f"# Step4 Report: {run_id}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- model_type: {metadata.get('model_type')}")
    lines.append(f"- seed: {metadata.get('seed')}")
    lines.append(f"- created_at: {metadata.get('created_at')}")
    lines.append(f"- config_hash: {metadata.get('config_hash')}")
    lines.append("")
    lines.append("## Final Loss Stats")
    lines.append(f"- actor_loss_final: {final_actor_loss}")
    lines.append(f"- critic_loss_final: {final_critic_loss}")
    lines.append(f"- entropy_loss_final: {final_entropy_loss}")
    lines.append("")
    lines.append("## Emergency Rate Summary")
    lines.append(f"- emergency_rate_mean: {emergency_mean}")
    lines.append(f"- emergency_rate_max: {emergency_max}")
    lines.append("")
    lines.append("## Regime Thresholds")
    lines.append(f"- q33: {thresholds.get('q33')}")
    lines.append(f"- q66: {thresholds.get('q66')}")
    lines.append("")
    lines.append("## Regime Metrics (low/mid/high)")
    lines.append(_render_regime_table(regime_df))
    lines.append("")
    lines.append("## Figures")
    for name in FIG_NAMES:
        lines.append(f"- {figures_dir / name}")
    lines.append("")

    report_path = paths["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
