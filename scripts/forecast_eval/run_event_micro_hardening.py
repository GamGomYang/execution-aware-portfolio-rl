#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

matplotlib.rcParams["ps.useafm"] = True
matplotlib.rcParams["font.family"] = "serif"


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(SCRIPT_DIR), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_same_interface_rank_summary import build_domain_rank_summary, write_summary_outputs  # noqa: E402
from revision_round_20260423 import (  # noqa: E402
    ANALYSIS_ADDITIONS_DIR,
    LOGICAL_CANONICAL_ROOT,
    NEW_RERUNS_DIR,
    PHYSICAL_STORAGE_ROOT,
    STORY_REVISION_DIR,
    EVENT_MICRO_REPORT_LABELS,
    build_q2_from_seed_metrics,
    compact_selection_summary,
    ensure_dir,
    ensure_logical_alias,
    friction_row,
    logical_root_relative,
    model_label,
    report_selection_table,
    physical_root_relative,
    repo_relative,
    write_json,
    write_markdown,
    write_table_bundle,
)


RUN_EVENT_MICRO_SCRIPT = SCRIPT_DIR / "run_event_micro.py"
HARDENING_CONFIG_DIR = ROOT / "configs" / "event_micro_revision_round_20260423" / "hardening"
HARDENING_ROOT = NEW_RERUNS_DIR / "event_micro_hardening"
CANONICAL_RUN_DIR = HARDENING_ROOT / "fixed_threshold_tau055_seed100"
TAU050_RUN_DIR = HARDENING_ROOT / "fixed_threshold_tau050_seed100"
HYSTERESIS_RUN_DIR = HARDENING_ROOT / "hysteresis_tau055_delta005_seed100"
HARDENING_ANALYSIS_DIR = ANALYSIS_ADDITIONS_DIR / "event_micro_hardening"
HARDENING_STORY_DIR = STORY_REVISION_DIR / "event_micro_hardening"

CANONICAL_CONFIG = HARDENING_CONFIG_DIR / "event_micro_tau055_seed100.yaml"
TAU050_CONFIG = HARDENING_CONFIG_DIR / "event_micro_tau050_seed100.yaml"
HYSTERESIS_CONFIG = HARDENING_CONFIG_DIR / "event_micro_hysteresis_tau055_delta005_seed100.yaml"

REPORT_DIR = ROOT / "results"
REPORT_TABLES_DIR = REPORT_DIR / "tables"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"
REPORT_MANIFEST = REPORT_DIR / "report_card_manifest.md"

SYNTHETIC_Q2_RAW = ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q2_diff_forecasts_same_interface.csv"
INVENTORY_Q2_SEED = (
    ROOT
    / "outputs"
    / "forecast_eval"
    / "inventory_step4_seed_stability_locked"
    / "inventory_v2_seed_stability_q2_selection_seed_level.csv"
)
LOAD_FOLLOWING_Q2_RAW = (
    ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_promotion_locked" / "q2_diff_forecasts_same_interface.csv"
)

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260423


@dataclass(frozen=True)
class VariantResult:
    name: str
    interface_id: str
    config_path: Path
    output_dir: Path
    raw_df: pd.DataFrame
    seed_metrics_df: pd.DataFrame
    outputs: dict[str, pd.DataFrame]

    @property
    def selection_summary(self) -> pd.DataFrame:
        return self.outputs["selection_summary_by_friction"].copy()

    @property
    def seed_selection(self) -> pd.DataFrame:
        return self.outputs["seed_level_selection_stats"].copy()

    @property
    def rank_summary(self) -> pd.DataFrame:
        return self.outputs["rank_correlation_by_friction"].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 7-9 event-micro hardening artifacts.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing raw outputs when present.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def logical_display_path(path: Path) -> str:
    return str(logical_root_relative(path))


def physical_display_path(path: Path) -> str:
    return str(physical_root_relative(path))


def bootstrap_interval(values: np.ndarray, *, statistic: str) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    n = values.size
    for idx in range(BOOTSTRAP_SAMPLES):
        sample = values[rng.integers(0, n, size=n)]
        if statistic == "mean":
            draws[idx] = float(np.mean(sample))
        elif statistic == "median":
            draws[idx] = float(np.median(sample))
        else:
            raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def human_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def human_pvalue(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def run_variant(
    *,
    name: str,
    config_path: Path,
    output_dir: Path,
    expected_interface_id: str,
    skip_existing: bool,
) -> VariantResult:
    ensure_dir(output_dir)
    raw_path = output_dir / "q2_diff_forecasts_same_interface.csv"
    seed_metrics_path = output_dir / "seed_level_metrics.csv"
    if not (skip_existing and raw_path.exists() and seed_metrics_path.exists()):
        run_command(
            [
                sys.executable,
                str(RUN_EVENT_MICRO_SCRIPT),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--skip-summary-refresh",
            ]
        )
    raw_df = pd.read_csv(raw_path)
    seed_metrics_df = pd.read_csv(seed_metrics_path)
    outputs, _meta = build_domain_rank_summary(
        raw_df,
        domain="event_micro",
        expected_interface_id=expected_interface_id,
    )
    write_summary_outputs(outputs, output_dir / "derived")
    return VariantResult(
        name=name,
        interface_id=expected_interface_id,
        config_path=config_path,
        output_dir=output_dir,
        raw_df=raw_df,
        seed_metrics_df=seed_metrics_df,
        outputs=outputs,
    )


def build_logloss_variant(canonical: VariantResult) -> VariantResult:
    logloss_q2_df = build_q2_from_seed_metrics(
        canonical.seed_metrics_df,
        forecast_metric_column="logloss",
        scenario_id="event_micro_hardening_seed100_logloss",
        interface_id=canonical.interface_id,
    )
    output_dir = canonical.output_dir / "derived_logloss"
    ensure_dir(output_dir)
    outputs, _meta = build_domain_rank_summary(
        logloss_q2_df,
        domain="event_micro",
        expected_interface_id=canonical.interface_id,
    )
    write_summary_outputs(outputs, output_dir)
    return VariantResult(
        name="logloss_rerank",
        interface_id=canonical.interface_id,
        config_path=canonical.config_path,
        output_dir=output_dir,
        raw_df=logloss_q2_df,
        seed_metrics_df=canonical.seed_metrics_df.copy(),
        outputs=outputs,
    )


def agreement_pattern(summary: pd.DataFrame) -> tuple[str, bool]:
    compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
    zero_agreement = float(friction_row(compact, 0.0)["agreement_rate"])
    mid_agreement = float(friction_row(compact, 0.5)["agreement_rate"])
    high_agreement = float(friction_row(compact, 1.0)["agreement_rate"])

    if zero_agreement > mid_agreement > high_agreement:
        return "strict_monotone_decline", True
    if zero_agreement > mid_agreement and zero_agreement > high_agreement:
        return "headline_supportive_decline", True
    if mid_agreement < zero_agreement and high_agreement <= zero_agreement - 0.40:
        return "strong_drop_fallback", True
    return "insufficient_decline", False


def evaluate_step7(summary: pd.DataFrame) -> dict[str, object]:
    compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
    zero_row = friction_row(compact, 0.0)
    mid_row = friction_row(compact, 0.5)
    high_row = friction_row(compact, 1.0)
    agreement_label, agreement_ok = agreement_pattern(summary)
    passed = bool(
        float(mid_row["deployed_suboptimal_seed_fraction"]) >= 0.60
        and float(high_row["deployed_suboptimal_seed_fraction"]) >= 0.90
        and float(mid_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
        and float(mid_row["median_deployed_gap_of_forecast_selected"]) > 0.0
        and float(high_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
        and float(high_row["median_deployed_gap_of_forecast_selected"]) > 0.0
        and agreement_ok
    )
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "agreement_pattern": agreement_label,
        "zero_agreement": float(zero_row["agreement_rate"]),
        "mid_agreement": float(mid_row["agreement_rate"]),
        "high_agreement": float(high_row["agreement_rate"]),
        "mid_suboptimal_share": float(mid_row["deployed_suboptimal_seed_fraction"]),
        "high_suboptimal_share": float(high_row["deployed_suboptimal_seed_fraction"]),
        "mid_mean_gap": float(mid_row["mean_deployed_gap_of_forecast_selected"]),
        "mid_median_gap": float(mid_row["median_deployed_gap_of_forecast_selected"]),
        "high_mean_gap": float(high_row["mean_deployed_gap_of_forecast_selected"]),
        "high_median_gap": float(high_row["median_deployed_gap_of_forecast_selected"]),
        "winner_descriptives": {
            "zero_forecast_winner": str(zero_row["most_frequent_forecast_best"]),
            "zero_deployed_winner": str(zero_row["most_frequent_deployed_best"]),
            "mid_forecast_winner": str(mid_row["most_frequent_forecast_best"]),
            "mid_deployed_winner": str(mid_row["most_frequent_deployed_best"]),
            "high_forecast_winner": str(high_row["most_frequent_forecast_best"]),
            "high_deployed_winner": str(high_row["most_frequent_deployed_best"]),
        },
    }


def evaluate_hysteresis(summary: pd.DataFrame) -> dict[str, object]:
    compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
    zero_row = friction_row(compact, 0.0)
    mid_row = friction_row(compact, 0.5)
    high_row = friction_row(compact, 1.0)
    agreement_label, agreement_ok = agreement_pattern(summary)
    strong = bool(
        float(mid_row["deployed_suboptimal_seed_fraction"]) > 0.50
        and float(high_row["deployed_suboptimal_seed_fraction"]) > 0.50
        and float(mid_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
        and float(mid_row["median_deployed_gap_of_forecast_selected"]) > 0.0
        and float(high_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
        and float(high_row["median_deployed_gap_of_forecast_selected"]) > 0.0
        and agreement_ok
    )
    weak = bool(
        float(high_row["deployed_suboptimal_seed_fraction"]) > 0.50
        and (
            (
                float(mid_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
                and float(mid_row["median_deployed_gap_of_forecast_selected"]) > 0.0
            )
            or (
                float(high_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
                and float(high_row["median_deployed_gap_of_forecast_selected"]) > 0.0
            )
        )
        and (
            float(mid_row["agreement_rate"]) <= float(zero_row["agreement_rate"]) - 0.10
            or float(high_row["agreement_rate"]) <= float(zero_row["agreement_rate"]) - 0.10
        )
    )
    if strong:
        category = "strong_pass"
    elif weak:
        category = "weak_pass"
    else:
        category = "fail"
    return {
        "category": category,
        "agreement_pattern": agreement_label,
        "zero_agreement": float(zero_row["agreement_rate"]),
        "mid_agreement": float(mid_row["agreement_rate"]),
        "high_agreement": float(high_row["agreement_rate"]),
        "mid_suboptimal_share": float(mid_row["deployed_suboptimal_seed_fraction"]),
        "high_suboptimal_share": float(high_row["deployed_suboptimal_seed_fraction"]),
        "mid_mean_gap": float(mid_row["mean_deployed_gap_of_forecast_selected"]),
        "mid_median_gap": float(mid_row["median_deployed_gap_of_forecast_selected"]),
        "high_mean_gap": float(high_row["mean_deployed_gap_of_forecast_selected"]),
        "high_median_gap": float(high_row["median_deployed_gap_of_forecast_selected"]),
        "winner_descriptives": {
            "zero_forecast_winner": str(zero_row["most_frequent_forecast_best"]),
            "zero_deployed_winner": str(zero_row["most_frequent_deployed_best"]),
            "mid_forecast_winner": str(mid_row["most_frequent_forecast_best"]),
            "mid_deployed_winner": str(mid_row["most_frequent_deployed_best"]),
            "high_forecast_winner": str(high_row["most_frequent_forecast_best"]),
            "high_deployed_winner": str(high_row["most_frequent_deployed_best"]),
        },
    }


def main_table(summary: pd.DataFrame) -> pd.DataFrame:
    compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
    rows = []
    for row in compact.itertuples(index=False):
        rows.append(
            {
                "Friction": f"{float(row.friction_level):.2f}",
                "Forecast-side winner": model_label(str(row.most_frequent_forecast_best), EVENT_MICRO_REPORT_LABELS),
                "Deployed winner": model_label(str(row.most_frequent_deployed_best), EVENT_MICRO_REPORT_LABELS),
                "Agreement rate": f"{float(row.agreement_rate):.2f}",
                "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
                "Deployed-suboptimal seeds / total": str(row.deployed_suboptimal_seeds_over_total),
            }
        )
    return pd.DataFrame(rows)


def full_table(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.sort_values("friction_level", kind="mergesort").reset_index(drop=True)
    rows = []
    for row in work.itertuples(index=False):
        rows.append(
            {
                "Friction": f"{float(row.friction_level):.2f}",
                "Forecast-side winner": model_label(str(row.most_frequent_forecast_best), EVENT_MICRO_REPORT_LABELS),
                "Deployed winner": model_label(str(row.most_frequent_deployed_best), EVENT_MICRO_REPORT_LABELS),
                "Agreement rate": f"{float(row.agreement_rate):.2f}",
                "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
            }
        )
    return pd.DataFrame(rows)


def threshold_robustness_table(canonical: pd.DataFrame, tau050: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold_name, summary in [("tau=0.55", canonical), ("tau=0.50", tau050)]:
        compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
        for row in compact.itertuples(index=False):
            rows.append(
                {
                    "Threshold": threshold_name,
                    "Friction": f"{float(row.friction_level):.2f}",
                    "Forecast-side winner": model_label(str(row.most_frequent_forecast_best), EVENT_MICRO_REPORT_LABELS),
                    "Deployed winner": model_label(str(row.most_frequent_deployed_best), EVENT_MICRO_REPORT_LABELS),
                    "Agreement rate": f"{float(row.agreement_rate):.2f}",
                    "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
                    "Median deployed gap": f"{float(row.median_deployed_gap_of_forecast_selected):.3f}",
                    "Deployed-suboptimal seeds / total": str(row.deployed_suboptimal_seeds_over_total),
                }
            )
    return pd.DataFrame(rows)


def logloss_table(summary: pd.DataFrame) -> pd.DataFrame:
    compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
    rows = []
    for row in compact.itertuples(index=False):
        rows.append(
            {
                "Friction": f"{float(row.friction_level):.2f}",
                "Forecast-side winner": model_label(str(row.most_frequent_forecast_best), EVENT_MICRO_REPORT_LABELS),
                "Deployed winner": model_label(str(row.most_frequent_deployed_best), EVENT_MICRO_REPORT_LABELS),
                "Agreement rate": f"{float(row.agreement_rate):.2f}",
                "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
                "Median deployed gap": f"{float(row.median_deployed_gap_of_forecast_selected):.3f}",
                "Deployed-suboptimal seeds / total": str(row.deployed_suboptimal_seeds_over_total),
            }
        )
    return pd.DataFrame(rows)


def hysteresis_table(summary: pd.DataFrame) -> pd.DataFrame:
    compact = compact_selection_summary(summary, frictions=(0.0, 0.5, 1.0))
    rows = []
    for row in compact.itertuples(index=False):
        rows.append(
            {
                "Interface": "Hysteresis threshold",
                "Friction": f"{float(row.friction_level):.2f}",
                "Forecast-side winner": model_label(str(row.most_frequent_forecast_best), EVENT_MICRO_REPORT_LABELS),
                "Deployed winner": model_label(str(row.most_frequent_deployed_best), EVENT_MICRO_REPORT_LABELS),
                "Agreement rate": f"{float(row.agreement_rate):.2f}",
                "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
                "Median deployed gap": f"{float(row.median_deployed_gap_of_forecast_selected):.3f}",
                "Deployed-suboptimal seeds / total": str(row.deployed_suboptimal_seeds_over_total),
            }
        )
    return pd.DataFrame(rows)


def majority_test_row(domain_label: str, friction: float, seed_df: pd.DataFrame) -> dict[str, str]:
    row_df = seed_df.loc[np.isclose(seed_df["friction_level"], friction, atol=1e-12)].copy()
    k = int(row_df["selection_disagreement_flag"].sum())
    n = int(row_df["seed"].nunique())
    result = binomtest(k, n, p=0.5, alternative="greater")
    ci = result.proportion_ci(confidence_level=0.95, method="exact")
    return {
        "Domain": domain_label,
        "Friction": f"{friction:.2f}",
        "Deployed-suboptimal seeds / total": f"{k}/{n}",
        "Share": f"{k / n:.2f}",
        "95% exact CI": human_ci(float(ci.low), float(ci.high)),
        "One-sided binomial p": human_pvalue(float(result.pvalue)),
    }


def gap_ci_row(domain_label: str, friction: float, seed_df: pd.DataFrame) -> dict[str, str]:
    row_df = seed_df.loc[np.isclose(seed_df["friction_level"], friction, atol=1e-12)].copy()
    gaps = row_df["deployed_gap_of_forecast_selected"].to_numpy(dtype=float)
    mean_gap = float(np.mean(gaps))
    median_gap = float(np.median(gaps))
    mean_lo, mean_hi = bootstrap_interval(gaps, statistic="mean")
    median_lo, median_hi = bootstrap_interval(gaps, statistic="median")
    return {
        "Domain": domain_label,
        "Friction": f"{friction:.2f}",
        "Mean deployed gap": f"{mean_gap:.3f}",
        "Mean gap 95% bootstrap CI": human_ci(mean_lo, mean_hi),
        "Median deployed gap": f"{median_gap:.3f}",
        "Median gap 95% bootstrap CI": human_ci(median_lo, median_hi),
    }


def interface_majority_test_row(interface_label: str, friction: float, seed_df: pd.DataFrame) -> dict[str, str]:
    row = majority_test_row(interface_label, friction, seed_df)
    row["Interface"] = row.pop("Domain")
    return row


def interface_gap_ci_row(interface_label: str, friction: float, seed_df: pd.DataFrame) -> dict[str, str]:
    row = gap_ci_row(interface_label, friction, seed_df)
    row["Interface"] = row.pop("Domain")
    return row


def build_support_figure(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.sort_values("friction_level", kind="mergesort").reset_index(drop=True).copy()
    plot_df["disagreement_rate"] = plot_df["disagreement_rate"].astype(float)

    fig, ax = plt.subplots(figsize=(3.3, 2.4), constrained_layout=True)
    ax.plot(
        plot_df["friction_level"],
        plot_df["disagreement_rate"],
        color="#c44e52",
        marker="o",
        linewidth=2.0,
    )
    ax.set_xlabel("Friction")
    ax.set_ylabel("Disagreement rate")
    ax.set_title("Event micro Q2")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linewidth=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_cross_domain_stats(canonical: VariantResult) -> None:
    synthetic_raw = pd.read_csv(SYNTHETIC_Q2_RAW)
    synthetic_outputs, _ = build_domain_rank_summary(
        synthetic_raw,
        domain="synthetic",
        expected_interface_id="tempered",
    )
    inventory_seed = pd.read_csv(INVENTORY_Q2_SEED)
    load_following_raw = pd.read_csv(LOAD_FOLLOWING_Q2_RAW)
    load_following_outputs, _ = build_domain_rank_summary(
        load_following_raw,
        domain="load_following_elecdiag",
        expected_interface_id="responsive",
    )

    recurrence_rows = [
        majority_test_row("Event micro", 0.50, canonical.seed_selection),
        majority_test_row("Event micro", 1.00, canonical.seed_selection),
        majority_test_row("Inventory", 0.50, inventory_seed),
        majority_test_row("Inventory", 1.00, inventory_seed),
        majority_test_row("Load-following", 0.25, load_following_outputs["seed_level_selection_stats"]),
        majority_test_row("Load-following", 0.50, load_following_outputs["seed_level_selection_stats"]),
        majority_test_row("Load-following", 1.00, load_following_outputs["seed_level_selection_stats"]),
    ]
    recurrence_table = pd.DataFrame(recurrence_rows)
    write_table_bundle(recurrence_table, HARDENING_ANALYSIS_DIR / "table_q2_recurrence_tests")
    write_table_bundle(recurrence_table, REPORT_TABLES_DIR / "table_q2_recurrence_tests")

    gap_rows = [
        gap_ci_row("Event micro", 0.50, canonical.seed_selection),
        gap_ci_row("Event micro", 1.00, canonical.seed_selection),
        gap_ci_row("Inventory", 0.50, inventory_seed),
        gap_ci_row("Inventory", 1.00, inventory_seed),
        gap_ci_row("Load-following", 0.25, load_following_outputs["seed_level_selection_stats"]),
        gap_ci_row("Load-following", 0.50, load_following_outputs["seed_level_selection_stats"]),
        gap_ci_row("Load-following", 1.00, load_following_outputs["seed_level_selection_stats"]),
    ]
    gap_table = pd.DataFrame(gap_rows)
    write_table_bundle(gap_table, HARDENING_ANALYSIS_DIR / "table_q2_gap_bootstrap_cis")
    write_table_bundle(gap_table, REPORT_TABLES_DIR / "table_q2_gap_bootstrap_cis")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    for label, color, rank_df, summary_df in [
        ("Synthetic", "#1f77b4", synthetic_outputs["rank_correlation_by_friction"], synthetic_outputs["selection_summary_by_friction"]),
        ("Event micro", "#d62728", canonical.rank_summary, canonical.selection_summary),
    ]:
        x = rank_df["friction_level"].to_numpy(dtype=float)
        mean_flip = rank_df["mean_flip_rate"].to_numpy(dtype=float)
        err = 1.96 * rank_df["stderr_flip_rate"].to_numpy(dtype=float)
        axes[0].plot(x, mean_flip, color=color, marker="o", linewidth=2.0, label=label)
        axes[0].fill_between(x, np.clip(mean_flip - err, 0.0, 1.0), np.clip(mean_flip + err, 0.0, 1.0), color=color, alpha=0.15)

        x2 = summary_df["friction_level"].to_numpy(dtype=float)
        y2 = summary_df["disagreement_rate"].to_numpy(dtype=float)
        lows = []
        highs = []
        for row in summary_df.itertuples(index=False):
            ci = binomtest(int(row.deployed_suboptimal_seed_count), int(row.n_seeds)).proportion_ci(
                confidence_level=0.95,
                method="exact",
            )
            lows.append(float(ci.low))
            highs.append(float(ci.high))
        axes[1].plot(x2, y2, color=color, marker="o", linewidth=2.0, label=label)
        axes[1].fill_between(x2, lows, highs, color=color, alpha=0.15)

    axes[0].set_title("Ranking disagreement with 95% bands")
    axes[0].set_xlabel("Friction")
    axes[0].set_ylabel("Mean flip rate")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_title("Deployed-suboptimal share with 95% bands")
    axes[1].set_xlabel("Friction")
    axes[1].set_ylabel("Forecast-selected not deployed-best")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    uncertainty_fig = HARDENING_ANALYSIS_DIR / "fig_q2_uncertainty_appendix.png"
    fig.savefig(uncertainty_fig, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(uncertainty_fig, REPORT_FIGURES_DIR / "fig_q2_uncertainty_appendix.png")

    strip_domains = [
        ("Event micro", canonical.seed_selection, [0.0, 0.5, 1.0]),
        ("Inventory", inventory_seed, [0.0, 0.25, 0.5, 1.0]),
        ("Load-following", load_following_outputs["seed_level_selection_stats"], [0.0, 0.25, 0.5, 1.0]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, (title, seed_df, frictions) in zip(axes, strip_domains):
        work = seed_df.loc[seed_df["friction_level"].isin(frictions)].copy()
        work["seed_order"] = work["seed"].rank(method="dense").astype(int)
        for x_idx, friction in enumerate(frictions):
            fr_df = work.loc[np.isclose(work["friction_level"], friction, atol=1e-12)].copy()
            y = fr_df["seed_order"].to_numpy(dtype=float)
            disagree = fr_df["selection_disagreement_flag"].to_numpy(dtype=bool)
            ax.scatter(
                np.full_like(y, x_idx),
                y,
                s=18,
                facecolors=np.where(disagree, "#111111", "white"),
                edgecolors="#111111",
                linewidths=0.6,
            )
        ax.set_title(title)
        ax.set_xticks(range(len(frictions)))
        ax.set_xticklabels([f"{value:.2f}" for value in frictions], rotation=0)
        ax.set_xlabel("Friction")
        ax.set_ylabel("Seed")
        ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    strip_fig = HARDENING_ANALYSIS_DIR / "fig_q2_seed_recurrence_appendix.png"
    fig.savefig(strip_fig, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(strip_fig, REPORT_FIGURES_DIR / "fig_q2_seed_recurrence_appendix.png")


def build_hysteresis_stats_addendum(canonical: VariantResult, hysteresis: VariantResult) -> None:
    recurrence_rows = [
        interface_majority_test_row("Fixed threshold", 0.50, canonical.seed_selection),
        interface_majority_test_row("Fixed threshold", 1.00, canonical.seed_selection),
        interface_majority_test_row("Hysteresis threshold", 0.50, hysteresis.seed_selection),
        interface_majority_test_row("Hysteresis threshold", 1.00, hysteresis.seed_selection),
    ]
    recurrence_table = pd.DataFrame(recurrence_rows)
    recurrence_table = recurrence_table[
        ["Interface", "Friction", "Deployed-suboptimal seeds / total", "Share", "95% exact CI", "One-sided binomial p"]
    ]
    write_table_bundle(recurrence_table, HARDENING_ANALYSIS_DIR / "table_event_micro_interface_recurrence_tests")
    write_table_bundle(recurrence_table, REPORT_TABLES_DIR / "table_event_micro_interface_recurrence_tests")

    gap_rows = [
        interface_gap_ci_row("Fixed threshold", 0.50, canonical.seed_selection),
        interface_gap_ci_row("Fixed threshold", 1.00, canonical.seed_selection),
        interface_gap_ci_row("Hysteresis threshold", 0.50, hysteresis.seed_selection),
        interface_gap_ci_row("Hysteresis threshold", 1.00, hysteresis.seed_selection),
    ]
    gap_table = pd.DataFrame(gap_rows)
    gap_table = gap_table[
        ["Interface", "Friction", "Mean deployed gap", "Mean gap 95% bootstrap CI", "Median deployed gap", "Median gap 95% bootstrap CI"]
    ]
    write_table_bundle(gap_table, HARDENING_ANALYSIS_DIR / "table_event_micro_interface_gap_bootstrap_cis")
    write_table_bundle(gap_table, REPORT_TABLES_DIR / "table_event_micro_interface_gap_bootstrap_cis")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    for row_idx, variant in enumerate([canonical, hysteresis]):
        rank_df = variant.rank_summary.sort_values("friction_level", kind="mergesort").reset_index(drop=True)
        summary_df = variant.selection_summary.sort_values("friction_level", kind="mergesort").reset_index(drop=True)
        color = "#d62728" if variant.interface_id == "fixed_threshold" else "#2ca02c"
        label = "Fixed threshold" if variant.interface_id == "fixed_threshold" else "Hysteresis threshold"

        x = rank_df["friction_level"].to_numpy(dtype=float)
        mean_flip = rank_df["mean_flip_rate"].to_numpy(dtype=float)
        err = 1.96 * rank_df["stderr_flip_rate"].to_numpy(dtype=float)
        axes[row_idx, 0].plot(x, mean_flip, color=color, marker="o", linewidth=2.0)
        axes[row_idx, 0].fill_between(
            x,
            np.clip(mean_flip - err, 0.0, 1.0),
            np.clip(mean_flip + err, 0.0, 1.0),
            color=color,
            alpha=0.15,
        )
        axes[row_idx, 0].set_title(f"{label}: mean flip rate")
        axes[row_idx, 0].set_xlabel("Friction")
        axes[row_idx, 0].set_ylabel("Mean flip rate")
        axes[row_idx, 0].grid(alpha=0.25, linewidth=0.6)

        x2 = summary_df["friction_level"].to_numpy(dtype=float)
        y2 = summary_df["disagreement_rate"].to_numpy(dtype=float)
        lows = []
        highs = []
        for row in summary_df.itertuples(index=False):
            ci = binomtest(int(row.deployed_suboptimal_seed_count), int(row.n_seeds)).proportion_ci(
                confidence_level=0.95,
                method="exact",
            )
            lows.append(float(ci.low))
            highs.append(float(ci.high))
        axes[row_idx, 1].plot(x2, y2, color=color, marker="o", linewidth=2.0)
        axes[row_idx, 1].fill_between(x2, lows, highs, color=color, alpha=0.15)
        axes[row_idx, 1].set_title(f"{label}: deployed-suboptimal share")
        axes[row_idx, 1].set_xlabel("Friction")
        axes[row_idx, 1].set_ylabel("Forecast-selected not deployed-best")
        axes[row_idx, 1].set_ylim(-0.02, 1.02)
        axes[row_idx, 1].grid(alpha=0.25, linewidth=0.6)

    uncertainty_fig = HARDENING_ANALYSIS_DIR / "fig_event_micro_interface_uncertainty_appendix.png"
    fig.savefig(uncertainty_fig, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(uncertainty_fig, REPORT_FIGURES_DIR / "fig_event_micro_interface_uncertainty_appendix.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for ax, variant in zip(axes, [canonical, hysteresis]):
        frictions = [0.0, 0.5, 1.0]
        work = variant.seed_selection.loc[variant.seed_selection["friction_level"].isin(frictions)].copy()
        work["seed_order"] = work["seed"].rank(method="dense").astype(int)
        title = "Fixed threshold" if variant.interface_id == "fixed_threshold" else "Hysteresis threshold"
        for x_idx, friction in enumerate(frictions):
            fr_df = work.loc[np.isclose(work["friction_level"], friction, atol=1e-12)].copy()
            y = fr_df["seed_order"].to_numpy(dtype=float)
            disagree = fr_df["selection_disagreement_flag"].to_numpy(dtype=bool)
            ax.scatter(
                np.full_like(y, x_idx),
                y,
                s=16,
                facecolors=np.where(disagree, "#111111", "white"),
                edgecolors="#111111",
                linewidths=0.5,
            )
        ax.set_title(title)
        ax.set_xticks(range(len(frictions)))
        ax.set_xticklabels([f"{value:.2f}" for value in frictions])
        ax.set_xlabel("Friction")
        ax.set_ylabel("Seed")
        ax.grid(alpha=0.15, linewidth=0.5, axis="y")

    seed_fig = HARDENING_ANALYSIS_DIR / "fig_event_micro_interface_seed_recurrence_appendix.png"
    fig.savefig(seed_fig, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(seed_fig, REPORT_FIGURES_DIR / "fig_event_micro_interface_seed_recurrence_appendix.png")


def write_report_artifacts(
    canonical: VariantResult,
    tau050: VariantResult,
    logloss: VariantResult,
    hysteresis: VariantResult | None,
    hysteresis_category: str,
) -> None:
    write_table_bundle(main_table(canonical.selection_summary), HARDENING_ANALYSIS_DIR / "table_q2_selection_drift_event_micro_main")
    write_table_bundle(main_table(canonical.selection_summary), REPORT_TABLES_DIR / "table_q2_selection_drift_event_micro_main")
    write_table_bundle(full_table(canonical.selection_summary), HARDENING_ANALYSIS_DIR / "table_q2_selection_drift_event_micro")
    write_table_bundle(full_table(canonical.selection_summary), REPORT_TABLES_DIR / "table_q2_selection_drift_event_micro")
    write_table_bundle(
        threshold_robustness_table(canonical.selection_summary, tau050.selection_summary),
        HARDENING_ANALYSIS_DIR / "table_q2_selection_drift_event_micro_threshold_robustness",
    )
    write_table_bundle(
        threshold_robustness_table(canonical.selection_summary, tau050.selection_summary),
        REPORT_TABLES_DIR / "table_q2_selection_drift_event_micro_threshold_robustness",
    )
    write_table_bundle(logloss_table(logloss.selection_summary), HARDENING_ANALYSIS_DIR / "table_q2_selection_drift_event_micro_logloss")
    write_table_bundle(logloss_table(logloss.selection_summary), REPORT_TABLES_DIR / "table_q2_selection_drift_event_micro_logloss")

    if hysteresis is not None and hysteresis_category in {"strong_pass", "weak_pass"}:
        write_table_bundle(
            hysteresis_table(hysteresis.selection_summary),
            HARDENING_ANALYSIS_DIR / "table_q2_selection_drift_event_micro_hysteresis_robustness",
        )
        write_table_bundle(
            hysteresis_table(hysteresis.selection_summary),
            REPORT_TABLES_DIR / "table_q2_selection_drift_event_micro_hysteresis_robustness",
        )

    build_support_figure(canonical.selection_summary, HARDENING_ANALYSIS_DIR / "fig_q2_event_micro_support.png")
    shutil.copy2(HARDENING_ANALYSIS_DIR / "fig_q2_event_micro_support.png", REPORT_FIGURES_DIR / "fig_q2_event_micro_support.png")

    support_note_lines = [
        "# Event Micro Hardening Note",
        "",
        f"- Canonical raw source: `{logical_display_path(canonical.output_dir / 'q2_diff_forecasts_same_interface.csv')}`",
        f"- Canonical seed count: {int(canonical.selection_summary['n_seeds'].iloc[0])}",
        "- Canonical interface: fixed threshold",
        "- Threshold robustness rerun: tau=0.50 under the same fixed-threshold interface.",
        "- Log-loss robustness is recomputed from the canonical seed-100 seed-level metrics.",
    ]
    if hysteresis is not None:
        support_note_lines.append(f"- Hysteresis verdict: {hysteresis_category}.")
    write_markdown(REPORT_TABLES_DIR / "event_micro_support_note.md", support_note_lines)

def build_manifest(
    canonical: VariantResult,
    tau050: VariantResult,
    logloss: VariantResult,
    step7_gate: dict[str, object],
    hysteresis: VariantResult | None,
    hysteresis_gate: dict[str, object],
) -> None:
    ensure_logical_alias()

    artifact_entries = [
        {
            "artifact": "Event-micro main table",
            "artifact_source": "results/tables/table_q2_selection_drift_event_micro_main.csv",
            "raw_source_of_truth": logical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
            "report_facing": True,
        },
        {
            "artifact": "Event-micro appendix full table",
            "artifact_source": "results/tables/table_q2_selection_drift_event_micro.csv",
            "raw_source_of_truth": logical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
            "report_facing": True,
        },
        {
            "artifact": "Event-micro threshold robustness table",
            "artifact_source": "results/tables/table_q2_selection_drift_event_micro_threshold_robustness.csv",
            "raw_source_of_truth": logical_display_path(tau050.output_dir / "q2_diff_forecasts_same_interface.csv"),
            "report_facing": True,
        },
        {
            "artifact": "Event-micro log-loss robustness table",
            "artifact_source": "results/tables/table_q2_selection_drift_event_micro_logloss.csv",
            "raw_source_of_truth": logical_display_path(logloss.output_dir / "selection_summary_by_friction.csv"),
            "report_facing": True,
        },
        {
            "artifact": "Main Q2 figure",
            "artifact_source": "results/figures/fig_q2_results_v2.png",
            "raw_source_of_truth": logical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
            "report_facing": True,
        },
        {
            "artifact": "Cross-domain recurrence table",
            "artifact_source": "results/tables/table_q2_recurrence_tests.csv",
            "raw_source_of_truth": logical_display_path(canonical.output_dir / "derived/seed_level_selection_stats.csv"),
            "report_facing": True,
        },
        {
            "artifact": "Cross-domain gap CI table",
            "artifact_source": "results/tables/table_q2_gap_bootstrap_cis.csv",
            "raw_source_of_truth": logical_display_path(canonical.output_dir / "derived/seed_level_selection_stats.csv"),
            "report_facing": True,
        },
    ]
    if hysteresis is not None and hysteresis_gate["category"] in {"strong_pass", "weak_pass"}:
        artifact_entries.extend(
            [
                {
                    "artifact": "Event-micro hysteresis robustness table",
                    "artifact_source": "results/tables/table_q2_selection_drift_event_micro_hysteresis_robustness.csv",
                    "raw_source_of_truth": logical_display_path(hysteresis.output_dir / "q2_diff_forecasts_same_interface.csv"),
                    "report_facing": True,
                },
                {
                    "artifact": "Event-micro interface recurrence table",
                    "artifact_source": "results/tables/table_event_micro_interface_recurrence_tests.csv",
                    "raw_source_of_truth": logical_display_path(hysteresis.output_dir / "derived/seed_level_selection_stats.csv"),
                    "report_facing": True,
                },
                {
                    "artifact": "Event-micro interface gap CI table",
                    "artifact_source": "results/tables/table_event_micro_interface_gap_bootstrap_cis.csv",
                    "raw_source_of_truth": logical_display_path(hysteresis.output_dir / "derived/seed_level_selection_stats.csv"),
                    "report_facing": True,
                },
            ]
        )

    manifest_payload = {
        "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
        "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
        "workspace_root": str(ROOT),
        "workstream": "event_micro_hardening_step7_9",
        "canonical_event_micro_raw": logical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
        "physical_canonical_event_micro_raw": physical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
        "step7_gate": step7_gate,
        "hysteresis_gate": hysteresis_gate,
        "artifacts": artifact_entries,
    }
    write_json(HARDENING_STORY_DIR / "event_micro_hardening_result_manifest.json", manifest_payload)
    write_markdown(
        HARDENING_STORY_DIR / "event_micro_hardening_result_manifest.md",
        [
            "# Event-Micro Hardening Result Manifest",
            "",
            f"- Logical canonical root: `{repo_relative(LOGICAL_CANONICAL_ROOT)}`",
            f"- Physical storage root: `{repo_relative(PHYSICAL_STORAGE_ROOT)}`",
            f"- Canonical event-micro raw source: `{logical_display_path(canonical.output_dir / 'q2_diff_forecasts_same_interface.csv')}`",
            f"- Step 7 status: `{step7_gate['status']}` with agreement pattern `{step7_gate['agreement_pattern']}`.",
            f"- Hysteresis status: `{hysteresis_gate['category']}`.",
        ],
    )

    claim_map = {
        "headline_q2_event_micro": {
            "claim": (
                "The canonical fixed-threshold event-micro benchmark is the report-card's main forecasting-native "
                "Q2 evidence block."
            ),
            "evidence": [
                "results/tables/table_q2_selection_drift_event_micro_main.csv",
                "results/tables/table_q2_selection_drift_event_micro.csv",
                "results/figures/fig_q2_results_v2.png",
            ],
            "raw_source_of_truth": logical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
        },
        "event_micro_robustness_package": {
            "claim": "The event-micro direction persists under alternate threshold and log-loss reranking.",
            "evidence": [
                "results/tables/table_q2_selection_drift_event_micro_threshold_robustness.csv",
                "results/tables/table_q2_selection_drift_event_micro_logloss.csv",
            ],
            "raw_source_of_truth": [
                logical_display_path(tau050.output_dir / "q2_diff_forecasts_same_interface.csv"),
                logical_display_path(logloss.output_dir / "selection_summary_by_friction.csv"),
            ],
        },
    }
    if hysteresis is not None and hysteresis_gate["category"] in {"strong_pass", "weak_pass"}:
        claim_map["event_micro_hysteresis_support"] = {
            "claim": "A second hysteresis-threshold interface preserves event-micro direction at appendix support strength.",
            "evidence": [
                "results/tables/table_q2_selection_drift_event_micro_hysteresis_robustness.csv",
                "results/tables/table_event_micro_interface_recurrence_tests.csv",
                "results/tables/table_event_micro_interface_gap_bootstrap_cis.csv",
            ],
            "raw_source_of_truth": logical_display_path(hysteresis.output_dir / "q2_diff_forecasts_same_interface.csv"),
        }
    write_json(HARDENING_STORY_DIR / "event_micro_hardening_claim_to_evidence_map.json", claim_map)
    write_markdown(
        HARDENING_STORY_DIR / "event_micro_hardening_claim_to_evidence_map.md",
        [
            "# Event-Micro Hardening Claim Map",
            "",
            "- `headline_q2_event_micro`: canonical seed-100 fixed-threshold table and figure package.",
            "- `event_micro_robustness_package`: alternate-threshold and log-loss robustness package.",
            *(
                ["- `event_micro_hysteresis_support`: appendix-only second-interface support package."]
                if hysteresis is not None and hysteresis_gate["category"] in {"strong_pass", "weak_pass"}
                else []
            ),
        ],
    )


def main() -> int:
    args = parse_args()
    ensure_logical_alias()
    for path in [HARDENING_ROOT, HARDENING_ANALYSIS_DIR, HARDENING_STORY_DIR]:
        ensure_dir(path)

    canonical = run_variant(
        name="canonical_seed100",
        config_path=CANONICAL_CONFIG,
        output_dir=CANONICAL_RUN_DIR,
        expected_interface_id="fixed_threshold",
        skip_existing=args.skip_existing,
    )
    tau050 = run_variant(
        name="tau050_seed100",
        config_path=TAU050_CONFIG,
        output_dir=TAU050_RUN_DIR,
        expected_interface_id="fixed_threshold",
        skip_existing=args.skip_existing,
    )
    hysteresis = run_variant(
        name="hysteresis_seed100",
        config_path=HYSTERESIS_CONFIG,
        output_dir=HYSTERESIS_RUN_DIR,
        expected_interface_id="hysteresis_threshold",
        skip_existing=args.skip_existing,
    )
    logloss = build_logloss_variant(canonical)

    step7_gate = evaluate_step7(canonical.selection_summary)
    hysteresis_gate = evaluate_hysteresis(hysteresis.selection_summary)

    write_report_artifacts(canonical, tau050, logloss, hysteresis, hysteresis_gate["category"])
    build_cross_domain_stats(canonical)
    if hysteresis_gate["category"] in {"strong_pass", "weak_pass"}:
        build_hysteresis_stats_addendum(canonical, hysteresis)

    ledger = {
        "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
        "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
        "canonical_run": {
            "config": str(CANONICAL_CONFIG.relative_to(ROOT)),
            "raw_csv": logical_display_path(canonical.output_dir / "q2_diff_forecasts_same_interface.csv"),
            "seed_metrics_csv": logical_display_path(canonical.output_dir / "seed_level_metrics.csv"),
        },
        "tau050_run": {
            "config": str(TAU050_CONFIG.relative_to(ROOT)),
            "raw_csv": logical_display_path(tau050.output_dir / "q2_diff_forecasts_same_interface.csv"),
        },
        "hysteresis_run": {
            "config": str(HYSTERESIS_CONFIG.relative_to(ROOT)),
            "raw_csv": logical_display_path(hysteresis.output_dir / "q2_diff_forecasts_same_interface.csv"),
        },
        "step7_gate": step7_gate,
        "hysteresis_gate": hysteresis_gate,
        "report_tex": str(REPORT_MANIFEST.relative_to(ROOT)),
    }
    write_json(HARDENING_ROOT / "event_micro_hardening_ledger.json", ledger)
    write_markdown(
        HARDENING_STORY_DIR / "event_micro_hardening_status.md",
        [
            "# Event-Micro Hardening Status",
            "",
            f"- Step 7 status: `{step7_gate['status']}`",
            f"- Step 7 agreement pattern: `{step7_gate['agreement_pattern']}`",
            f"- Hysteresis verdict: `{hysteresis_gate['category']}`",
            f"- Canonical raw source: `{logical_display_path(canonical.output_dir / 'q2_diff_forecasts_same_interface.csv')}`",
            f"- Hysteresis raw source: `{logical_display_path(hysteresis.output_dir / 'q2_diff_forecasts_same_interface.csv')}`",
        ],
    )
    build_manifest(canonical, tau050, logloss, step7_gate, hysteresis, hysteresis_gate)
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

