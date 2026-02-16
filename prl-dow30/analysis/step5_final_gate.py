from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.step5_common import describe, load_latest_archive_frames, pair_seed_values
from scripts.prl_gate_utils import aggregate_prl_gate


def _normalize_turnover_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "avg_turnover_exec" not in out.columns and "avg_turnover" in out.columns:
        out["avg_turnover_exec"] = pd.to_numeric(out["avg_turnover"], errors="coerce")
    return out


def _rule(value: bool, detail: dict) -> dict:
    return {"pass": bool(value), **detail}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Step5 final gate.")
    parser.add_argument("--input-root", required=True, help="Step5 run root")
    parser.add_argument(
        "--out-path",
        help="Output JSON path. Default: <input-root>/reports/paper/step5/step5_gate_result.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_root = Path(args.input_root)
    out_path = Path(args.out_path) if args.out_path else input_root / "reports" / "paper" / "step5" / "step5_gate_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_loaded = load_latest_archive_frames(input_root, prefix="metrics")
    if "A" not in metrics_loaded or "B" not in metrics_loaded:
        raise ValueError("STEP5_GATE_MISSING_MAIN_COMPARISON: need archive metrics for A and B")

    base_path, base_df_raw = metrics_loaded["A"]
    prl_path, prl_df_raw = metrics_loaded["B"]
    base_df = _normalize_turnover_cols(base_df_raw)
    prl_df = _normalize_turnover_cols(prl_df_raw)

    base_turnover_mean = float(pd.to_numeric(base_df["avg_turnover_exec"], errors="coerce").mean())
    prl_turnover_mean = float(pd.to_numeric(prl_df["avg_turnover_exec"], errors="coerce").mean())
    turnover_limit = base_turnover_mean * 1.05
    turnover_ok = bool(prl_turnover_mean <= turnover_limit)

    sharpe_seeds, _, _, sharpe_delta = pair_seed_values(base_df, prl_df, "sharpe_net_exp")
    sharpe_stats = describe(sharpe_delta)
    sharpe_delta_median = float(sharpe_stats["median"])
    sharpe_ok = bool(sharpe_delta_median >= -0.02)
    collapse_count = int(np.sum(sharpe_delta <= -0.2))
    collapse_ok = bool(collapse_count <= 1)

    drawdown_ok = True
    drawdown_delta_median = float("nan")
    if "max_drawdown_net_exp" in base_df.columns and "max_drawdown_net_exp" in prl_df.columns:
        _, _, _, drawdown_delta = pair_seed_values(base_df, prl_df, "max_drawdown_net_exp")
        drawdown_delta_median = float(np.median(drawdown_delta)) if drawdown_delta.size else float("nan")
        drawdown_ok = bool(drawdown_delta_median >= -0.03)

    prl_run_ids = sorted(prl_df["run_id"].dropna().astype(str).unique().tolist())
    prl_gate = aggregate_prl_gate(
        prl_run_ids,
        {
            "reports_dir": str(input_root / "reports"),
            "logs_dir": str(input_root / "logs"),
        },
        prl_path,
    )

    required_checks = {
        "prl_gate_prl_only": _rule(
            prl_gate.passed,
            {
                "reason": prl_gate.reason,
                "emergency_rate": prl_gate.emergency_rate,
                "prl_prob_p05": prl_gate.prl_prob_p05,
                "prl_prob_p95": prl_gate.prl_prob_p95,
                "prl_prob_std": prl_gate.prl_prob_std,
                "source": prl_gate.source,
            },
        ),
        "turnover_increase_forbidden": _rule(
            turnover_ok,
            {
                "prl_turnover_exec_mean": prl_turnover_mean,
                "baseline_turnover_exec_mean": base_turnover_mean,
                "limit": turnover_limit,
            },
        ),
        "median_sharpe_delta": _rule(
            sharpe_ok,
            {
                "median_delta": sharpe_delta_median,
                "threshold": -0.02,
                "n_pairs": len(sharpe_seeds),
            },
        ),
    }

    optional_checks = {
        "drawdown_delta_median": _rule(
            drawdown_ok,
            {
                "median_delta": drawdown_delta_median,
                "threshold": -0.03,
            },
        ),
        "collapse_seed_limit": _rule(
            collapse_ok,
            {
                "collapse_seed_count": collapse_count,
                "threshold": 1,
            },
        ),
    }

    required_pass = all(item["pass"] for item in required_checks.values())

    failed_required = [name for name, value in required_checks.items() if not value["pass"]]
    failed_optional = [name for name, value in optional_checks.items() if not value["pass"]]

    if failed_required:
        reason = "fail_required:" + ",".join(failed_required)
    elif failed_optional:
        reason = "pass_required_warn_optional:" + ",".join(failed_optional)
    else:
        reason = "pass"

    result = {
        "step5_gate_pass": bool(required_pass),
        "step5_gate_reason": reason,
        "comparison": {
            "baseline": {
                "exp_key": "A",
                "source_metrics": str(base_path),
                "prl_gate_pass": "SKIP",
                "prl_gate_reason": "SKIP_PRL_OFF",
            },
            "prl": {
                "exp_key": "B",
                "source_metrics": str(prl_path),
                "prl_gate_pass": bool(prl_gate.passed),
                "prl_gate_reason": prl_gate.reason,
            },
        },
        "required_checks": required_checks,
        "optional_checks": optional_checks,
        "paired_delta_summary": {
            "sharpe_net_exp": {
                "mean": sharpe_stats["mean"],
                "std": sharpe_stats["std"],
                "median": sharpe_stats["median"],
                "p25": sharpe_stats["p25"],
                "p75": sharpe_stats["p75"],
                "iqr": sharpe_stats["iqr"],
                "n_pairs": len(sharpe_seeds),
            },
            "max_drawdown_net_exp": {
                "median": drawdown_delta_median,
            },
        },
    }

    out_path.write_text(json.dumps(result, indent=2))

    summary_path = out_path.with_suffix(".md")
    summary_lines = [
        "# Step5 Final Gate",
        f"- step5_gate_pass: {result['step5_gate_pass']}",
        f"- step5_gate_reason: {result['step5_gate_reason']}",
        f"- baseline PRL gate: SKIP",
        f"- PRL gate pass: {prl_gate.passed} ({prl_gate.reason})",
        f"- turnover mean (base/prl): {base_turnover_mean:.6f} / {prl_turnover_mean:.6f}",
        f"- sharpe delta median: {sharpe_delta_median:.6f}",
        f"- drawdown delta median: {drawdown_delta_median:.6f}",
        f"- collapse seed count: {collapse_count}",
    ]
    summary_path.write_text("\n".join(summary_lines))

    print(f"Step5 gate complete. Result: {out_path}")


if __name__ == "__main__":
    main()
