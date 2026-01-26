"""
Gate3 leaderboard with mean/std deltas and guardrails.

Decision:
  - HARD FAIL if cand_turnover >= ref*hardcut OR delta_sharpe <= fail_delta_sharpe
  - PASS if delta_sharpe >= pass_delta_sharpe AND cand_mdd >= ref_mdd (no worsening)
  - Otherwise BORDERLINE
Guardrail flag if cand_turnover >= ref*guardrail.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _collect_paths(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            path = Path(p)
            if path.exists():
                paths.append(path)
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _load_run_index(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["run_index_path"] = str(path)
    return data


def _load_with_filter(csv_path: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    run_ids = set(run_ids)
    if run_ids:
        df = df[df["run_id"].isin(run_ids)].copy()
    return df


def _mean_by_window(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("eval_window")[cols].mean(numeric_only=True)


def _score(sharpe: float, mdd: float, turnover: float, *, turnover_weight: float = 0.10) -> float:
    return float(sharpe - 0.25 * abs(mdd) - turnover_weight * turnover)


def _reason_from_mean_std(delta_mean: float, delta_std: float) -> str:
    import math

    dm = 0.0 if delta_mean is None or (isinstance(delta_mean, float) and math.isnan(delta_mean)) else float(delta_mean)
    ds = 0.0 if delta_std is None or (isinstance(delta_std, float) and math.isnan(delta_std)) else float(delta_std)
    if abs(dm) < 1e-3 and abs(ds) < 1e-3:
        return "penalty scale too small or not effective (mean/std unchanged)"
    if ds < 0 and dm < 0:
        return "std decreased but mean also dropped (offset)"
    return ""


def _safe_float(val):
    try:
        import pandas as pd
    except Exception:
        pd = None
    if val is None:
        return 0.0
    try:
        if pd is not None and pd.isna(val):
            return 0.0
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return 0.0


def _compute_window_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sharpe_net_exp",
        "max_drawdown_net_exp",
        "avg_turnover",
        "mean_daily_net_return_exp",
        "std_daily_net_return_exp",
    ]
    present = [c for c in cols if c in metrics.columns]
    df = _mean_by_window(metrics, present)
    # ensure all expected columns exist (old runs may lack mean/std)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df.reset_index().rename(columns={"eval_window": "window"})


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Build Gate3 leaderboard from ref/candidate PACK run indexes.")
    parser.add_argument("--reference-run-index", required=True)
    parser.add_argument("--candidate-run-indexes", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/exp_runs/gate3")
    parser.add_argument("--turnover-guardrail-mult", type=float, default=1.2)
    parser.add_argument("--hardcut-turnover-mult", type=float, default=1.1)
    parser.add_argument("--pass-delta-sharpe", type=float, default=0.20)
    parser.add_argument("--fail-delta-sharpe", type=float, default=-0.05)
    args = parser.parse_args(argv)

    ref_idx = _load_run_index(Path(args.reference_run_index))
    ref_metrics = _load_with_filter(Path(ref_idx["metrics_path"]), ref_idx.get("run_ids", []))
    if ref_metrics.empty:
        raise SystemExit("Reference metrics empty after filtering run_ids.")
    ref_summary = _compute_window_summary(ref_metrics)

    cand_paths = _collect_paths(args.candidate_run_indexes)
    if not cand_paths:
        raise SystemExit("No candidate run_index paths found.")

    rows = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cand_path in cand_paths:
        idx = _load_run_index(cand_path)
        metrics = _load_with_filter(Path(idx["metrics_path"]), idx.get("run_ids", []))
        if metrics.empty:
            continue
        cand_summary = _compute_window_summary(metrics)
        merged = cand_summary.merge(ref_summary, on="window", suffixes=("_cand", "_ref"))
        if merged.empty:
            continue

        # deltas per window
        merged["delta_sharpe"] = merged["sharpe_net_exp_cand"] - merged["sharpe_net_exp_ref"]
        merged["delta_mdd"] = merged["max_drawdown_net_exp_cand"] - merged["max_drawdown_net_exp_ref"]
        merged["delta_turnover"] = merged["avg_turnover_cand"] - merged["avg_turnover_ref"]
        merged["delta_mean_daily_net_return_exp"] = (
            merged["mean_daily_net_return_exp_cand"] - merged["mean_daily_net_return_exp_ref"]
        )
        merged["delta_std_daily_net_return_exp"] = (
            merged["std_daily_net_return_exp_cand"] - merged["std_daily_net_return_exp_ref"]
        )

        # guardrails and decisions
        hard_fail_turnover = (merged["avg_turnover_cand"] >= args.hardcut_turnover_mult * merged["avg_turnover_ref"]).any()
        hard_fail_sharpe = (merged["delta_sharpe"] <= args.fail_delta_sharpe).any()
        guardrail = (merged["avg_turnover_cand"] >= args.turnover_guardrail_mult * merged["avg_turnover_ref"]).any()
        pass_gate3 = (
            (merged["delta_sharpe"] >= args.pass_delta_sharpe).all()
            and (merged["max_drawdown_net_exp_cand"] >= merged["max_drawdown_net_exp_ref"]).all()
        )

        if hard_fail_turnover or hard_fail_sharpe:
            decision = "FAIL"
            decision_reason = "hard_cut_turnover" if hard_fail_turnover else "sharpe_below_fail_threshold"
        elif pass_gate3:
            decision = "PASS"
            decision_reason = "delta_sharpe_pass_and_mdd_guardrail"
        else:
            decision = "BORDERLINE"
            decision_reason = "mixed_deltas_or_guardrail"

        exp_name = idx.get("exp_name", Path(idx.get("config_path", "")).stem)
        # aggregate score across windows using candidate means
        agg_sharpe = float(merged["sharpe_net_exp_cand"].mean())
        agg_mdd = float(merged["max_drawdown_net_exp_cand"].mean())
        agg_turnover = float(merged["avg_turnover_cand"].mean())
        score = _score(agg_sharpe, agg_mdd, agg_turnover, turnover_weight=0.10)
        score2 = _score(agg_sharpe, agg_mdd, agg_turnover, turnover_weight=0.30)

        for _, row_w in merged.iterrows():
            reason_comment = _reason_from_mean_std(
                _safe_float(row_w.get("delta_mean_daily_net_return_exp", 0.0)),
                _safe_float(row_w.get("delta_std_daily_net_return_exp", 0.0)),
            )
            rows.append(
                {
                    "exp_name": exp_name,
                    "eval_window": row_w["window"],
                    "mean_sharpe_net_exp": row_w["sharpe_net_exp_cand"],
                    "mean_mdd_net_exp": row_w["max_drawdown_net_exp_cand"],
                    "mean_avg_turnover": row_w["avg_turnover_cand"],
                    "mean_mean_daily_net_return_exp": row_w.get("mean_daily_net_return_exp_cand", float("nan")),
                    "mean_std_daily_net_return_exp": row_w.get("std_daily_net_return_exp_cand", float("nan")),
                    "delta_sharpe_vs_ref": row_w["delta_sharpe"],
                    "delta_mdd_vs_ref": row_w["delta_mdd"],
                    "delta_turnover_vs_ref": row_w["delta_turnover"],
                    "delta_mean_daily_net_return_exp_vs_ref": row_w.get("delta_mean_daily_net_return_exp", float("nan")),
                    "delta_std_daily_net_return_exp_vs_ref": row_w.get("delta_std_daily_net_return_exp", float("nan")),
                    "score": score,
                    "score2": score2,
                    "decision": decision,
                    "decision_reason": decision_reason,
                    "guardrail": guardrail,
                    "reference_run_index_path": args.reference_run_index,
                    "candidate_run_index_path": str(cand_path),
                    "fail_turnover_hard_cut": hard_fail_turnover,
                    "fail_sharpe": hard_fail_sharpe,
                    "comment": reason_comment,
                }
            )

        merged.to_csv(out_dir / f"{exp_name}_summary.csv", index=False)

    leaderboard = pd.DataFrame(rows)
    leaderboard_path = out_dir / "Gate3_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    summary_lines = [
        "# Gate3 summary",
        f"- Reference: {args.reference_run_index}",
        f"- Candidates: {len(cand_paths)}",
        f"- Guardrail turnover x{args.turnover_guardrail_mult}, hardcut x{args.hardcut_turnover_mult}",
        f"- PASS if ΔSharpe >= {args.pass_delta_sharpe} AND MDD 악화 없음; hard fail if ΔSharpe <= {args.fail_delta_sharpe} or turnover hardcut.",
        "",
        "## Leaderboard",
        leaderboard.to_markdown(index=False),
    ]
    (out_dir / "gate3_summary.md").write_text("\n".join(summary_lines))
    print(f"Wrote {leaderboard_path} and gate3_summary.md")


if __name__ == "__main__":
    main()
