"""
Build Gate2 leaderboard with mid-focused deltas between reference and candidates.

Usage:
  python -m scripts.build_gate2_leaderboard \\
    --reference-run-index outputs/exp_runs/gate2/reference_baseline_sac_PACK/reports/run_index.json \\
    --candidate-run-indexes "outputs/exp_runs/gate2/gate2_A2l01_midplast_m07/*/reports/run_index.json" \\
    --output-dir outputs/exp_runs/gate2

Decision rule:
  PASS if (min delta_mid_sharpe_net_exp across windows >= -0.01) AND (turnover <= guardrail)
  guardrail = 1.2 * ref avg_turnover (per eval_window)
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
    uniq = []
    seen = set()
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


def _compute_summary(ref_metrics: pd.DataFrame, cand_metrics: pd.DataFrame, *, regime: bool = False) -> pd.DataFrame:
    cols = ["sharpe_net_exp", "avg_turnover", "max_drawdown_net_exp", "cumulative_return_net_exp"]
    ref_mean = _mean_by_window(ref_metrics, cols)
    cand_mean = _mean_by_window(cand_metrics, cols)
    delta = cand_mean - ref_mean
    combined = pd.concat(
        [ref_mean.add_suffix("_ref"), cand_mean.add_suffix("_cand"), delta.add_suffix("_delta")],
        axis=1,
    )
    combined["regime"] = "mid" if regime else "all"
    return combined.reset_index().rename(columns={"eval_window": "window"})


def main():
    parser = argparse.ArgumentParser(description="Build Gate2 leaderboard (mid-focused) from ref/candidate run indexes.")
    parser.add_argument("--reference-run-index", required=True)
    parser.add_argument("--candidate-run-indexes", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/exp_runs/gate2")
    args = parser.parse_args()

    ref_idx = _load_run_index(Path(args.reference_run_index))
    ref_run_ids = ref_idx.get("run_ids", [])
    ref_metrics = _load_with_filter(Path(ref_idx["metrics_path"]), ref_run_ids)
    ref_regime = _load_with_filter(Path(ref_idx["regime_metrics_path"]), ref_run_ids)
    ref_regime = ref_regime[ref_regime["regime"] == "mid"].copy()

    cand_paths = _collect_paths(args.candidate_run_indexes)
    if not cand_paths:
        raise SystemExit("No candidate run_index paths found.")

    rows = []
    for cand_path in cand_paths:
        idx = _load_run_index(cand_path)
        run_ids = idx.get("run_ids", [])
        metrics = _load_with_filter(Path(idx["metrics_path"]), run_ids)
        regime = _load_with_filter(Path(idx["regime_metrics_path"]), run_ids)
        regime_mid = regime[regime["regime"] == "mid"].copy()

        all_summary = _compute_summary(ref_metrics, metrics, regime=False)
        mid_summary = _compute_summary(ref_regime, regime_mid, regime=True)
        summary = pd.concat([all_summary, mid_summary], ignore_index=True)
        # Guardrail & pass/fail evaluation on mid summary
        mid_rows = summary[summary["regime"] == "mid"]
        guardrail_flags = []
        pass_mid_flags = []
        for _, r in mid_rows.iterrows():
            guardrail = 1.2 * r["avg_turnover_ref"]
            guardrail_flags.append(r["avg_turnover_cand"] > guardrail)
            pass_mid_flags.append(r["sharpe_net_exp_delta"] >= -0.01)
        guardrail_hit = any(guardrail_flags)
        pass_mid = all(pass_mid_flags)
        decision = "PASS" if pass_mid and not guardrail_hit else "FAIL"

        row = {
            "exp_name": idx.get("exp_name", Path(idx.get("config_path", "")).stem),
            "run_index_path": str(cand_path),
            "decision": decision,
            "guardrail_hit": guardrail_hit,
            "pass_mid": pass_mid,
        }
        for w in ["W1", "W2"]:
            sub = mid_rows[mid_rows["window"] == w]
            if not sub.empty:
                r = sub.iloc[0]
                row[f"mid_sharpe_ref_{w}"] = r["sharpe_net_exp_ref"]
                row[f"mid_sharpe_cand_{w}"] = r["sharpe_net_exp_cand"]
                row[f"mid_sharpe_delta_{w}"] = r["sharpe_net_exp_delta"]
                row[f"turnover_ref_{w}"] = r["avg_turnover_ref"]
                row[f"turnover_cand_{w}"] = r["avg_turnover_cand"]
        rows.append(row)

        # Write per-candidate summary CSV for inspection
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cand_summary_path = out_dir / f"{row['exp_name']}_mid_summary.csv"
        summary.to_csv(cand_summary_path, index=False)

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values(["decision", "mid_sharpe_delta_W1"], ascending=[False, False])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = out_dir / "Gate2_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    summary_lines = []
    summary_lines.append("# Gate2 summary (mid-focused)")
    summary_lines.append("")
    summary_lines.append(f"- Reference: {args.reference_run_index}")
    summary_lines.append(f"- Candidates: {len(cand_paths)} run_index files")
    summary_lines.append("- PASS rule: min(delta_mid_sharpe) >= -0.01 and no guardrail breach (turnover > 1.2x ref).")
    summary_lines.append("")
    summary_lines.append("## Leaderboard")
    summary_lines.append(leaderboard.to_markdown(index=False))
    (out_dir / "gate2_summary.md").write_text("\n".join(summary_lines))
    print(f"Wrote {leaderboard_path} and gate2_summary.md")


if __name__ == "__main__":
    main()
