"""
Build Gate1 leaderboard and summary using a reference baseline_sac run and PRL candidates.

Rules encoded from STEP 2 확장 명세 v1.1:
- Gate1 uses force_refresh=false and W1 only.
- Reference baseline_sac is run once; candidates are PRL-only.
- Analysis always filters via run_index.json.
- PASS rules:
  - T1: avg_turnover <= 0.70 * baseline_ref_avg_turnover AND sharpe_net_exp >= baseline_ref_sharpe_net_exp
  - T2: sharpe_net_exp >= baseline_ref_sharpe_net_exp + 0.10
- FAIL (hard cut):
  - sharpe_net_exp <= baseline_ref_sharpe_net_exp - 0.05
  - avg_turnover >= 1.10 * baseline_ref_avg_turnover
- Score = sharpe_net_exp - 0.25 * abs(max_drawdown_net_exp) - 0.10 * avg_turnover
  - If mid sharpe is worse than baseline mid, apply -0.05 penalty.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


LOGGER = logging.getLogger("gate1_leaderboard")


def _read_run_index(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["run_index_path"] = str(path)
    return data


def _load_with_filter(csv_path: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    run_ids = set(run_ids)
    if run_ids:
        df = df[df["run_id"].isin(run_ids)].copy()
    return df


def _pick_eval_window(df: pd.DataFrame) -> pd.DataFrame:
    if "period" in df.columns:
        df = df[df["period"] == "test"].copy()
    if "eval_window" in df.columns and not df["eval_window"].isna().all():
        target = sorted(df["eval_window"].dropna().unique().tolist())[0]
        df = df[df["eval_window"] == target].copy()
    return df


def _load_timesteps(config_path: str | Path) -> Optional[int]:
    try:
        cfg = yaml.safe_load(Path(config_path).read_text())
        sac_cfg = cfg.get("sac", {})
        return int(sac_cfg.get("total_timesteps")) if sac_cfg.get("total_timesteps") is not None else None
    except Exception:
        LOGGER.warning("Could not read timesteps from config=%s", config_path)
        return None


def _build_reference(
    run_index_path: Path,
    baseline_model_type: str,
) -> Tuple[Dict[int, dict], dict, pd.DataFrame]:
    idx = _read_run_index(run_index_path)
    metrics = _pick_eval_window(_load_with_filter(Path(idx["metrics_path"]), idx.get("run_ids", [])))
    regime = _pick_eval_window(_load_with_filter(Path(idx["regime_metrics_path"]), idx.get("run_ids", [])))
    baseline_rows = metrics[metrics["model_type"] == baseline_model_type].copy()
    if baseline_rows.empty:
        raise ValueError(f"No baseline rows (model_type={baseline_model_type}) found in {run_index_path}")

    ref_by_seed: Dict[int, dict] = {}
    for _, row in baseline_rows.iterrows():
        seed = int(row["seed"])
        ref_by_seed[seed] = {
            "sharpe_net_exp": float(row.get("sharpe_net_exp", 0.0)),
            "avg_turnover": float(row.get("avg_turnover", 0.0)),
            "max_drawdown_net_exp": float(row.get("max_drawdown_net_exp", 0.0)),
            "cumulative_return_net_exp": float(row.get("cumulative_return_net_exp", 0.0)),
            "timesteps": _load_timesteps(idx.get("config_path", "")),
        }

    regime_mid = regime[(regime["regime"] == "mid") & (regime["model_type"] == baseline_model_type)]
    for _, row in regime_mid.iterrows():
        seed = int(row["seed"])
        ref = ref_by_seed.get(seed)
        if ref is not None:
            ref["sharpe_net_exp_mid"] = float(row.get("sharpe_net_exp", 0.0))
            ref["cumulative_return_net_exp_mid"] = float(row.get("cumulative_return_net_exp", 0.0))

    # Use seed 0 as fallback reference if specific seed is missing later.
    fallback_seed = sorted(ref_by_seed.keys())[0]
    return ref_by_seed, ref_by_seed[fallback_seed], metrics


def _decision_and_score(
    cand: dict,
    ref: dict,
) -> Tuple[str, str, float]:
    sharpe = cand["sharpe_net_exp"]
    turnover = cand["avg_turnover"]
    mdd = cand["max_drawdown_net_exp"]
    ref_sharpe = ref["sharpe_net_exp"]
    ref_turnover = ref["avg_turnover"]

    pass_t1 = turnover <= 0.70 * ref_turnover and sharpe >= ref_sharpe - 1e-9
    pass_t2 = sharpe >= ref_sharpe + 0.10 - 1e-9
    fail_sharpe = sharpe <= ref_sharpe - 0.05 + 1e-9
    fail_turnover = turnover >= 1.10 * ref_turnover - 1e-9

    if fail_turnover:
        decision = "FAIL"
        reason = "fail_turnover_hard_cut"
    elif pass_t1 or pass_t2:
        decision = "PASS"
        reason = "T1" if pass_t1 else "T2"
    elif fail_sharpe:
        decision = "FAIL"
        reason = "fail_sharpe"
    else:
        decision = "FAIL"
        reason = "no_pass_criteria"

    score = sharpe - 0.25 * abs(mdd) - 0.10 * turnover
    mid_sharpe = cand.get("sharpe_net_exp_mid")
    ref_mid = ref.get("sharpe_net_exp_mid")
    if mid_sharpe is not None and ref_mid is not None and mid_sharpe < ref_mid:
        score -= 0.05
    return decision, reason, score


def _expand_candidates(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        for path in glob.glob(pat):
            p = Path(path)
            if p.exists():
                paths.append(p)
    unique = []
    seen = set()
    for p in paths:
        if str(p) not in seen:
            unique.append(p)
            seen.add(str(p))
    return unique


def build_leaderboard(
    *,
    reference_run_index: Path,
    candidate_run_indexes: List[Path],
    baseline_model_type: str,
    prl_model_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_seed, fallback_ref, ref_metrics = _build_reference(reference_run_index, baseline_model_type)

    rows = []
    for cand_idx_path in candidate_run_indexes:
        idx = _read_run_index(cand_idx_path)
        metrics = _pick_eval_window(_load_with_filter(Path(idx["metrics_path"]), idx.get("run_ids", [])))
        regime = _pick_eval_window(_load_with_filter(Path(idx["regime_metrics_path"]), idx.get("run_ids", [])))
        prl_rows = metrics[metrics["model_type"] == prl_model_type].copy()
        if prl_rows.empty:
            LOGGER.warning("No PRL rows found in %s; skipping", cand_idx_path)
            continue
        timesteps = _load_timesteps(idx.get("config_path", ""))
        mid_rows = regime[(regime["model_type"] == prl_model_type) & (regime["regime"] == "mid")]
        mid_by_seed = {int(r.seed): r for _, r in mid_rows.iterrows()}

        for _, row in prl_rows.iterrows():
            seed = int(row["seed"])
            ref = ref_by_seed.get(seed, fallback_ref)
            cand = {
                "exp_name": idx.get("exp_name", Path(idx.get("config_path", "")).stem),
                "seed": seed,
                "timesteps": timesteps,
                "sharpe_net_exp": float(row.get("sharpe_net_exp", 0.0)),
                "cumulative_return_net_exp": float(row.get("cumulative_return_net_exp", 0.0)),
                "max_drawdown_net_exp": float(row.get("max_drawdown_net_exp", 0.0)),
                "avg_turnover": float(row.get("avg_turnover", 0.0)),
                "sharpe_net_exp_mid": None,
                "cumulative_return_net_exp_mid": None,
            }
            mid_row = mid_by_seed.get(seed)
            if mid_row is not None:
                cand["sharpe_net_exp_mid"] = float(mid_row.get("sharpe_net_exp", 0.0))
                cand["cumulative_return_net_exp_mid"] = float(mid_row.get("cumulative_return_net_exp", 0.0))

            decision, reason, score = _decision_and_score(cand, ref)
            row_out = {
                **cand,
                "baseline_ref_sharpe_net_exp": ref["sharpe_net_exp"],
                "baseline_ref_avg_turnover": ref["avg_turnover"],
                "baseline_ref_max_drawdown_net_exp": ref["max_drawdown_net_exp"],
                "baseline_ref_cumulative_return_net_exp": ref["cumulative_return_net_exp"],
                "baseline_ref_sharpe_net_exp_mid": ref.get("sharpe_net_exp_mid"),
                "baseline_ref_cumret_net_exp_mid": ref.get("cumulative_return_net_exp_mid"),
                "delta_sharpe_net_exp_vs_ref": cand["sharpe_net_exp"] - ref["sharpe_net_exp"],
                "delta_turnover_vs_ref": cand["avg_turnover"] - ref["avg_turnover"],
                "score": score,
                "decision": decision,
                "decision_reason": reason,
                "run_index_path": str(cand_idx_path),
            }
            rows.append(row_out)

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["decision", "score"], ascending=[False, False]).reset_index(drop=True)
    return leaderboard, ref_metrics


def _write_summary(
    summary_path: Path,
    leaderboard: pd.DataFrame,
    reference_run_index: Path,
) -> None:
    lines = []
    lines.append("# Gate1 summary (reference baseline + PRL candidates)")
    lines.append("")
    lines.append(f"- Reference run_index: {reference_run_index}")
    if leaderboard.empty:
        lines.append("- No candidate rows found; check run_index paths.")
        summary_path.write_text("\n".join(lines))
        return

    top = leaderboard.head(5)
    lines.append("- PASS rules: T1 (turnover <= 70% of ref and Sharpe >= ref) or T2 (Sharpe >= ref + 0.10).")
    lines.append("- FAIL rules: Sharpe <= ref - 0.05 or turnover >= 110% of ref.")
    lines.append("- Score = sharpe_net_exp - 0.25*|mdd_net_exp| - 0.10*avg_turnover (mid worse than ref -> -0.05 penalty).")
    lines.append("- Gate1 is 방향성 확인 단계: 통계 검정은 참고용이며 판정은 지표/스코어 기반.")
    lines.append("")
    lines.append("## Top candidates (by score)")
    lines.append("")
    lines.append("| exp_name | seed | score | decision | delta_sharpe_net_exp_vs_ref | delta_turnover_vs_ref |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['exp_name']} | {row['seed']} | {row['score']:.4f} | {row['decision']} | "
            f"{row['delta_sharpe_net_exp_vs_ref']:.4f} | {row['delta_turnover_vs_ref']:.4f} |"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build Gate1 leaderboard from reference baseline + PRL candidates.")
    parser.add_argument("--reference-run-index", required=True, help="run_index.json path for reference baseline_sac.")
    parser.add_argument(
        "--candidate-run-indexes",
        nargs="+",
        required=True,
        help="run_index.json paths or globs for PRL candidates.",
    )
    parser.add_argument("--output-dir", default="outputs/exp_runs/gate1", help="Output directory for leaderboard/summary.")
    parser.add_argument("--baseline-model-type", default="baseline_sac")
    parser.add_argument("--prl-model-type", default="prl_sac")
    args = parser.parse_args()

    ref_path = Path(args.reference_run_index).resolve()
    candidates = [p for p in _expand_candidates(args.candidate_run_indexes) if p.resolve() != ref_path]
    if not candidates:
        raise SystemExit("No candidate run_index paths resolved.")

    leaderboard, ref_metrics = build_leaderboard(
        reference_run_index=ref_path,
        candidate_run_indexes=candidates,
        baseline_model_type=args.baseline_model_type,
        prl_model_type=args.prl_model_type,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "Gate1_leaderboard.csv"
    summary_path = output_dir / "gate1_summary.md"
    ref_row_path = output_dir / "reference_row.csv"

    leaderboard.to_csv(leaderboard_path, index=False)
    _write_summary(summary_path, leaderboard, Path(args.reference_run_index))

    # Save the reference baseline rows for traceability.
    ref_baseline = ref_metrics
    if "model_type" in ref_metrics.columns:
        ref_baseline = ref_metrics[ref_metrics["model_type"] == args.baseline_model_type]
    ref_baseline.to_csv(ref_row_path, index=False)

    LOGGER.info("Wrote %s, %s, %s", leaderboard_path, summary_path, ref_row_path)


if __name__ == "__main__":
    main()
