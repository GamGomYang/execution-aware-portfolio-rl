import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd


def _collect_run_indexes(paths: Iterable[Path]) -> List[dict]:
    run_indexes = []
    for path in paths:
        if path.is_dir():
            path = path / "reports" / "run_index.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            data["_run_index_path"] = str(path)
            run_indexes.append(data)
        except Exception:
            continue
    return run_indexes


def _summarize_gate(idx: dict, reports_dir: Path) -> Dict:
    metrics_path = Path(idx.get("metrics_path", reports_dir / "metrics.csv"))
    regime_path = Path(idx.get("regime_metrics_path", reports_dir / "regime_metrics.csv"))
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    regime_df = pd.read_csv(regime_path) if regime_path.exists() else pd.DataFrame()
    model_scores = []
    if not metrics_df.empty:
        for model_type, group in metrics_df.groupby("model_type"):
            net_sharpe = group["sharpe_net_exp"].mean()
            gross_sharpe = group["sharpe"].mean() if "sharpe" in group else float("nan")
            net_cum = group["cumulative_return_net_exp"].mean()
            gross_cum = group["cumulative_return"].mean() if "cumulative_return" in group else float("nan")
            model_scores.append(
                {
                    "model_type": model_type,
                    "sharpe_net_exp": net_sharpe,
                    "sharpe_gross": gross_sharpe,
                    "cumulative_return_net_exp": net_cum,
                    "cumulative_return_gross": gross_cum,
                }
            )
    mid_regime = None
    if not regime_df.empty and "regime" in regime_df.columns:
        mid = regime_df[regime_df["regime"] == "mid"]
        if not mid.empty:
            mid_regime = {
                "sharpe_net_exp": mid["sharpe_net_exp"].mean() if "sharpe_net_exp" in mid else float("nan"),
                "cumulative_return_net_exp": mid["cumulative_return_net_exp"].mean() if "cumulative_return_net_exp" in mid else float("nan"),
            }
    return {
        "exp_name": idx.get("exp_name") or idx.get("name"),
        "config_path": idx.get("config_path"),
        "eval_windows": idx.get("eval_windows"),
        "run_ids": idx.get("run_ids", []),
        "model_scores": model_scores,
        "mid_regime": mid_regime,
        "reports_dir": str(reports_dir),
    }


def _pick_final_candidates(gate_summaries: List[Dict]) -> pd.DataFrame:
    rows = []
    for gate in gate_summaries:
        for score in gate.get("model_scores", []):
            rows.append(
                {
                    "gate": gate["exp_name"],
                    "config": gate["config_path"],
                    "model_type": score["model_type"],
                    "sharpe_net_exp": score.get("sharpe_net_exp"),
                    "cumulative_return_net_exp": score.get("cumulative_return_net_exp"),
                    "sharpe_gross": score.get("sharpe_gross"),
                    "cumulative_return_gross": score.get("cumulative_return_gross"),
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["sharpe_net_exp", "cumulative_return_net_exp"], ascending=False)
    return df


def _render_roadmap_markdown(gate_summaries: List[Dict]) -> str:
    lines = []
    lines.append("# Roadmap Results")
    for gate in gate_summaries:
        lines.append(f"## {gate['exp_name']}")
        lines.append(f"- config: {gate.get('config_path')}")
        lines.append(f"- eval_windows: {gate.get('eval_windows')}")
        lines.append(f"- run_ids: {', '.join(gate.get('run_ids', []))}")
        if gate.get("model_scores"):
            lines.append("### Model Metrics (net_exp primary)")
            for score in gate["model_scores"]:
                lines.append(
                    f"- {score['model_type']}: sharpe_net_exp={score['sharpe_net_exp']:.3f}, cumret_net_exp={score['cumulative_return_net_exp']:.4f} "
                    f"(gross sharpe={score['sharpe_gross']:.3f}, gross cumret={score['cumulative_return_gross']:.4f})"
                )
        if gate.get("mid_regime"):
            mid = gate["mid_regime"]
            lines.append(f"- Mid regime (avg): sharpe_net_exp={mid.get('sharpe_net_exp'):.3f}, cumret_net_exp={mid.get('cumulative_return_net_exp'):.4f}")
        lines.append("")
    return "\n".join(lines)


def _render_final_decision(candidates: pd.DataFrame) -> str:
    lines = ["# Final Decision"]
    if candidates.empty:
        lines.append("No candidates available.")
        return "\n".join(lines)
    top = candidates.iloc[0]
    lines.append(f"- Selected gate: {top['gate']}")
    lines.append(f"- Config: {top['config']}")
    lines.append(f"- Model: {top['model_type']}")
    lines.append(f"- Sharpe(net_exp): {top['sharpe_net_exp']:.3f}, CumReturn(net_exp): {top['cumulative_return_net_exp']:.4f}")
    lines.append(f"- Gross (ref): Sharpe={top['sharpe_gross']:.3f}, CumReturn={top['cumulative_return_gross']:.4f}")
    lines.append("")
    lines.append("## Rationale")
    lines.append("- Net_exp metrics prioritized; gross provided as reference.")
    lines.append("- Consider mid-regime improvement and cost sensitivity if available.")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate exp_runs outputs into roadmap artifacts.")
    parser.add_argument(
        "--run-index-paths",
        nargs="+",
        default=[],
        help="Paths to run_index.json or their parent output roots (expects reports/run_index.json).",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Where to write roadmap artifacts.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.run_index_paths:
        exp_runs = Path("outputs/exp_runs")
        run_indexes = _collect_run_indexes(exp_runs.glob("**/run_index.json"))
    else:
        run_indexes = _collect_run_indexes([Path(p) for p in args.run_index_paths])
    gate_summaries = []
    for idx in run_indexes:
        reports_dir = Path(idx.get("reports_dir") or Path(idx["_run_index_path"]).parent)
        gate_summaries.append(_summarize_gate(idx, reports_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roadmap_md = _render_roadmap_markdown(gate_summaries)
    (output_dir / "roadmap_results.md").write_text(roadmap_md)
    candidates_df = _pick_final_candidates(gate_summaries)
    candidates_path = output_dir / "final_candidates_table.csv"
    if not candidates_df.empty:
        candidates_df.to_csv(candidates_path, index=False)
    else:
        candidates_path.write_text("")
    final_decision_md = _render_final_decision(candidates_df)
    (output_dir / "final_decision.md").write_text(final_decision_md)
    print("Wrote roadmap artifacts to", output_dir)


if __name__ == "__main__":
    main()
