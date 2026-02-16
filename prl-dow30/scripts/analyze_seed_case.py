from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prl.eval import assert_env_compatible, compute_regime_labels, eval_model_to_trace, load_model
from prl.metrics import compute_metrics
from prl.train import build_env_for_range, create_scheduler, prepare_market_and_features


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _prepare_market(cfg: dict):
    data_cfg = cfg.get("data", {})
    market, features = prepare_market_and_features(
        config=cfg,
        lv=cfg["env"]["Lv"],
        force_refresh=data_cfg.get("force_refresh", True),
        offline=bool(data_cfg.get("offline", False) or data_cfg.get("paper_mode", False) or data_cfg.get("require_cache", False)),
        require_cache=bool(data_cfg.get("require_cache", False) or data_cfg.get("paper_mode", False)),
        paper_mode=bool(data_cfg.get("paper_mode", False)),
        cache_only=bool(data_cfg.get("offline", False) or data_cfg.get("paper_mode", False) or data_cfg.get("require_cache", False)),
        session_opts=data_cfg.get("session_opts"),
    )
    return market, features


def _build_eval_env(cfg: dict, market, features, *, seed: int, eval_start: str, eval_end: str):
    env_cfg = cfg["env"]
    return build_env_for_range(
        market=market,
        features=features,
        start=eval_start,
        end=eval_end,
        window_size=env_cfg["L"],
        c_tc=env_cfg["c_tc"],
        seed=seed,
        logit_scale=env_cfg["logit_scale"],
        risk_lambda=env_cfg.get("risk_lambda", 0.0),
        risk_penalty_type=env_cfg.get("risk_penalty_type", "r2"),
        rebalance_eta=env_cfg.get("rebalance_eta"),
    )


def _compute_thresholds(market, features, *, eval_start: str, eval_end: str) -> dict:
    returns = market.returns.loc[eval_start:eval_end]
    volatility = features.volatility.loc[eval_start:eval_end]
    volatility = volatility.dropna()
    idx = returns.index.intersection(volatility.index)
    volatility = volatility.loc[idx]
    portfolio_vol = volatility.mean(axis=1)
    stats = json.loads(Path(features.stats_path).read_text())
    mean = float(stats["mean"])
    std = float(stats["std"])
    vz = (portfolio_vol - mean) / (std + 1e-8)
    q33, q66 = np.quantile(vz.values, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "q33": float(q33),
        "q66": float(q66),
        "vz": vz,
    }


def _label_regime(trace_df: pd.DataFrame, vz_series: pd.Series, thresholds: dict) -> pd.DataFrame:
    df = trace_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    vz_df = pd.DataFrame({"date": pd.to_datetime(vz_series.index), "vz": vz_series.values})
    df = df.merge(vz_df, on="date", how="left")
    df = compute_regime_labels(df, thresholds)
    return df


def _regime_summary(df: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    rows = []
    for regime in ["low", "mid", "high", "all"]:
        sub = df if regime == "all" else df[df["regime"] == regime]
        if sub.empty:
            continue
        metrics = compute_metrics(
            rewards=sub["reward"].tolist(),
            portfolio_returns=sub["portfolio_return"].tolist(),
            turnovers=sub["turnover_exec"].tolist(),
            turnovers_target=sub["turnover_target"].tolist(),
            net_returns_exp=sub["net_return_exp"].tolist(),
            net_returns_lin=sub["net_return_lin"].tolist(),
        )
        gap = pd.to_numeric(sub["turnover_target"], errors="coerce") - pd.to_numeric(sub["turnover_exec"], errors="coerce")
        rows.append(
            {
                "model": model_name,
                "regime": regime,
                "days": int(len(sub)),
                "cumulative_return_net_exp": float(metrics.cumulative_return_net_exp),
                "sharpe_net_exp": float(metrics.sharpe_net_exp),
                "max_drawdown_net_exp": float(metrics.max_drawdown_net_exp),
                "avg_turnover_exec": float(metrics.avg_turnover_exec or 0.0),
                "avg_turnover_target": float(metrics.avg_turnover_target or np.nan),
                "avg_turnover_gap": float(gap.mean()) if len(gap) else float("nan"),
                "median_turnover_gap": float(gap.median()) if len(gap) else float("nan"),
                "cost_sum": float(pd.to_numeric(sub["cost"], errors="coerce").sum()),
                "net_return_exp_mean": float(pd.to_numeric(sub["net_return_exp"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _top_turnover_gap(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    out = df[["date", "regime", "turnover_exec", "turnover_target", "cost", "net_return_exp"]].copy()
    out["turnover_gap"] = pd.to_numeric(out["turnover_target"], errors="coerce") - pd.to_numeric(out["turnover_exec"], errors="coerce")
    out["abs_turnover_gap"] = out["turnover_gap"].abs()
    out = out.sort_values("abs_turnover_gap", ascending=False).head(n).reset_index(drop=True)
    return out


def _write_report(
    out_path: Path,
    *,
    base_run_id: str,
    cand_run_id: str,
    summary_df: pd.DataFrame,
    top_gap: pd.DataFrame,
    drawdown_daily: pd.DataFrame,
    drawdown_by_regime: pd.DataFrame,
    seed: int,
) -> None:
    base_all = summary_df[(summary_df["model"] == "eta_none") & (summary_df["regime"] == "all")].iloc[0]
    cand_all = summary_df[(summary_df["model"] == "eta_010") & (summary_df["regime"] == "all")].iloc[0]
    dd_base_min = float(drawdown_daily["drawdown_base"].min())
    dd_base_min_date = drawdown_daily.loc[drawdown_daily["drawdown_base"].idxmin(), "date"]
    dd_cand_min = float(drawdown_daily["drawdown_cand"].min())
    dd_cand_min_date = drawdown_daily.loc[drawdown_daily["drawdown_cand"].idxmin(), "date"]
    top_dd = drawdown_daily.sort_values("drawdown_improvement", ascending=False).head(10)

    lines = []
    lines.append(f"# Seed {seed} Case Analysis: eta=None vs eta=0.10")
    lines.append("")
    lines.append(f"- base_run_id: `{base_run_id}`")
    lines.append(f"- cand_run_id: `{cand_run_id}`")
    lines.append("")
    lines.append("## Overall (all regime)")
    lines.append("")
    lines.append("| model | sharpe_net_exp | cumret_net_exp | max_drawdown_net_exp | avg_turnover_exec | avg_turnover_target |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in summary_df[summary_df["regime"] == "all"].iterrows():
        lines.append(
            f"| {r['model']} | {r['sharpe_net_exp']:.4f} | {r['cumulative_return_net_exp']:.4f} | "
            f"{r['max_drawdown_net_exp']:.4f} | {r['avg_turnover_exec']:.4f} | {r['avg_turnover_target']:.4f} |"
        )
    lines.append("")
    lines.append("## Regime Breakdown")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Turnover Gap (eta=0.10)")
    lines.append("")
    lines.append(
        f"- mean(turnover_target - turnover_exec): {float(top_gap['turnover_gap'].mean()):.4f} (top-{len(top_gap)} days sample)"
    )
    lines.append(
        f"- median(turnover_target - turnover_exec): {float(top_gap['turnover_gap'].median()):.4f} (top-{len(top_gap)} days sample)"
    )
    lines.append("")
    lines.append(top_gap.to_markdown(index=False))
    lines.append("")
    lines.append("## Drawdown Decomposition")
    lines.append("")
    lines.append(f"- worst drawdown eta=None: {dd_base_min:.4f} on {pd.to_datetime(dd_base_min_date).date()}")
    lines.append(f"- worst drawdown eta=0.10: {dd_cand_min:.4f} on {pd.to_datetime(dd_cand_min_date).date()}")
    lines.append(
        f"- worst drawdown improvement: {(dd_cand_min - dd_base_min):.4f} "
        "(positive means less severe drawdown with eta=0.10)"
    )
    lines.append("")
    lines.append("Top 10 dates by drawdown improvement:")
    lines.append("")
    lines.append(top_dd.to_markdown(index=False))
    lines.append("")
    lines.append("Drawdown improvement by regime (cand - base):")
    lines.append("")
    lines.append(drawdown_by_regime.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- eta=0.10 keeps `turnover_exec` far below `turnover_target`, reducing realized trading cost while preserving target policy direction."
    )
    lines.append(
        "- Improvement concentration can be checked in `top_turnover_gap` and `drawdown_top20` CSV artifacts for manuscript appendix."
    )
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one seed case between eta=None and eta=0.10 runs.")
    parser.add_argument("--output-root", required=True, help="Experiment output root containing models/reports.")
    parser.add_argument("--base-config", required=True, help="Config path for eta=None run.")
    parser.add_argument("--cand-config", required=True, help="Config path for eta=0.10 run.")
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--cand-run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2025-12-30")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    models_dir = output_root / "models"
    out_dir = Path(args.out_dir) if args.out_dir else reports_dir / "archive" / f"seed{args.seed}_case_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_config(Path(args.base_config))
    cand_cfg = _load_config(Path(args.cand_config))
    market, features = _prepare_market(cand_cfg)

    thresholds = _compute_thresholds(market, features, eval_start=args.eval_start, eval_end=args.eval_end)
    q_map = {"q33": thresholds["q33"], "q66": thresholds["q66"]}

    base_env = _build_eval_env(base_cfg, market, features, seed=args.seed, eval_start=args.eval_start, eval_end=args.eval_end)
    cand_env = _build_eval_env(cand_cfg, market, features, seed=args.seed, eval_start=args.eval_start, eval_end=args.eval_end)

    base_meta = json.loads((reports_dir / f"run_metadata_{args.base_run_id}.json").read_text())
    cand_meta = json.loads((reports_dir / f"run_metadata_{args.cand_run_id}.json").read_text())
    assert_env_compatible(base_env, base_meta, Lv=base_cfg["env"].get("Lv"))
    assert_env_compatible(cand_env, cand_meta, Lv=cand_cfg["env"].get("Lv"))

    base_scheduler = create_scheduler(base_cfg["prl"], base_cfg["env"]["L"], market.returns.shape[1], features.stats_path)
    cand_scheduler = create_scheduler(cand_cfg["prl"], cand_cfg["env"]["L"], market.returns.shape[1], features.stats_path)

    base_model = load_model(models_dir / f"{args.base_run_id}_final.zip", "prl", base_env, scheduler=base_scheduler)
    cand_model = load_model(models_dir / f"{args.cand_run_id}_final.zip", "prl", cand_env, scheduler=cand_scheduler)

    _, base_trace = eval_model_to_trace(
        base_model,
        base_env,
        eval_id=args.base_run_id,
        run_id=args.base_run_id,
        model_type="prl_sac",
        seed=args.seed,
    )
    _, cand_trace = eval_model_to_trace(
        cand_model,
        cand_env,
        eval_id=args.cand_run_id,
        run_id=args.cand_run_id,
        model_type="prl_sac",
        seed=args.seed,
    )

    base_trace = _label_regime(base_trace, thresholds["vz"], q_map)
    cand_trace = _label_regime(cand_trace, thresholds["vz"], q_map)
    base_trace.to_csv(out_dir / "trace_base.csv", index=False)
    cand_trace.to_csv(out_dir / "trace_cand.csv", index=False)

    base_reg = _regime_summary(base_trace, model_name="eta_none")
    cand_reg = _regime_summary(cand_trace, model_name="eta_010")
    summary_df = pd.concat([base_reg, cand_reg], ignore_index=True)
    summary_df.to_csv(out_dir / "regime_summary.csv", index=False)

    top_gap = _top_turnover_gap(cand_trace, n=20)
    top_gap.to_csv(out_dir / "top_turnover_gap_eta010.csv", index=False)

    dd_base = _drawdown_series(base_trace["equity_net_exp"].astype(float))
    dd_cand = _drawdown_series(cand_trace["equity_net_exp"].astype(float))
    drawdown_daily = pd.DataFrame(
        {
            "date": pd.to_datetime(base_trace["date"]),
            "drawdown_base": dd_base.values,
            "drawdown_cand": dd_cand.values,
        }
    )
    drawdown_daily["drawdown_improvement"] = drawdown_daily["drawdown_cand"] - drawdown_daily["drawdown_base"]
    drawdown_by_regime = (
        drawdown_daily.merge(cand_trace[["date", "regime"]], on="date", how="left")
        .groupby("regime", dropna=False)["drawdown_improvement"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    drawdown_daily.to_csv(out_dir / "drawdown_daily_compare.csv", index=False)
    drawdown_by_regime.to_csv(out_dir / "drawdown_improvement_by_regime.csv", index=False)
    drawdown_daily.sort_values("drawdown_improvement", ascending=False).head(20).to_csv(
        out_dir / "drawdown_top20_improvement.csv", index=False
    )

    _write_report(
        out_dir / "seed_case_analysis.md",
        base_run_id=args.base_run_id,
        cand_run_id=args.cand_run_id,
        summary_df=summary_df,
        top_gap=top_gap,
        drawdown_daily=drawdown_daily,
        drawdown_by_regime=drawdown_by_regime,
        seed=args.seed,
    )
    print(f"Wrote seed case analysis under: {out_dir}")


if __name__ == "__main__":
    main()
