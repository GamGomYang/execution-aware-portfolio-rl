#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.data import slice_frame
from prl.eval import eval_strategies_to_trace
from prl.train import prepare_market_and_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched external heuristic baselines for a fixed eval window.")
    parser.add_argument("--config", type=str, required=True, help="YAML config used to define the eval window and data.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for baseline artifacts.")
    parser.add_argument("--kappas", nargs="+", type=float, required=True, help="Transaction-cost kappas to evaluate.")
    parser.add_argument("--start", type=str, default="", help="Optional override for eval window start.")
    parser.add_argument("--end", type=str, default="", help="Optional override for eval window end.")
    parser.add_argument("--window-name", type=str, default="", help="Optional label for the eval window.")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback for minimum-variance / mean-variance baselines.")
    parser.add_argument("--history-min", type=int, default=30, help="Minimum history rows before optimization baselines activate.")
    parser.add_argument(
        "--mean-variance-risk-aversion",
        type=float,
        default=10.0,
        help="Risk-aversion coefficient for the long-only mean-variance heuristic.",
    )
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    return parser.parse_args()


def _align_returns_vol(returns: pd.DataFrame, volatility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vol_clean = volatility.dropna(how="any")
    idx = returns.index.intersection(vol_clean.index)
    return returns.loc[idx], vol_clean.loc[idx]


def _compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    final = float(arr[-1])
    if final <= 0.0:
        return float("nan")
    years = float(arr.size) / float(periods_per_year)
    if years <= 0.0:
        return 0.0
    return float(final ** (1.0 / years) - 1.0)


def _compute_max_drawdown(equity: pd.Series) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    run_max = np.maximum.accumulate(arr)
    dd = arr / run_max - 1.0
    return float(np.min(dd))


def _metric_row(
    *,
    window_name: str,
    start: str,
    end: str,
    kappa: float,
    strategy: str,
    trace_df: pd.DataFrame,
    metrics: Any,
    trace_path: Path,
) -> dict[str, Any]:
    return {
        "eval_window": window_name,
        "eval_start": start,
        "eval_end": end,
        "kappa": float(kappa),
        "strategy": strategy,
        "seed": 0,
        "sharpe_net_lin": float(metrics.sharpe_net_lin) if metrics.sharpe_net_lin is not None else float("nan"),
        "cumulative_return_net_lin": (
            float(metrics.cumulative_return_net_lin) if metrics.cumulative_return_net_lin is not None else float("nan")
        ),
        "cagr": _compute_cagr(trace_df["equity_net_lin"]),
        "maxdd": (
            float(metrics.max_drawdown_net_lin)
            if metrics.max_drawdown_net_lin is not None
            else _compute_max_drawdown(trace_df["equity_net_lin"])
        ),
        "avg_turnover_exec": float(metrics.avg_turnover_exec) if metrics.avg_turnover_exec is not None else float("nan"),
        "total_turnover_exec": (
            float(metrics.total_turnover_exec) if metrics.total_turnover_exec is not None else float("nan")
        ),
        "sharpe_annualization": "sqrt(252)",
        "risk_free_rate": 0.0,
        "cost_definition": "kappa * executed_turnover",
        "primary_metric": "executed_path_sharpe_net_lin",
        "trace_path": str(trace_path),
    }


def _strategy_order(strategy: str) -> tuple[int, str]:
    order = {
        "buy_and_hold_equal_weight": 0,
        "daily_rebalanced_equal_weight": 1,
        "inverse_vol_risk_parity": 2,
        "minimum_variance": 3,
        "mean_variance_long_only": 4,
    }
    return order.get(strategy, 999), strategy


def _write_md(path: Path, rows: list[dict[str, Any]], *, config_path: str, start: str, end: str, kappas: list[float]) -> None:
    lines = [
        "# External Heuristic Baselines",
        "",
        f"- config: {config_path}",
        f"- eval_window: {start} to {end}",
        f"- kappas: {', '.join(str(float(k)) for k in kappas)}",
        "- matched definitions:",
        "  same window, same kappa, same Sharpe annualization sqrt(252), rf=0, same executed-path net-linear metrics.",
        "",
    ]
    if not rows:
        lines.append("- no rows")
        path.write_text("\n".join(lines) + "\n")
        return
    df = pd.DataFrame(rows)
    try:
        lines.append(df.to_markdown(index=False))
    except ImportError:
        headers = [str(column) for column in df.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text())
    env_cfg = cfg.get("env", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    dates = cfg.get("dates", {}) or {}
    start = args.start or dates.get("test_start")
    end = args.end or dates.get("test_end")
    if not start or not end:
        raise ValueError("Missing evaluation window start/end.")
    window_name = args.window_name or (cfg.get("eval", {}) or {}).get("name") or "test"

    paper_mode = bool(data_cfg.get("paper_mode", False))
    require_cache_cfg = bool(data_cfg.get("require_cache", False))
    offline_cfg = bool(data_cfg.get("offline", False))
    offline = bool(args.offline or offline_cfg or paper_mode or require_cache_cfg)
    require_cache = bool(require_cache_cfg or paper_mode or offline)
    cache_only = bool(paper_mode or require_cache_cfg or offline_cfg or args.offline)

    market, features = prepare_market_and_features(
        cfg,
        lv=int(env_cfg["Lv"]),
        force_refresh=bool(data_cfg.get("force_refresh", True)),
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=data_cfg.get("session_opts"),
        cache_only=cache_only,
    )
    returns_slice = slice_frame(market.returns, start, end)
    vol_slice = slice_frame(features.volatility, start, end)
    returns_slice, vol_slice = _align_returns_vol(returns_slice, vol_slice)
    if returns_slice.empty or vol_slice.empty:
        raise ValueError("Aligned returns/volatility slices are empty for the requested window.")

    metrics_rows: list[dict[str, Any]] = []
    traces_dir = out_root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for kappa in args.kappas:
        eval_id = f"heuristic_baselines_{window_name}_kappa_{float(kappa):.6g}"
        run_id = f"heuristic_baselines_kappa_{float(kappa):.6g}"
        metrics_by_name, trace_df = eval_strategies_to_trace(
            returns_slice,
            vol_slice,
            transaction_cost=float(kappa),
            eval_id=eval_id,
            run_id=run_id,
            seed=0,
            lookback=int(args.lookback),
            history_min=int(args.history_min),
            mean_variance_risk_aversion=float(args.mean_variance_risk_aversion),
        )
        for strategy, metrics in metrics_by_name.items():
            strategy_trace = trace_df[trace_df["model_type"] == strategy].copy()
            trace_path = traces_dir / f"kappa_{float(kappa):.6g}_{strategy}.parquet"
            strategy_trace.to_parquet(trace_path, index=False)
            metrics_rows.append(
                _metric_row(
                    window_name=window_name,
                    start=start,
                    end=end,
                    kappa=float(kappa),
                    strategy=strategy,
                    trace_df=strategy_trace,
                    metrics=metrics,
                    trace_path=trace_path,
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df["_strategy_rank"] = metrics_df["strategy"].map(lambda s: _strategy_order(str(s))[0])
        metrics_df = metrics_df.sort_values(["kappa", "_strategy_rank", "strategy"]).drop(columns=["_strategy_rank"]).reset_index(drop=True)
    metrics_df.to_csv(out_root / "metrics.csv", index=False)
    metrics_df.to_csv(out_root / "aggregate.csv", index=False)

    protocol = {
        "config": str(config_path),
        "eval_window": {"name": window_name, "start": start, "end": end},
        "kappas": [float(k) for k in args.kappas],
        "matched_definitions": {
            "sharpe_annualization": "sqrt(252)",
            "risk_free_rate": 0.0,
            "cost_definition": "kappa * executed_turnover",
            "primary_metric": "executed_path_sharpe_net_lin",
        },
        "strategies": [
            "buy_and_hold_equal_weight",
            "daily_rebalanced_equal_weight",
            "inverse_vol_risk_parity",
            "minimum_variance",
            "mean_variance_long_only",
        ],
        "heuristic_parameters": {
            "lookback": int(args.lookback),
            "history_min": int(args.history_min),
            "mean_variance_risk_aversion": float(args.mean_variance_risk_aversion),
        },
    }
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    _write_md(
        out_root / "report.md",
        metrics_rows,
        config_path=str(config_path),
        start=start,
        end=end,
        kappas=[float(k) for k in args.kappas],
    )
    print(f"WROTE_METRICS={out_root / 'metrics.csv'}")
    print(f"WROTE_AGGREGATE={out_root / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
