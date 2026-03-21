from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.diagnostics import (
    _align_returns_vol,
    build_manifest,
    compute_beta_report,
    compute_multi_signal_ic_ls,
    compute_momentum_ic,
    derive_diagnosis_message,
    format_beta_line,
    format_momentum_line,
    format_sharpe_line,
    load_config,
    prepare_market_and_features,
    run_baselines_test,
    select_signals_by_screening,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostics for baseline/beta/momentum root causes.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="YAML config path.")
    parser.add_argument("--outdir", type=str, default="outputs/diagnostics", help="Output root directory.")
    parser.add_argument("--run_id", type=str, required=True, help="Run identifier.")
    parser.add_argument("--include_rl", type=int, default=0, help="Include RL diagnostics (optional).")
    parser.add_argument("--lookback", type=int, default=20, help="Momentum lookback window.")
    parser.add_argument("--momentum_quantile", type=float, default=0.30, help="Top/bottom quantile for LS.")
    parser.add_argument("--include_ls", type=int, default=0, help="Include momentum long-short curve.")
    parser.add_argument("--ic_start", type=str, default=None, help="Signal IC window start date (YYYY-MM-DD).")
    parser.add_argument("--ic_end", type=str, default=None, help="Signal IC window end date (YYYY-MM-DD).")
    parser.add_argument(
        "--signals",
        type=str,
        default="all",
        help="Comma-separated signal list or 'all'.",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    args = parse_args()

    cfg_path = _resolve_path(args.config)
    out_root = _resolve_path(args.outdir)
    run_dir = out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    dates = cfg["dates"]
    env_cfg = cfg.get("env", {})
    data_cfg = cfg.get("data", {})
    test_start = dates["test_start"]
    test_end = dates["test_end"]
    ic_start = args.ic_start or dates["train_start"]
    ic_end = args.ic_end or dates["train_end"]

    if pd.to_datetime(ic_start) > pd.to_datetime(ic_end):
        raise ValueError(f"Invalid IC window: ic_start ({ic_start}) > ic_end ({ic_end})")
    if pd.to_datetime(ic_end) >= pd.to_datetime(test_start):
        raise ValueError(
            "IC screening window must end before test_start to avoid leakage. "
            f"ic_end={ic_end}, test_start={test_start}"
        )

    cache_only = bool(
        data_cfg.get("offline", False) or data_cfg.get("require_cache", False) or data_cfg.get("paper_mode", False)
    )
    market, vol_features = prepare_market_and_features(cfg, cache_only=cache_only)

    returns_test = market.returns.loc[test_start:test_end]
    volatility_test = vol_features.volatility.loc[test_start:test_end]
    returns_aligned, vol_aligned, align_dropped = _align_returns_vol(returns_test, volatility_test)

    transaction_cost = float(env_cfg.get("c_tc", 0.0))
    baselines = run_baselines_test(
        returns_aligned,
        vol_aligned,
        transaction_cost=transaction_cost,
        alignment_dropped_days=align_dropped,
    )

    baselines_df = baselines.metrics.copy()
    baselines_df = baselines_df[
        ["strategy", "period", "mean_daily_return", "daily_vol", "sharpe", "avg_turnover", "total_cost"]
    ]
    _write_csv(run_dir / "baselines_test_metrics.csv", baselines_df)

    beta = compute_beta_report(baselines.returns, market_key="daily_rebalanced_equal_weight")
    beta_df = beta.report.copy()
    beta_df = beta_df[["strategy", "beta", "alpha_daily", "r2", "corr"]]
    _write_csv(run_dir / "beta_report.csv", beta_df)

    momentum = compute_momentum_ic(
        market.prices,
        market.returns,
        test_start=test_start,
        test_end=test_end,
        lookback=int(args.lookback),
        quantile=float(args.momentum_quantile),
        include_longshort=bool(args.include_ls),
    )

    ic_ts = momentum.ic_timeseries.copy()
    if not ic_ts.empty:
        ic_ts["date"] = pd.to_datetime(ic_ts["date"]).dt.strftime("%Y-%m-%d")
    _write_csv(run_dir / "feature_ic_timeseries.csv", ic_ts)
    (run_dir / "feature_ic_summary.json").write_text(json.dumps(momentum.ic_summary, indent=2))

    if momentum.longshort_curve is not None:
        ls_curve = momentum.longshort_curve.copy()
        if not ls_curve.empty:
            ls_curve["date"] = pd.to_datetime(ls_curve["date"]).dt.strftime("%Y-%m-%d")
        _write_csv(run_dir / "momentum_longshort_equity_curve.csv", ls_curve)

    multi_signal = compute_multi_signal_ic_ls(
        market.prices,
        market.returns,
        ic_start=ic_start,
        ic_end=ic_end,
        signals=args.signals,
        quantile=float(args.momentum_quantile),
        include_longshort=bool(args.include_ls),
    )

    signal_ic_summary = multi_signal.ic_summary.copy()
    signal_ic_timeseries = multi_signal.ic_timeseries.copy()
    if not signal_ic_timeseries.empty:
        signal_ic_timeseries["date"] = pd.to_datetime(signal_ic_timeseries["date"]).dt.strftime("%Y-%m-%d")
    signal_ls_summary = multi_signal.longshort_summary.copy()
    _write_csv(run_dir / "signal_ic_summary.csv", signal_ic_summary)
    _write_csv(run_dir / "signal_ic_timeseries.csv", signal_ic_timeseries)
    _write_csv(run_dir / "signal_longshort_summary.csv", signal_ls_summary)

    selected_signals = select_signals_by_screening(
        signal_ic_summary,
        signal_ls_summary,
        tstat_abs_threshold=2.0,
        ls_sharpe_threshold=0.0,
    )
    selected_payload = {
        "ic_start": ic_start,
        "ic_end": ic_end,
        "signals_arg": args.signals,
        "screening": {
            "abs_tstat_gt": 2.0,
            "ls_sharpe_gt": 0.0,
        },
        "selected_signals": selected_signals,
        "n_selected": int(len(selected_signals)),
    }
    (run_dir / "selected_signals.json").write_text(json.dumps(selected_payload, indent=2))

    n_assets = int(market.returns.shape[1])
    n_days_test = int(len(returns_aligned))
    dropped_nan_alignment = int(align_dropped + beta.alignment_dropped_days + momentum.dropped_due_to_nan_alignment)

    manifest = build_manifest(
        run_id=args.run_id,
        config=cfg,
        test_start=test_start,
        test_end=test_end,
        n_assets=n_assets,
        n_days_test=n_days_test,
        lookback_momentum=int(args.lookback),
        dropped_due_to_lookback=momentum.dropped_due_to_lookback,
        dropped_due_to_nan_alignment=dropped_nan_alignment,
        market_manifest=market.manifest,
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    if int(args.include_rl) != 0:
        LOGGER.info("include_rl requested but RL diagnostics are not implemented in this script.")

    diagnosis = derive_diagnosis_message(
        baselines_df=baselines_df,
        beta_df=beta_df,
        ic_summary=momentum.ic_summary,
    )

    print(format_sharpe_line(baselines_df))
    print(format_beta_line(beta_df))
    print(format_momentum_line(momentum.ic_summary, diagnosis=diagnosis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
