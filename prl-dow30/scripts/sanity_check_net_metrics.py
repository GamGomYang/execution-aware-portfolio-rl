import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_METRIC_COLS = {
    "sharpe_net_exp",
    "cumulative_return_net_exp",
    "max_drawdown_net_exp",
}

REQUIRED_REGIME_COLS = {
    "sharpe_net_exp",
    "cumulative_return_net_exp",
    "max_drawdown_net_exp",
}


class SanityError(Exception):
    """Raised when a sanity check fails."""


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, name: str, path: Path) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise SanityError(f"{name} is missing columns {sorted(missing)} at {path}")


def _check_trace(trace_path: Path, *, tol: float) -> bool:
    df = pd.read_parquet(trace_path)
    if df.empty or "reward" not in df.columns or "net_return_exp" not in df.columns:
        return False
    rewards = pd.to_numeric(df["reward"], errors="coerce")
    net_exp = pd.to_numeric(df["net_return_exp"], errors="coerce")
    mask = rewards.notna() & net_exp.notna()
    if not mask.any():
        return False
    delta = np.abs(np.expm1(rewards[mask].to_numpy()) - net_exp[mask].to_numpy())
    if np.nanmax(delta) > tol:
        raise SanityError(f"Trace mismatch: exp(reward)-1 != net_return_exp in {trace_path}")
    return True


def run_checks(output_root: Path, *, trace_tolerance: float = 1e-8, require_trace: bool = False) -> None:
    reports_dir = output_root / "reports"
    metrics_path = reports_dir / "metrics.csv"
    regime_path = reports_dir / "regime_metrics.csv"

    if not metrics_path.exists():
        raise SanityError(f"metrics.csv not found under {reports_dir}")
    metrics_df = pd.read_csv(metrics_path)
    _require_columns(metrics_df, REQUIRED_METRIC_COLS, name="metrics.csv", path=metrics_path)

    if not regime_path.exists():
        raise SanityError(f"regime_metrics.csv not found under {reports_dir}")
    regime_df = pd.read_csv(regime_path)
    _require_columns(regime_df, REQUIRED_REGIME_COLS, name="regime_metrics.csv", path=regime_path)

    trace_paths = sorted(reports_dir.glob("trace_*.parquet"))
    checked = False
    for trace_path in trace_paths:
        checked |= _check_trace(trace_path, tol=trace_tolerance)

    if require_trace and not checked:
        raise SanityError("No trace files contained both reward and net_return_exp to verify.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check net metrics outputs.")
    parser.add_argument("--output-root", type=str, default="outputs", help="Base output directory (default: outputs).")
    parser.add_argument(
        "--trace-tolerance",
        type=float,
        default=1e-8,
        help="Absolute tolerance for exp(reward)-1 vs net_return_exp comparisons.",
    )
    parser.add_argument(
        "--require-trace",
        action="store_true",
        help="Fail if no trace contains reward and net_return_exp columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_checks(Path(args.output_root), trace_tolerance=float(args.trace_tolerance), require_trace=bool(args.require_trace))
    except SanityError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
    print("[OK] net metrics sanity checks passed.")


if __name__ == "__main__":
    main()
