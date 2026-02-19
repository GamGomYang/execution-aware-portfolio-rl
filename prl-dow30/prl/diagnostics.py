from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .baselines import run_baseline_strategy_detailed
from .data import MarketData, load_market_data, slice_frame
from .features import VolatilityFeatures, compute_volatility_features
from .metrics import compute_metrics
from .signals import compute_signal_frames, parse_signal_list
from .utils.signature import canonical_json, sha256_bytes

LOGGER = logging.getLogger(__name__)


BASELINE_MAP = {
    "equal_weight": "buy_and_hold_equal_weight",
    "daily_rebalanced_equal_weight": "daily_rebalanced_equal_weight",
    "inverse_vol_risk_parity": "inverse_vol_risk_parity",
}


@dataclass
class BaselineDiagnostics:
    metrics: pd.DataFrame
    returns: Dict[str, pd.Series]
    raw_metrics: Dict[str, Any]
    alignment_dropped_days: int


@dataclass
class BetaDiagnostics:
    report: pd.DataFrame
    alignment_dropped_days: int


@dataclass
class MomentumDiagnostics:
    ic_timeseries: pd.DataFrame
    ic_summary: Dict[str, float]
    longshort_curve: pd.DataFrame | None
    dropped_due_to_lookback: int
    dropped_due_to_nan_alignment: int


@dataclass
class MultiSignalDiagnostics:
    ic_summary: pd.DataFrame
    ic_timeseries: pd.DataFrame
    longshort_summary: pd.DataFrame


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["config_path"] = str(cfg_path)
    return cfg


def _config_hash(config: Dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True).encode("utf-8")
    return sha256_bytes(blob)


def _manifest_hash(manifest: Dict[str, Any]) -> str:
    if not manifest:
        return ""
    if "data_manifest_hash" not in manifest:
        payload = {key: value for key, value in manifest.items() if key != "data_manifest_hash"}
        return sha256_bytes(canonical_json(payload))
    return str(manifest.get("data_manifest_hash") or "")


def _vol_stats_filename(config: Dict[str, Any], lv: int, manifest_hash: str) -> str:
    dates = config["dates"]
    stats_key = manifest_hash or _config_hash(config)
    return f"vol_stats_{stats_key[:8]}_Lv{lv}_{dates['train_start']}_{dates['train_end']}.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def prepare_market_and_features(
    config: Dict[str, Any],
    *,
    cache_only: bool = False,
) -> Tuple[MarketData, VolatilityFeatures]:
    dates = config["dates"]
    data_cfg = {**config.get("data", {})}
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    data_cfg.update(
        {
            "raw_dir": data_cfg.get("raw_dir", "data/raw"),
            "processed_dir": processed_dir,
            "offline": data_cfg.get("offline", False),
            "require_cache": data_cfg.get("require_cache", False),
            "paper_mode": data_cfg.get("paper_mode", False),
            "session_opts": data_cfg.get("session_opts"),
        }
    )
    load_cfg = {
        **config,
        "dates": {"train_start": dates["train_start"], "test_end": dates["test_end"]},
        "data": data_cfg,
    }
    prices, returns, manifest, quality_summary = load_market_data(
        load_cfg,
        offline=bool(data_cfg.get("offline", False)),
        require_cache=bool(data_cfg.get("require_cache", False)),
        cache_only=cache_only,
        force_refresh=bool(data_cfg.get("force_refresh", True)),
    )
    market = MarketData(prices=prices, returns=returns, manifest=manifest, quality_summary=quality_summary)
    lv = int(config.get("env", {}).get("Lv"))
    manifest_hash = _manifest_hash(manifest)
    stats_filename = _vol_stats_filename(config, lv, manifest_hash)
    vol_features = compute_volatility_features(
        returns=market.returns,
        lv=lv,
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        processed_dir=processed_dir,
        stats_filename=stats_filename,
    )
    return market, vol_features


def _align_returns_vol(returns: pd.DataFrame, volatility: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    vol_clean = volatility.dropna()
    idx = returns.index.intersection(vol_clean.index)
    aligned_returns = returns.loc[idx]
    aligned_vol = vol_clean.loc[idx]
    dropped = int(len(returns.index) - len(idx))
    return aligned_returns, aligned_vol, dropped


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def _safe_sum(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.nansum(arr))


def run_baselines_test(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float,
    alignment_dropped_days: int = 0,
) -> BaselineDiagnostics:
    metrics_rows: List[Dict[str, Any]] = []
    returns_map: Dict[str, pd.Series] = {}
    metrics_map: Dict[str, Any] = {}
    for display_name, baseline_name in BASELINE_MAP.items():
        metrics, trace = run_baseline_strategy_detailed(
            returns,
            volatility,
            baseline_name,
            transaction_cost=transaction_cost,
        )
        metrics_map[display_name] = metrics
        series = pd.Series(trace["portfolio_returns"], index=pd.to_datetime(trace["dates"]))
        returns_map[display_name] = series
        avg_turnover = _safe_mean(trace["turnovers"])
        total_cost = _safe_sum(trace["costs"])
        metrics_rows.append(
            {
                "strategy": display_name,
                "period": "test",
                "mean_daily_return": float(metrics.mean_daily_return_gross or 0.0),
                "daily_vol": float(metrics.std_daily_return_gross or 0.0),
                "sharpe": float(metrics.sharpe),
                "avg_turnover": avg_turnover,
                "total_cost": total_cost,
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    return BaselineDiagnostics(
        metrics=metrics_df,
        returns=returns_map,
        raw_metrics=metrics_map,
        alignment_dropped_days=int(alignment_dropped_days),
    )


def _ols_alpha_beta(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coeffs[0], coeffs[1]
    return float(alpha), float(beta)


def compute_beta_report(returns_map: Dict[str, pd.Series], *, market_key: str) -> BetaDiagnostics:
    if market_key not in returns_map:
        raise ValueError(f"Market proxy {market_key} missing for beta report.")
    market_returns = returns_map[market_key]
    rows = []
    dropped_total = 0
    for strategy, series in returns_map.items():
        aligned = pd.concat([series, market_returns], axis=1, join="inner")
        aligned.columns = ["strategy", "market"]
        before = len(aligned)
        aligned = aligned.dropna()
        dropped = before - len(aligned)
        dropped_total = max(dropped_total, dropped)
        if aligned.empty:
            raise ValueError("BETA_ALIGNMENT_EMPTY")
        x = aligned["market"].to_numpy(dtype=np.float64)
        y = aligned["strategy"].to_numpy(dtype=np.float64)
        if np.var(x) <= 1e-12:
            raise ValueError("MARKET_RETURN_VARIANCE_ZERO")
        alpha, beta = _ols_alpha_beta(x, y)
        y_hat = alpha + beta * x
        resid = y - y_hat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
        rows.append(
            {
                "strategy": strategy,
                "beta": float(beta),
                "alpha_daily": float(alpha),
                "r2": float(r2),
                "corr": float(corr),
            }
        )
    report = pd.DataFrame(rows)
    return BetaDiagnostics(report=report, alignment_dropped_days=dropped_total)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_momentum_ic(
    prices: pd.DataFrame,
    returns_log: pd.DataFrame,
    *,
    test_start: str,
    test_end: str,
    lookback: int,
    quantile: float,
    include_longshort: bool,
) -> MomentumDiagnostics:
    aligned_prices = prices.loc[returns_log.index]
    returns_arith = np.expm1(returns_log)
    momentum = aligned_prices / aligned_prices.shift(lookback) - 1.0
    returns_tplus1 = returns_arith.shift(-1)
    # Lookahead guard: use t momentum vs t+1 returns via the -1 shift.
    assert returns_tplus1.index.equals(returns_arith.index)

    mom_test = slice_frame(momentum, test_start, test_end)
    ret_test = slice_frame(returns_tplus1, test_start, test_end)

    lookback_drop_mask = mom_test.isna().all(axis=1)
    dropped_lookback = int(lookback_drop_mask.sum())
    candidate_dates = mom_test.index[~lookback_drop_mask]

    ic_values: List[float] = []
    ic_dates: List[pd.Timestamp] = []
    ls_returns: List[float] = []
    ls_dates: List[pd.Timestamp] = []
    dropped_nan = 0

    for date in candidate_dates:
        mom_vec = mom_test.loc[date]
        ret_vec = ret_test.loc[date]
        valid = mom_vec.notna() & ret_vec.notna()
        if int(valid.sum()) < 2:
            dropped_nan += 1
            continue
        mom_valid = mom_vec[valid].to_numpy(dtype=np.float64)
        ret_valid = ret_vec[valid].to_numpy(dtype=np.float64)
        ic = _spearman_corr(mom_valid, ret_valid)
        if not np.isfinite(ic):
            dropped_nan += 1
            continue
        ic_values.append(ic)
        ic_dates.append(pd.to_datetime(date))
        if include_longshort:
            n = int(valid.sum())
            n_top = max(1, int(math.floor(quantile * n)))
            if n_top * 2 > n:
                n_top = max(1, n // 2)
            ranked = pd.Series(mom_vec[valid]).sort_values()
            bottom_idx = ranked.index[:n_top]
            top_idx = ranked.index[-n_top:]
            ls_ret = float(ret_vec.loc[top_idx].mean() - ret_vec.loc[bottom_idx].mean())
            ls_returns.append(ls_ret)
            ls_dates.append(pd.to_datetime(date))

    ic_series = pd.DataFrame(
        {
            "date": ic_dates,
            "ic_spearman": ic_values,
            "lookback": lookback,
        }
    )
    ic_arr = np.asarray(ic_values, dtype=np.float64)
    ic_mean = float(np.mean(ic_arr)) if ic_arr.size else 0.0
    ic_std = float(np.std(ic_arr, ddof=0)) if ic_arr.size else 0.0
    icir = float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0
    tstat = float(ic_mean / (ic_std / math.sqrt(ic_arr.size))) if ic_std > 1e-12 and ic_arr.size else 0.0
    ic_summary = {
        "lookback": int(lookback),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "tstat": tstat,
        "n_days": int(ic_arr.size),
    }

    ls_curve = None
    if include_longshort:
        ls_arr = np.asarray(ls_returns, dtype=np.float64)
        if ls_arr.size:
            log_rewards = []
            for val in ls_arr:
                log_rewards.append(math.log(max(1.0 + float(val), 1e-8)))
            metrics = compute_metrics(
                rewards=log_rewards,
                portfolio_returns=ls_arr,
                turnovers=np.zeros_like(ls_arr),
            )
            cum_equity = np.cumprod(1.0 + ls_arr)
            ls_curve = pd.DataFrame(
                {
                    "date": ls_dates,
                    "r_ls": ls_arr,
                    "cum_equity": cum_equity,
                    "ls_sharpe": float(metrics.sharpe),
                }
            )
        else:
            ls_curve = pd.DataFrame(columns=["date", "r_ls", "cum_equity", "ls_sharpe"])

    return MomentumDiagnostics(
        ic_timeseries=ic_series,
        ic_summary=ic_summary,
        longshort_curve=ls_curve,
        dropped_due_to_lookback=dropped_lookback,
        dropped_due_to_nan_alignment=dropped_nan,
    )


def _mean_std_tstat(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std <= 1e-12:
        return mean, std, 0.0
    tstat = float(mean / (std / math.sqrt(arr.size)))
    return mean, std, tstat


def compute_multi_signal_ic_ls(
    prices: pd.DataFrame,
    returns_log: pd.DataFrame,
    *,
    ic_start: str,
    ic_end: str,
    signals: str | Sequence[str] | None = None,
    quantile: float = 0.30,
    include_longshort: bool = True,
) -> MultiSignalDiagnostics:
    if quantile <= 0.0 or quantile >= 0.5:
        raise ValueError("quantile must be in (0, 0.5)")

    signal_names = parse_signal_list(signals)
    signal_frames = compute_signal_frames(prices, returns_log, signals=signal_names)
    # Fixed protocol: signal_t must be evaluated against return_{t+1}.
    returns_tplus1 = np.expm1(returns_log).shift(-1)

    ic_rows: list[dict[str, Any]] = []
    ic_summary_rows: list[dict[str, Any]] = []
    ls_summary_rows: list[dict[str, Any]] = []

    for signal_name in signal_names:
        signal_slice = slice_frame(signal_frames[signal_name], ic_start, ic_end)
        return_slice = slice_frame(returns_tplus1, ic_start, ic_end)
        idx = signal_slice.index.intersection(return_slice.index)

        ic_values: list[float] = []
        ls_values: list[float] = []
        for date in idx:
            sig_vec = signal_slice.loc[date]
            ret_vec = return_slice.loc[date]
            valid = sig_vec.notna() & ret_vec.notna()
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue

            sig_valid = sig_vec[valid].to_numpy(dtype=np.float64)
            ret_valid = ret_vec[valid].to_numpy(dtype=np.float64)
            ic = _spearman_corr(sig_valid, ret_valid)
            if not np.isfinite(ic):
                continue

            ic_values.append(float(ic))
            ic_rows.append(
                {
                    "date": pd.to_datetime(date),
                    "signal": signal_name,
                    "ic_spearman": float(ic),
                    "n_assets": n_valid,
                }
            )

            if include_longshort:
                n_top = max(1, int(math.floor(quantile * n_valid)))
                if n_top * 2 > n_valid:
                    n_top = max(1, n_valid // 2)
                ranked = pd.Series(sig_vec[valid]).sort_values()
                bottom_idx = ranked.index[:n_top]
                top_idx = ranked.index[-n_top:]
                ls_ret = float(ret_vec.loc[top_idx].mean() - ret_vec.loc[bottom_idx].mean())
                if np.isfinite(ls_ret):
                    ls_values.append(ls_ret)

        ic_mean, ic_std, ic_tstat = _mean_std_tstat(ic_values)
        ic_summary_rows.append(
            {
                "signal": signal_name,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0,
                "tstat": ic_tstat,
                "n_days_ic": int(len(ic_values)),
            }
        )

        if include_longshort:
            ls_mean, ls_std, ls_tstat = _mean_std_tstat(ls_values)
            ls_sharpe = 0.0
            if ls_values:
                ls_arr = np.asarray(ls_values, dtype=np.float64)
                log_rewards = [math.log(max(1.0 + float(v), 1e-8)) for v in ls_arr]
                ls_sharpe = float(
                    compute_metrics(
                        rewards=log_rewards,
                        portfolio_returns=ls_arr,
                        turnovers=np.zeros_like(ls_arr),
                    ).sharpe
                )
            ls_summary_rows.append(
                {
                    "signal": signal_name,
                    "ls_mean": ls_mean,
                    "ls_std": ls_std,
                    "ls_tstat": ls_tstat,
                    "ls_sharpe": ls_sharpe,
                    "n_days": int(len(ls_values)),
                }
            )

    ic_summary = pd.DataFrame(ic_summary_rows)
    if not ic_summary.empty:
        ic_summary = ic_summary.sort_values("signal").reset_index(drop=True)
    else:
        ic_summary = pd.DataFrame(columns=["signal", "ic_mean", "ic_std", "icir", "tstat", "n_days_ic"])

    ic_timeseries = pd.DataFrame(ic_rows)
    if not ic_timeseries.empty:
        ic_timeseries = ic_timeseries.sort_values(["date", "signal"]).reset_index(drop=True)
    else:
        ic_timeseries = pd.DataFrame(columns=["date", "signal", "ic_spearman", "n_assets"])

    if include_longshort:
        longshort_summary = pd.DataFrame(ls_summary_rows)
        if not longshort_summary.empty:
            longshort_summary = longshort_summary.sort_values("signal").reset_index(drop=True)
        else:
            longshort_summary = pd.DataFrame(
                columns=["signal", "ls_mean", "ls_std", "ls_tstat", "ls_sharpe", "n_days"]
            )
    else:
        longshort_summary = pd.DataFrame(columns=["signal", "ls_mean", "ls_std", "ls_tstat", "ls_sharpe", "n_days"])

    return MultiSignalDiagnostics(
        ic_summary=ic_summary,
        ic_timeseries=ic_timeseries,
        longshort_summary=longshort_summary,
    )


def select_signals_by_screening(
    ic_summary: pd.DataFrame,
    longshort_summary: pd.DataFrame,
    *,
    tstat_abs_threshold: float = 2.0,
    ls_sharpe_threshold: float = 0.0,
) -> list[str]:
    if ic_summary.empty or longshort_summary.empty:
        return []

    left = ic_summary[["signal", "tstat"]].copy()
    right = longshort_summary[["signal", "ls_sharpe"]].copy()
    merged = left.merge(right, on="signal", how="inner")
    passed = merged[
        (pd.to_numeric(merged["tstat"], errors="coerce").abs() > float(tstat_abs_threshold))
        & (pd.to_numeric(merged["ls_sharpe"], errors="coerce") > float(ls_sharpe_threshold))
    ]
    if passed.empty:
        return []
    return sorted(passed["signal"].astype(str).drop_duplicates().tolist())


def build_manifest(
    *,
    run_id: str,
    config: Dict[str, Any],
    test_start: str,
    test_end: str,
    n_assets: int,
    n_days_test: int,
    lookback_momentum: int,
    dropped_due_to_lookback: int,
    dropped_due_to_nan_alignment: int,
    market_manifest: Dict[str, Any] | None,
) -> Dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "config_path": str(config.get("config_path", "")),
        "config_hash": _config_hash(config),
        "test_start": test_start,
        "test_end": test_end,
        "n_assets": int(n_assets),
        "n_days_test": int(n_days_test),
        "lookback_momentum": int(lookback_momentum),
        "git_commit_hash": _git_commit(),
        "dropped_days_due_to_lookback": int(dropped_due_to_lookback),
        "dropped_days_due_to_nan_alignment": int(dropped_due_to_nan_alignment),
    }
    if market_manifest:
        manifest["data_manifest_hash"] = market_manifest.get("data_manifest_hash") or _manifest_hash(market_manifest)
        manifest["data_manifest_created_at"] = market_manifest.get("created_at")
    return manifest


def derive_diagnosis_message(
    *,
    baselines_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    ic_summary: Dict[str, float],
) -> str:
    sharpe_all_negative = bool((baselines_df["sharpe"] < 0.0).all()) if not baselines_df.empty else False
    betas = beta_df["beta"] if "beta" in beta_df.columns else pd.Series(dtype=float)
    beta_near_one = bool(((betas - 1.0).abs() <= 0.1).all()) if not betas.empty else False
    ic_mean = float(ic_summary.get("ic_mean", 0.0))
    tstat = float(ic_summary.get("tstat", 0.0))
    ic_mean_near_zero = abs(ic_mean) <= 0.01
    ic_tstat_small = abs(tstat) < 2.0
    ic_significant = abs(tstat) >= 2.0

    if sharpe_all_negative and beta_near_one:
        return "tough regime + full exposure (no cash/hedge) likely"
    if ic_mean_near_zero and ic_tstat_small:
        return "predictive signal weak → features likely insufficient"
    if ic_significant and sharpe_all_negative:
        return "signal exists but constraints/costs/long-only may block harvesting"
    if ic_significant:
        return "signal exists but constraints/costs/long-only may block harvesting"
    return "predictive signal weak → features likely insufficient"


def format_sharpe_line(baselines_df: pd.DataFrame) -> str:
    def _lookup(name: str) -> float:
        row = baselines_df.loc[baselines_df["strategy"] == name]
        if row.empty:
            return float("nan")
        return float(row["sharpe"].iloc[0])

    ew = _lookup("equal_weight")
    dr = _lookup("daily_rebalanced_equal_weight")
    inv = _lookup("inverse_vol_risk_parity")
    return f"Baselines(test) Sharpe: EW={ew:.3f}, DR-EW={dr:.3f}, InvVol={inv:.3f}"


def format_beta_line(beta_df: pd.DataFrame) -> str:
    def _lookup(name: str) -> float:
        row = beta_df.loc[beta_df["strategy"] == name]
        if row.empty:
            return float("nan")
        return float(row["beta"].iloc[0])

    ew = _lookup("equal_weight")
    dr = _lookup("daily_rebalanced_equal_weight")
    inv = _lookup("inverse_vol_risk_parity")
    return f"Beta(test): EW beta={ew:.3f}, DR-EW beta={dr:.3f}, InvVol beta={inv:.3f}"


def format_momentum_line(ic_summary: Dict[str, float], diagnosis: str | None = None) -> str:
    mean = float(ic_summary.get("ic_mean", 0.0))
    tstat = float(ic_summary.get("tstat", 0.0))
    n_days = int(ic_summary.get("n_days", 0))
    line = f"Momentum IC(test): mean={mean:.4f}, t={tstat:.2f}, N={n_days}"
    if diagnosis:
        line = f"{line} | {diagnosis}"
    return line
