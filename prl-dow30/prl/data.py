from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from .data_sources import data_quality_check, fetch_yfinance, hash_file
from .utils.signature import canonical_json, compute_env_signature, sha256_bytes

LOGGER = logging.getLogger(__name__)
ENV_SCHEMA_VERSION = "v1"

DOW30_TICKERS: Sequence[str] = (
    "AAPL",
    "AMGN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WBA",
    "WMT",
    "DOW",
)

DEFAULT_QUALITY_PARAMS: Dict[str, float] = {
    "min_vol_std": 0.002,
    "min_max_abs_return": 0.01,
    "max_flat_fraction": 0.30,
    "max_missing_fraction": 0.05,
}


@dataclass
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    manifest: Dict | None = None
    quality_summary: pd.DataFrame | None = None


def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ratio = prices / prices.shift(1)
    returns = pd.DataFrame(np.log(ratio), index=prices.index, columns=prices.columns)
    returns = returns.replace([np.inf, -np.inf], pd.NA)
    returns = returns.dropna(how="any")
    return returns


def _business_day_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="B")


def _download_single_series(
    ticker: str,
    start: str,
    end: str,
    session_opts: Dict | None,
    ticker_subs: Dict[str, str] | None,
    allow_subs: bool,
) -> tuple[pd.Series | None, str | None, str | None]:
    try:
        df = fetch_yfinance([ticker], start, end, session_opts=session_opts)
        ser = df[ticker] if ticker in df else df.iloc[:, 0]
        ser = ser.dropna()
        ser.name = ticker
        if ser.empty:
            raise RuntimeError("empty series")
        return ser, None, None
    except Exception as exc:
        if allow_subs and ticker_subs and ticker in ticker_subs:
            alt = ticker_subs[ticker]
            try:
                df_alt = fetch_yfinance([alt], start, end, session_opts=session_opts)
                ser_alt = df_alt[alt] if alt in df_alt else df_alt.iloc[:, 0]
                ser_alt = ser_alt.dropna()
                ser_alt.name = ticker
                if ser_alt.empty:
                    raise RuntimeError("empty series from substitution ticker")
                LOGGER.warning("Using substitution ticker %s for %s due to download failure: %s", alt, ticker, exc)
                return ser_alt, None, alt
            except Exception as sub_exc:
                LOGGER.warning("Substitution ticker %s for %s failed: %s", alt, ticker, sub_exc)
        LOGGER.warning("Download failed for %s: %s", ticker, exc)
        return None, "DOWNLOAD_FAILED", None


def _download_all_series(
    tickers: List[str],
    start: str,
    end: str,
    session_opts: Dict | None,
    ticker_substitutions: Dict[str, str],
    allow_subs: bool,
) -> tuple[Dict[str, pd.Series], Dict[str, str], Dict[str, str]]:
    series_map: Dict[str, pd.Series] = {}
    drop_reasons: Dict[str, str] = {}
    subs_used: Dict[str, str] = {}
    for ticker in tickers:
        ser, reason, sub_used = _download_single_series(
            ticker,
            start,
            end,
            session_opts=session_opts,
            ticker_subs=ticker_substitutions,
            allow_subs=allow_subs,
        )
        if ser is not None:
            series_map[ticker] = ser
        if reason:
            drop_reasons[ticker] = reason
        if sub_used:
            subs_used[ticker] = sub_used
    return series_map, drop_reasons, subs_used


def _align_raw_prices(
    tickers: List[str], series_map: Dict[str, pd.Series], start: str, end: str
) -> pd.DataFrame:
    calendar = _business_day_index(start, end)
    frame = pd.DataFrame(index=calendar)
    for ticker in tickers:
        ser = series_map.get(ticker, pd.Series(dtype=float, name=ticker))
        frame[ticker] = ser.reindex(calendar)
    return frame


def _drop_market_closed_days(raw_prices: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float | int | str]]:
    """
    Remove rows where all tickers are NaN (market-closed days created by freq='B' alignment).
    Returns: (cleaned_prices, info_dict).
    """
    total = int(len(raw_prices))
    if raw_prices.empty:
        info = {
            "calendar_freq": "B",
            "market_closed_days_removed": 0,
            "market_closed_fraction_removed": 0.0,
            "market_closed_rule": "drop_all_nan_rows_before_missing_fraction",
            "total_days_before_drop": 0,
            "total_days_after_drop": 0,
        }
        return raw_prices, info
    mask = raw_prices.isna().all(axis=1)
    removed = int(mask.sum())
    cleaned = raw_prices.loc[~mask].copy() if removed else raw_prices
    removed_fraction = removed / max(total, 1)
    info = {
        "calendar_freq": "B",
        "market_closed_days_removed": removed,
        "market_closed_fraction_removed": float(removed_fraction),
        "market_closed_rule": "drop_all_nan_rows_before_missing_fraction",
        "total_days_before_drop": total,
        "total_days_after_drop": int(len(cleaned)),
    }
    return cleaned, info


def _align_prices_raw_then_fill(
    raw_prices: pd.DataFrame,
    *,
    ffill: bool = True,
    bfill: bool = True,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    raw_missing_fraction = raw_prices.isna().mean().to_dict()
    filled = raw_prices.copy()
    if ffill:
        filled = filled.ffill()
    if bfill:
        filled = filled.bfill()
    prices = filled.dropna(how="any")
    return prices, raw_missing_fraction


def _compute_raw_metrics(
    raw_prices: pd.DataFrame, raw_missing_fraction: Dict[str, float] | None = None
) -> Dict[str, Dict]:
    metrics: Dict[str, Dict] = {}
    missing_fraction = raw_missing_fraction or {}
    for ticker in raw_prices.columns:
        ser = raw_prices[ticker]
        valid = ser.dropna()
        first_valid_date = pd.NaT if valid.empty else valid.index[0]
        if len(ser):
            miss = float(missing_fraction.get(ticker, ser.isna().mean()))
            if pd.isna(miss):
                miss = 1.0
        else:
            miss = 1.0
        metrics[ticker] = {
            "missing_fraction": miss,
            "valid_obs_count": int(len(valid)),
            "first_valid_date": None if pd.isna(first_valid_date) else pd.to_datetime(first_valid_date),
        }
    return metrics


def _select_universe(
    tickers: List[str],
    raw_metrics: Dict[str, Dict],
    drop_reasons: Dict[str, str],
    start: str,
    min_history_days: int,
    history_tolerance_days: int,
    max_missing_fraction: float,
) -> tuple[list[str], list[str], Dict[str, str]]:
    kept: list[str] = []
    dropped: list[str] = []
    start_ts = pd.to_datetime(start)
    tolerance_cutoff = start_ts + pd.Timedelta(days=history_tolerance_days)
    for ticker in tickers:
        metrics = raw_metrics.get(ticker, {})
        if ticker in drop_reasons:
            dropped.append(ticker)
            continue
        first_valid = metrics.get("first_valid_date")
        if first_valid is None:
            drop_reasons[ticker] = "DOWNLOAD_FAILED"
            dropped.append(ticker)
            continue
        if first_valid > tolerance_cutoff:
            drop_reasons[ticker] = "FIRST_DATE_TOO_LATE"
            dropped.append(ticker)
            continue
        if metrics.get("valid_obs_count", 0) < min_history_days:
            drop_reasons[ticker] = "INSUFFICIENT_VALID_DAYS"
            dropped.append(ticker)
            continue
        if metrics.get("missing_fraction", 1.0) > max_missing_fraction:
            drop_reasons[ticker] = "RAW_MISSING_FRACTION_EXCEEDED"
            dropped.append(ticker)
            continue
        kept.append(ticker)
    return kept, dropped, drop_reasons


def _build_quality_summary(
    raw_prices: pd.DataFrame,
    cleaned_prices: pd.DataFrame,
    returns: pd.DataFrame,
    raw_metrics: Dict[str, Dict],
    kept: list[str],
    drop_reasons: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    kept_set = set(kept)
    arithmetic_returns = returns.apply(np.expm1) if not returns.empty else pd.DataFrame()
    for ticker in raw_prices.columns:
        metrics = raw_metrics.get(ticker, {})
        first_valid = metrics.get("first_valid_date")
        kept_flag = ticker in kept_set
        drop_reason = drop_reasons.get(ticker)
        row = {
            "ticker": ticker,
            "kept": kept_flag,
            "drop_reason": drop_reason or "",
            "missing_fraction_raw": metrics.get("missing_fraction", 1.0),
            "valid_obs_count": metrics.get("valid_obs_count", 0),
            "first_valid_date": None if first_valid is None else pd.to_datetime(first_valid).date().isoformat(),
        }
        if kept_flag:
            p = cleaned_prices[ticker]
            r = arithmetic_returns[ticker] if ticker in arithmetic_returns else pd.Series(dtype=float)
            row.update(
                {
                    "return_std": float(r.std()),
                    "max_abs_return": float(r.abs().max()),
                    "flat_fraction": float((p.diff().fillna(0.0).abs() < 1e-9).mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _load_processed_cache(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
    prices_path = processed_dir / "prices.parquet"
    returns_path = processed_dir / "returns.parquet"
    manifest_path = processed_dir / "data_manifest.json"
    quality_path = processed_dir / "quality_summary.csv"
    if not prices_path.exists() or not returns_path.exists():
        raise RuntimeError("CACHE_MISSING: processed parquet files not found; run build_cache.py")
    LOGGER.info("Loading processed cache from %s", processed_dir)
    prices = pd.read_parquet(prices_path)
    returns = pd.read_parquet(returns_path)
    manifest = {}
    quality_summary = pd.DataFrame()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        kept = manifest.get("kept_tickers")
        if kept:
            missing_kept = [t for t in kept if t not in prices.columns]
            if missing_kept:
                LOGGER.warning("Manifest kept_tickers missing in prices cache: %s", missing_kept)
            else:
                prices = prices[kept]
                returns = returns[kept]
        if "asset_list" not in manifest and kept:
            manifest["asset_list"] = kept
        if "num_assets" not in manifest and kept:
            manifest["num_assets"] = len(kept)
        if "data_manifest_hash" not in manifest:
            manifest["data_manifest_hash"] = _compute_data_manifest_hash(manifest)
    if quality_path.exists():
        quality_summary = pd.read_csv(quality_path)
    return prices, returns, manifest, quality_summary


def _manifest_payload(
    processed_dir: Path,
    start: str,
    end: str,
    requested_tickers: list[str],
    kept_tickers: list[str],
    dropped_reasons: Dict[str, str],
    processed_hashes: Dict[str, str],
    quality_params: Dict,
    min_history_days: int,
    min_assets: int,
    history_tolerance_days: int,
    universe_policy: str,
    substitutions_used: Dict[str, str],
    market_closed_days_removed: int,
    market_closed_fraction_removed: float,
    total_days_before_drop: int,
    total_days_after_drop: int,
    L: int | None,
    Lv: int | None,
    feature_flags: Dict,
    cost_params: Dict,
    schema_version: str,
) -> Dict:
    num_assets = len(kept_tickers)
    obs_dim_expected = num_assets * (int(L) + 2) if L is not None else None
    env_signature_hash = None
    if L is not None and Lv is not None:
        env_signature_hash = compute_env_signature(
            kept_tickers,
            int(L),
            int(Lv),
            feature_flags=feature_flags,
            cost_params=cost_params,
            schema_version=schema_version,
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "universe_policy": universe_policy,
        "calendar_freq": "B",
        "market_closed_days_removed": market_closed_days_removed,
        "market_closed_fraction_removed": market_closed_fraction_removed,
        "market_closed_rule": "drop_all_nan_rows_before_missing_fraction",
        "total_days_before_drop": total_days_before_drop,
        "total_days_after_drop": total_days_after_drop,
        "requested_tickers": requested_tickers,
        "kept_tickers": kept_tickers,
        "dropped_tickers": [t for t in requested_tickers if t in dropped_reasons],
        "dropped_reasons": dropped_reasons,
        "N_assets_final": len(kept_tickers),
        "asset_list": kept_tickers,
        "num_assets": num_assets,
        "L": L,
        "Lv": Lv,
        "obs_dim_expected": obs_dim_expected,
        "env_signature_hash": env_signature_hash,
        "env_schema_version": schema_version,
        "feature_flags": feature_flags,
        "cost_params": cost_params,
        "source": "yfinance_only",
        "price_type": "adj_close",
        "processed_hashes": processed_hashes,
        "quality_params": quality_params,
        "min_history_days": min_history_days,
        "min_assets": min_assets,
        "history_tolerance_days": history_tolerance_days,
        "python_version": sys.version,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
    }
    if substitutions_used:
        manifest["substitutions_used"] = substitutions_used
    return manifest


def _compute_data_manifest_hash(manifest: Dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "data_manifest_hash"}
    return sha256_bytes(canonical_json(payload))


def _write_manifest(path: Path, manifest: Dict) -> Dict:
    manifest_with_hash = dict(manifest)
    manifest_with_hash["data_manifest_hash"] = _compute_data_manifest_hash(manifest_with_hash)
    path.write_text(json.dumps(manifest_with_hash, indent=2))
    return manifest_with_hash


def _ensure_manifest_env_fields(
    manifest: Dict,
    asset_list: list[str],
    L: int | None,
    Lv: int | None,
    feature_flags: Dict,
    cost_params: Dict,
) -> Dict:
    updated = dict(manifest)
    if "asset_list" not in updated:
        updated["asset_list"] = asset_list
    if "num_assets" not in updated:
        updated["num_assets"] = len(asset_list)
    if "L" not in updated:
        updated["L"] = L
    if "Lv" not in updated:
        updated["Lv"] = Lv
    if "obs_dim_expected" not in updated and L is not None:
        updated["obs_dim_expected"] = len(asset_list) * (int(L) + 2)
    if "env_schema_version" not in updated:
        updated["env_schema_version"] = ENV_SCHEMA_VERSION
    if "feature_flags" not in updated:
        updated["feature_flags"] = feature_flags
    if "cost_params" not in updated:
        updated["cost_params"] = cost_params
    if "env_signature_hash" not in updated and L is not None and Lv is not None:
        updated["env_signature_hash"] = compute_env_signature(
            asset_list,
            int(L),
            int(Lv),
            feature_flags=feature_flags,
            cost_params=cost_params,
            schema_version=updated.get("env_schema_version", ENV_SCHEMA_VERSION),
        )
    if "data_manifest_hash" not in updated:
        updated["data_manifest_hash"] = _compute_data_manifest_hash(updated)
    return updated


def _save_artifacts(
    processed_dir: Path,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    quality_summary: pd.DataFrame,
    manifest_kwargs: Dict,
) -> tuple[Dict, pd.DataFrame]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    prices_path = processed_dir / "prices.parquet"
    returns_path = processed_dir / "returns.parquet"
    quality_path = processed_dir / "quality_summary.csv"
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_quality_path = reports_dir / "data_quality_summary.csv"
    prices_tmp = prices_path.with_suffix(prices_path.suffix + ".tmp")
    returns_tmp = returns_path.with_suffix(returns_path.suffix + ".tmp")
    quality_tmp = quality_path.with_suffix(quality_path.suffix + ".tmp")
    report_quality_tmp = report_quality_path.with_suffix(report_quality_path.suffix + ".tmp")
    manifest_path = processed_dir / "data_manifest.json"
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    try:
        prices.to_parquet(prices_tmp)
        returns.to_parquet(returns_tmp)
        quality_summary.to_csv(quality_tmp, index=False)
        quality_summary.to_csv(report_quality_tmp, index=False)
        processed_hashes = {
            "prices.parquet": hash_file(prices_tmp),
            "returns.parquet": hash_file(returns_tmp),
        }
        manifest_payload = _manifest_payload(processed_dir, processed_hashes=processed_hashes, **manifest_kwargs)
        manifest_payload = _write_manifest(manifest_tmp, manifest_payload)
    except Exception:
        for tmp in [prices_tmp, returns_tmp, quality_tmp, report_quality_tmp, manifest_tmp]:
            if tmp.exists():
                tmp.unlink()
        raise

    prices_tmp.replace(prices_path)
    returns_tmp.replace(returns_path)
    quality_tmp.replace(quality_path)
    report_quality_tmp.replace(report_quality_path)
    manifest_tmp.replace(manifest_path)
    return manifest_payload, quality_summary


def _build_failed_quality_summary(
    raw_prices: pd.DataFrame,
    raw_metrics: Dict[str, Dict],
    kept: list[str],
    drop_reasons: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    kept_set = set(kept)
    for ticker in raw_prices.columns:
        metrics = raw_metrics.get(ticker, {})
        first_valid = metrics.get("first_valid_date")
        rows.append(
            {
                "ticker": ticker,
                "kept": ticker in kept_set,
                "drop_reason": drop_reasons.get(ticker, ""),
                "missing_fraction_raw": metrics.get("missing_fraction", 1.0),
                "valid_obs_count": metrics.get("valid_obs_count", 0),
                "first_valid_date": None if first_valid is None else pd.to_datetime(first_valid).date().isoformat(),
                "return_std": np.nan,
                "max_abs_return": np.nan,
                "flat_fraction": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_failed_artifacts(
    processed_dir: Path,
    *,
    start: str,
    end: str,
    requested_tickers: list[str],
    kept_tickers: list[str],
    drop_reasons: Dict[str, str],
    raw_metrics: Dict[str, Dict],
    min_assets: int,
    error_code: str,
    raw_prices: pd.DataFrame,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "requested_tickers": requested_tickers,
        "kept_tickers": kept_tickers,
        "dropped_tickers": [t for t in requested_tickers if t in drop_reasons],
        "dropped_reasons": drop_reasons,
        "N_assets_final": len(kept_tickers),
        "min_assets": min_assets,
        "error_code": error_code,
    }
    (processed_dir / "data_manifest_failed.json").write_text(json.dumps(failed_manifest, indent=2))
    failed_quality = _build_failed_quality_summary(raw_prices, raw_metrics, kept_tickers, drop_reasons)
    failed_quality_path = processed_dir / "data_quality_failed.csv"
    failed_quality.to_csv(failed_quality_path, index=False)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    failed_quality.to_csv(reports_dir / "data_quality_failed.csv", index=False)


def _write_drop_report(raw_metrics: Dict[str, Dict], drop_reasons: Dict[str, str]) -> None:
    if not drop_reasons:
        return
    rows = []
    for ticker, reason in drop_reasons.items():
        metrics = raw_metrics.get(ticker, {})
        first_valid = metrics.get("first_valid_date")
        rows.append(
            {
                "ticker": ticker,
                "reason": reason,
                "first_valid_date": None if first_valid is None else pd.to_datetime(first_valid).date().isoformat(),
                "valid_days": metrics.get("valid_obs_count", 0),
                "raw_missing_fraction": metrics.get("missing_fraction", 1.0),
            }
        )
    drop_report = pd.DataFrame(rows)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    drop_report.to_csv(reports_dir / "universe_drop_report.csv", index=False)


def load_market_data(
    cfg: Dict,
    *,
    offline: bool,
    require_cache: bool,
    cache_only: bool,
    force_refresh: bool,
    debug_return_intermediates: bool = False,
) -> tuple:
    data_cfg = cfg.get("data", {})
    env_cfg = cfg.get("env", {})
    universe_cfg = cfg.get("universe", {})
    dates = cfg["dates"]
    start = dates["train_start"]
    end = dates["test_end"]
    source = data_cfg.get("source", "yfinance_only")
    if source != "yfinance_only":
        raise ValueError("Unsupported data source; only yfinance_only is permitted.")
    universe_policy = universe_cfg.get("policy", data_cfg.get("universe_policy", "availability_filtered"))
    if universe_policy not in {"availability_filtered", "quality_gate", "fixed_list"}:
        raise ValueError(f"Unsupported universe_policy={universe_policy}.")

    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    offline_cfg = data_cfg.get("offline", False)
    require_cache_flag = bool(require_cache or require_cache_cfg or paper_mode)
    offline_flag = bool(offline or offline_cfg)
    cache_only_flag = bool(cache_only or require_cache_flag or offline_flag)

    ticker_substitutions = data_cfg.get("ticker_substitutions") or {}
    if (paper_mode or require_cache_flag) and ticker_substitutions:
        raise ValueError("SUBSTITUTIONS_DISABLED_IN_PAPER_MODE")

    quality_params = {**DEFAULT_QUALITY_PARAMS, **(data_cfg.get("quality_params") or {})}
    max_missing_fraction = float(quality_params.get("max_missing_fraction", DEFAULT_QUALITY_PARAMS["max_missing_fraction"]))
    min_history_days = int(data_cfg.get("min_history_days", 500))
    min_assets = int(universe_cfg.get("min_assets", data_cfg.get("min_assets", 25)))
    history_tolerance_days = int(data_cfg.get("history_tolerance_days", 0))

    fixed_asset_list = universe_cfg.get("fixed_asset_list")
    if fixed_asset_list is not None:
        if not isinstance(fixed_asset_list, list) or not fixed_asset_list:
            raise ValueError("universe.fixed_asset_list must be a non-empty list when provided.")
        tickers = list(fixed_asset_list)
    else:
        tickers = list(data_cfg.get("tickers", DOW30_TICKERS))
    if len(tickers) != len(set(tickers)):
        raise ValueError("Ticker universe contains duplicates.")

    feature_flags = {
        "returns_window": True,
        "volatility": True,
        "prev_weights": True,
    }
    cost_params = {
        "transaction_cost": env_cfg.get("c_tc"),
    }
    L = env_cfg.get("L")
    Lv = env_cfg.get("Lv")

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    session_opts = data_cfg.get("session_opts", None)
    market_closed_info = {
        "calendar_freq": "B",
        "market_closed_days_removed": 0,
        "market_closed_fraction_removed": 0.0,
        "market_closed_rule": "drop_all_nan_rows_before_missing_fraction",
        "total_days_before_drop": 0,
        "total_days_after_drop": 0,
    }

    if cache_only_flag:
        prices, returns, manifest, quality_summary = _load_processed_cache(processed_dir)
        manifest = _ensure_manifest_env_fields(
            manifest,
            asset_list=list(returns.columns),
            L=L,
            Lv=Lv,
            feature_flags=feature_flags,
            cost_params=cost_params,
        )
        if debug_return_intermediates:
            debug_info = {
                "raw_aligned": pd.DataFrame(),
                "raw_clean": pd.DataFrame(),
                "market_closed_info": market_closed_info,
            }
            return prices, returns, manifest, quality_summary, pd.DataFrame(), pd.DataFrame(), {}, {}, market_closed_info, debug_info
        return prices, returns, manifest, quality_summary

    cache_exists = processed_dir.joinpath("prices.parquet").exists() and processed_dir.joinpath("returns.parquet").exists()
    if cache_exists and not force_refresh:
        LOGGER.info("Using existing processed cache (force_refresh=False).")
        prices, returns, manifest, quality_summary = _load_processed_cache(processed_dir)
        manifest = _ensure_manifest_env_fields(
            manifest,
            asset_list=list(returns.columns),
            L=L,
            Lv=Lv,
            feature_flags=feature_flags,
            cost_params=cost_params,
        )
        if debug_return_intermediates:
            debug_info = {
                "raw_aligned": pd.DataFrame(),
                "raw_clean": pd.DataFrame(),
                "market_closed_info": market_closed_info,
            }
            return prices, returns, manifest, quality_summary, pd.DataFrame(), pd.DataFrame(), {}, {}, market_closed_info, debug_info
        return prices, returns, manifest, quality_summary

    allow_subs = not (paper_mode or require_cache_flag)
    raw_metrics: Dict[str, Dict] = {}
    raw_missing_fraction: Dict[str, float] = {}
    kept: list[str] = []
    drop_reasons: Dict[str, str] = {}
    subs_used: Dict[str, str] = {}
    raw_prices_aligned = pd.DataFrame()
    raw_prices_clean = pd.DataFrame()

    try:
        series_map, drop_reasons, subs_used = _download_all_series(
            tickers,
            start,
            end,
            session_opts=session_opts,
            ticker_substitutions=ticker_substitutions,
            allow_subs=allow_subs,
        )
        raw_prices_aligned = _align_raw_prices(tickers, series_map, start, end)
        raw_prices_clean, market_closed_info = _drop_market_closed_days(raw_prices_aligned)
        raw_missing_fraction = raw_prices_clean.isna().mean().to_dict()
        raw_metrics = _compute_raw_metrics(raw_prices_clean, raw_missing_fraction=raw_missing_fraction)
        if universe_policy == "fixed_list":
            if drop_reasons:
                _write_drop_report(raw_metrics, drop_reasons)
                raise RuntimeError("FIXED_LIST_DOWNLOAD_FAILED")
            kept = tickers
        else:
            kept, _, drop_reasons = _select_universe(
                tickers,
                raw_metrics,
                drop_reasons,
                start,
                min_history_days,
                history_tolerance_days,
                max_missing_fraction,
            )
        _write_drop_report(raw_metrics, drop_reasons)

        if len(kept) < min_assets:
            raise RuntimeError("DATA_UNIVERSE_TOO_SMALL")

        cleaned_prices, kept_missing_fraction = _align_prices_raw_then_fill(
            raw_prices_clean[kept], ffill=True, bfill=True
        )
        returns = _compute_log_returns(cleaned_prices)
        try:
            data_quality_check(
                cleaned_prices,
                returns,
                min_days=min_history_days,
                min_vol_std=quality_params.get("min_vol_std", DEFAULT_QUALITY_PARAMS["min_vol_std"]),
                min_max_abs_return=quality_params.get(
                    "min_max_abs_return", DEFAULT_QUALITY_PARAMS["min_max_abs_return"]
                ),
                max_flat_fraction=quality_params.get("max_flat_fraction", DEFAULT_QUALITY_PARAMS["max_flat_fraction"]),
                missing_fraction=kept_missing_fraction,
                max_missing_fraction=max_missing_fraction,
            )
        except RuntimeError:
            for ticker in kept:
                drop_reasons.setdefault(ticker, "QUALITY_GATE_FAILED")
            _write_drop_report(raw_metrics, drop_reasons)
            raise

        quality_summary = _build_quality_summary(raw_prices_clean, cleaned_prices, returns, raw_metrics, kept, drop_reasons)
        quality_summary["market_closed_days_removed"] = market_closed_info["market_closed_days_removed"]
        quality_summary["market_closed_fraction_removed"] = market_closed_info["market_closed_fraction_removed"]
        quality_summary["total_days_before_drop"] = market_closed_info["total_days_before_drop"]
        quality_summary["total_days_after_drop"] = market_closed_info["total_days_after_drop"]

        manifest_kwargs = {
            "start": start,
            "end": end,
            "requested_tickers": tickers,
            "kept_tickers": kept,
            "dropped_reasons": drop_reasons,
            "quality_params": quality_params,
            "min_history_days": min_history_days,
            "min_assets": min_assets,
            "history_tolerance_days": history_tolerance_days,
            "universe_policy": universe_policy,
            "substitutions_used": subs_used,
            "market_closed_days_removed": market_closed_info["market_closed_days_removed"],
            "market_closed_fraction_removed": market_closed_info["market_closed_fraction_removed"],
            "total_days_before_drop": market_closed_info["total_days_before_drop"],
            "total_days_after_drop": market_closed_info["total_days_after_drop"],
            "L": L,
            "Lv": Lv,
            "feature_flags": feature_flags,
            "cost_params": cost_params,
            "schema_version": ENV_SCHEMA_VERSION,
        }
        manifest, quality_summary = _save_artifacts(
            processed_dir,
            cleaned_prices,
            returns,
            quality_summary,
            manifest_kwargs,
        )
        if debug_return_intermediates:
            filled_prices = raw_prices_clean[kept].ffill().bfill()
            raw_missing = {ticker: raw_missing_fraction.get(ticker, 1.0) for ticker in tickers}
            drop_decisions = {ticker: drop_reasons.get(ticker) for ticker in tickers}
            debug_info = {
                "raw_aligned": raw_prices_aligned,
                "raw_clean": raw_prices_clean,
                "market_closed_info": market_closed_info,
            }
            return (
                cleaned_prices,
                returns,
                manifest,
                quality_summary,
                raw_prices_clean,
                filled_prices,
                raw_missing,
                drop_decisions,
                market_closed_info,
                debug_info,
            )
        return cleaned_prices, returns, manifest, quality_summary
    except Exception as exc:
        _write_drop_report(raw_metrics, drop_reasons)
        _write_failed_artifacts(
            processed_dir,
            start=start,
            end=end,
            requested_tickers=tickers,
            kept_tickers=kept,
            drop_reasons=drop_reasons,
            raw_metrics=raw_metrics,
            min_assets=min_assets,
            error_code=str(exc),
            raw_prices=raw_prices_clean,
        )
        raise


def build_cache_snapshot(cfg: Dict, *, force_refresh: bool = True) -> Dict:
    data_cfg = {**cfg.get("data", {})}
    cfg_for_build = {
        **cfg,
        "data": {**data_cfg, "offline": False, "require_cache": False, "paper_mode": False},
    }
    _, _, manifest, _ = load_market_data(
        cfg_for_build,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=force_refresh,
    )
    return manifest


def slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[start:end]


def split_returns(
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = slice_frame(returns, train_start, train_end)
    test = slice_frame(returns, test_start, test_end)
    return train, test
