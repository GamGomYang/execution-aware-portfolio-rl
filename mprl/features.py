"""State assembly for the Dow30 + sector + macro feature vector."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .utils import cvar, rolling_rsi, safe_corrcoef


class MarketFeatureAssembler:
    """Construct the 259-dim state vector and risk encoder inputs with NaN guards."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self._macro_history: Dict[str, deque[float]] = {
            name: deque(maxlen=512) for name in self.cfg.macro_features
        }
        self._portfolio_history: Dict[str, deque[float]] = {
            name: deque(maxlen=512) for name in self.cfg.portfolio_features if name != "weights"
        }
        self._last_weights: np.ndarray | None = None
        self._last_signal: float = 0.0
        self._portfolio_defaults: Dict[str, float] = {
            "rolling_vol": 0.01,
            "drawdown": 0.0,
            "cash_ratio": 0.0,
            "turnover": 0.0,
            "win_rate": 0.5,
            "mdd_30d": 0.0,
            "exposure_var": 0.0,
            "hedge_ratio": 0.0,
            "beta_to_spx": 1.0,
            "beta_to_vix": -0.2,
            "port_skew": 0.0,
            "port_kurt": 3.0,
            "leverage": 1.0,
            "liquidity_score": 0.8,
        }

    @staticmethod
    def _assert_not_all_nan(name: str, array: np.ndarray) -> None:
        arr = np.asarray(array, dtype=np.float32)
        if arr.size == 0 or not np.isfinite(arr).any():
            raise ValueError(f"{name} contains no finite observations.")

    def sanitize_history(self, history: np.ndarray, label: str) -> np.ndarray:
        """Forward-fill + median fill for historical matrices."""
        arr = np.asarray(history, dtype=np.float32)
        self._assert_not_all_nan(label, arr)
        if arr.ndim != 2:
            raise ValueError(f"{label} must be 2D, got shape {arr.shape}")
        if not np.isnan(arr).any():
            return arr
        df = pd.DataFrame(arr)
        df = df.ffill()
        if df.isna().values.any():
            medians = df.median(skipna=True)
            df = df.fillna(medians)
        if df.isna().values.any():
            raise ValueError(f"{label} still has NaN entries after sanitization")
        return df.to_numpy(dtype=np.float32)

    def _sanitize_scalar_history(
        self,
        history: deque[float],
        name: str,
        value: float | np.ndarray,
        default: float = 0.0,
    ) -> float:
        val = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
        if np.isfinite(val):
            history.append(val)
            return val
        if not history:
            history.append(default)
            return default
        median_val = float(np.median(history))
        history.append(median_val)
        return median_val

    def _sanitize_macro_state(self, macro_state: Dict[str, float]) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key in self.cfg.macro_features:
            history = self._macro_history[key]
            value = float(macro_state.get(key, np.nan))
            if np.isfinite(value):
                history.append(value)
                cleaned[key] = value
                continue
            if not history:
                raise ValueError(f"Macro feature {key} has no valid observations to use as fallback.")
            fallback = float(np.median(history))
            history.append(fallback)
            cleaned[key] = fallback
        return cleaned

    def _sanitize_portfolio_ctx(self, ctx: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | float]:
        sanitized: Dict[str, np.ndarray | float] = {}
        weights = np.asarray(ctx.get("weights"), dtype=np.float32)
        self._assert_not_all_nan("portfolio_weights", weights)
        mask = ~np.isfinite(weights)
        if mask.any():
            weights = weights.copy()
            if self._last_weights is not None:
                weights[mask] = self._last_weights[mask]
                mask = ~np.isfinite(weights)
            if mask.any():
                valid = weights[~mask]
                if valid.size > 0:
                    median_val = float(np.median(valid))
                    weights[mask] = median_val
            if (~np.isfinite(weights)).any():
                raise ValueError("Portfolio weights still contain invalid entries after fill.")
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= total
        self._last_weights = weights.copy()
        sanitized["weights"] = weights
        for key in self.cfg.portfolio_features:
            if key == "weights":
                continue
            history = self._portfolio_history[key]
            default = float(self._portfolio_defaults.get(key, 0.0))
            sanitized[key] = self._sanitize_scalar_history(history, key, ctx.get(key, default), default)
        return sanitized

    def build_state(
        self,
        asset_hist: np.ndarray,
        sector_hist: np.ndarray,
        macro_state: Dict[str, float],
        portfolio_ctx: Dict[str, np.ndarray],
        *,
        precleaned: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if precleaned:
            asset_hist_clean = np.asarray(asset_hist, dtype=np.float32)
            sector_hist_clean = np.asarray(sector_hist, dtype=np.float32)
            self._assert_not_all_nan("asset_history", asset_hist_clean)
            self._assert_not_all_nan("sector_history", sector_hist_clean)
        else:
            asset_hist_clean = self.sanitize_history(asset_hist, "asset_history")
            sector_hist_clean = self.sanitize_history(sector_hist, "sector_history")

        macro_clean = self._sanitize_macro_state(macro_state)
        portfolio_clean = self._sanitize_portfolio_ctx(portfolio_ctx)

        features: List[float] = []

        for col, symbol in enumerate(self.cfg.dow_symbols):
            series = asset_hist_clean[:, col]
            features.extend(
                [
                    series[-1],
                    series[-5:].std(),
                    series.std(),
                    series[-5:].sum(),
                    series.sum(),
                    rolling_rsi(series),
                ]
            )
            features.extend(self.cfg.sector_one_hot(symbol))

        for col in range(len(self.cfg.sector_etfs)):
            series = sector_hist_clean[:, col]
            features.extend(
                [
                    series[-1],
                    series[-5:].std(),
                    series.std(),
                    series.sum(),
                ]
            )

        features.extend([macro_clean[key] for key in self.cfg.macro_features])
        sector_group = self.cfg.assets_by_sector
        for sector in self.cfg.sector_names:
            idxs = sector_group.get(sector, [])
            if not idxs:
                features.extend([0.0, 0.0, 0.0])
                continue
            sub = asset_hist_clean[:, idxs]
            latest = float(sub[-1].mean())
            vol5 = float(sub[-5:].std())
            momentum = float(sub[-5:].sum())
            features.extend([latest, vol5, momentum])

        weights = portfolio_clean["weights"]
        features.extend(weights.tolist())
        port_stats = [
            float(portfolio_clean["rolling_vol"]),
            float(portfolio_clean["drawdown"]),
            float(portfolio_clean["cash_ratio"]),
            float(portfolio_clean["turnover"]),
            float(portfolio_clean["win_rate"]),
            float(portfolio_clean["mdd_30d"]),
            float(portfolio_clean["exposure_var"]),
            float(portfolio_clean["hedge_ratio"]),
            float(portfolio_clean["beta_to_spx"]),
            float(portfolio_clean["beta_to_vix"]),
            float(portfolio_clean["port_skew"]),
            float(portfolio_clean["port_kurt"]),
            float(portfolio_clean["leverage"]),
            float(portfolio_clean["liquidity_score"]),
        ]
        features.extend(port_stats)

        state_vec = np.asarray(features, dtype=np.float32)

        corr = safe_corrcoef(asset_hist_clean)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals.sort()
        mean_corr = float(np.mean(corr))
        max_corr = float(np.max(corr))
        vix = macro_clean["vix"]
        corr_z = (mean_corr - 0.2) / 0.1
        dd = float(portfolio_clean["drawdown"])
        dd_z = dd / 0.1
        vix_z = (vix - 15.0) / 5.0

        risk_feats = np.array(
            [
                asset_hist_clean[:, :].std(),
                asset_hist_clean[-5:, :].std(),
                asset_hist_clean[-1, :].std(),
                cvar(asset_hist_clean.flatten()),
                vix_z,
                macro_clean["move"],
                macro_clean["vvix"],
                macro_clean["ted_spread"],
                macro_clean["hy_ig_spread"],
                corr_z,
                max_corr,
                float(eigvals[-1]),
                float(eigvals[-2]),
                dd_z,
                np.abs(asset_hist_clean[-1]).mean(),
                sector_hist_clean[-1].std(),
            ],
            dtype=np.float32,
        )

        stress_score = 0.5 * vix_z + 0.3 * dd_z + 0.2 * corr_z
        signal_val = float(np.clip(stress_score, 0.0, 1.0))
        if not np.isfinite(signal_val):
            signal_val = self._last_signal
        else:
            self._last_signal = signal_val
        return state_vec, risk_feats, signal_val
