"""State assembly for the Dow30 + sector + macro feature vector."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import FeatureConfig
from .utils import cvar, rolling_rsi


class MarketFeatureAssembler:
    """Construct the 259-dim state vector and risk encoder inputs."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def build_state(
        self,
        asset_hist: np.ndarray,
        sector_hist: np.ndarray,
        macro_state: Dict[str, float],
        portfolio_ctx: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        features: List[float] = []

        for col in range(self.cfg.num_assets):
            series = asset_hist[:, col]
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

        for col in range(len(self.cfg.sector_etfs)):
            series = sector_hist[:, col]
            features.extend(
                [
                    series[-1],
                    series[-5:].std(),
                    series.std(),
                    series.sum(),
                ]
            )

        features.extend([macro_state[key] for key in self.cfg.macro_features])

        weights = portfolio_ctx["weights"]
        features.extend(weights.tolist())
        port_stats = [
            portfolio_ctx["rolling_vol"],
            portfolio_ctx["drawdown"],
            portfolio_ctx["cash_ratio"],
            portfolio_ctx["turnover"],
            portfolio_ctx["win_rate"],
            portfolio_ctx["mdd_30d"],
            portfolio_ctx["exposure_var"],
            portfolio_ctx["hedge_ratio"],
            portfolio_ctx["beta_to_spx"],
            portfolio_ctx["beta_to_vix"],
            portfolio_ctx["port_skew"],
            portfolio_ctx["port_kurt"],
            portfolio_ctx["leverage"],
            portfolio_ctx["liquidity_score"],
        ]
        features.extend(port_stats)

        state_vec = np.asarray(features, dtype=np.float32)

        corr = np.corrcoef(asset_hist.T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals.sort()
        mean_corr = float(np.mean(corr))
        max_corr = float(np.max(corr))
        risk_feats = np.array(
            [
                asset_hist[:, :].std(),
                asset_hist[-5:, :].std(),
                asset_hist[-1, :].std(),
                cvar(asset_hist.flatten()),
                macro_state["vix"],
                macro_state["move"],
                macro_state["vvix"],
                macro_state["ted_spread"],
                macro_state["hy_ig_spread"],
                mean_corr,
                max_corr,
                float(eigvals[-1]),
                float(eigvals[-2]),
                portfolio_ctx["drawdown"],
                np.abs(asset_hist[-1]).mean(),
                sector_hist[-1].std(),
            ],
            dtype=np.float32,
        )

        stress_score = (
            0.5 * (macro_state["vix"] / 80.0)
            + 0.3 * max(0.0, portfolio_ctx["drawdown"])
            + 0.2 * max(0.0, mean_corr)
        )
        signal = float(np.clip(stress_score, 0.0, 1.0))
        return state_vec, risk_feats, signal
