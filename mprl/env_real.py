"""Environment that replays historical data fetched via yfinance."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import FeatureConfig, TrainingConfig
from .data_pipeline import MarketDataset
from .features import MarketFeatureAssembler


class YFinancePortfolioEnv:
    def __init__(self, feature_cfg: FeatureConfig, train_cfg: TrainingConfig, dataset: MarketDataset):
        self.cfg = feature_cfg
        self.train_cfg = train_cfg
        self.dataset = dataset
        self.assembler = MarketFeatureAssembler(feature_cfg)
        self.portfolio_ctx = self._init_portfolio_ctx()
        self.portfolio_returns: List[float] = []
        self.prev_weights = np.ones(self.cfg.num_assets, dtype=np.float32) / self.cfg.num_assets
        self.index = self.cfg.lookback

    def _init_portfolio_ctx(self) -> Dict[str, float]:
        return {
            "weights": np.ones(self.cfg.num_assets, dtype=np.float32) / self.cfg.num_assets,
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

    def reset(self) -> Tuple[np.ndarray, np.ndarray, float]:
        self.index = self.cfg.lookback
        self.portfolio_ctx = self._init_portfolio_ctx()
        self.portfolio_returns.clear()
        self.prev_weights = self.portfolio_ctx["weights"].copy()
        return self._observe()

    def _build_macro_state(self, asset_hist: np.ndarray) -> Dict[str, float]:
        macro = self.dataset.macro_state(self.index - 1)
        corr = np.corrcoef(asset_hist.T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals.sort()
        macro["mean_corr"] = float(np.mean(corr))
        macro["max_corr"] = float(np.max(corr))
        macro["eig1"] = float(eigvals[-1])
        macro["eig2"] = float(eigvals[-2])
        return macro

    def _observe(self) -> Tuple[np.ndarray, np.ndarray, float]:
        start = self.index - self.cfg.lookback
        asset_hist = self.dataset.asset_returns[start:self.index]
        sector_hist = self.dataset.sector_returns[start:self.index]
        macro_state = self._build_macro_state(asset_hist)
        return self.assembler.build_state(asset_hist, sector_hist, macro_state, self.portfolio_ctx)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, float, Dict]:
        returns = self.dataset.asset_returns[self.index]
        date = str(self.dataset.dates[self.index])
        portfolio_return = float(np.dot(action, returns))
        risk_metric = float(np.abs(returns).mean())
        tc = float(np.abs(action - self.prev_weights).sum() * self.train_cfg.transaction_cost)
        reward = portfolio_return - self.train_cfg.tc_penalty * tc - 0.5 * risk_metric

        self.prev_weights = action.copy()
        self.portfolio_returns.append(portfolio_return)
        if len(self.portfolio_returns) > 60:
            self.portfolio_returns.pop(0)

        equity_curve = np.cumprod([1.0] + [r + 1.0 for r in self.portfolio_returns])
        peak = np.max(equity_curve)
        current = equity_curve[-1]
        drawdown = (current - peak) / max(peak, 1e-6)

        self.portfolio_ctx.update(
            {
                "weights": action,
                "rolling_vol": float(np.std(self.portfolio_returns[-10:]) if self.portfolio_returns else 0.01),
                "drawdown": float(drawdown),
                "cash_ratio": float(max(0.0, 1.0 - action.sum())),
                "turnover": float(tc),
                "win_rate": float(np.mean(np.array(self.portfolio_returns) > 0)) if self.portfolio_returns else 0.5,
                "mdd_30d": float(np.min((equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve))),
                "exposure_var": float(np.var(action)),
                "hedge_ratio": float(action[:5].sum()),
                "beta_to_spx": float(np.cov(action, returns)[0, 1]) if len(action) > 1 else 1.0,
                "beta_to_vix": float(-risk_metric),
                "port_skew": float(np.mean((np.array(self.portfolio_returns) - np.mean(self.portfolio_returns)) ** 3))
                if len(self.portfolio_returns) > 2
                else 0.0,
                "port_kurt": float(np.mean((np.array(self.portfolio_returns) - np.mean(self.portfolio_returns)) ** 4))
                if len(self.portfolio_returns) > 3
                else 3.0,
                "leverage": float(1.0 + 0.1 * len(self.portfolio_returns) / 60.0),
                "liquidity_score": float(0.75 + 0.1 * np.exp(-abs(drawdown))),
            }
        )

        self.index += 1
        done = self.index >= self.dataset.length
        if done:
            next_state = np.zeros(self.cfg.total_state_dim, dtype=np.float32)
            risk_features = np.zeros(self.cfg.risk_feature_dim, dtype=np.float32)
            signal = 0.0
        else:
            next_state, risk_features, signal = self._observe()

        info = {
            "risk_features": risk_features,
            "signal": signal,
            "date": date,
            "done": done,
        }
        return next_state, reward, risk_metric, signal, info
