"""Mock environment emulating Dow30 portfolio dynamics."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from .config import FeatureConfig, TrainingConfig
from .features import MarketFeatureAssembler


class PortfolioEnvStub:
    """Stochastic generator to test the MPRL loop without market data."""

    def __init__(self, feature_cfg: FeatureConfig, train_cfg: TrainingConfig):
        self.cfg = feature_cfg
        self.train_cfg = train_cfg
        self.history = feature_cfg.lookback
        self.asset_hist = np.random.normal(0.0005, 0.01, size=(self.history, self.cfg.num_assets))
        self.sector_hist = np.random.normal(0.0004, 0.008, size=(self.history, len(self.cfg.sector_etfs)))
        self.macro_state = self._init_macro_state()
        self.assembler = MarketFeatureAssembler(feature_cfg)
        self.portfolio_ctx = self._init_portfolio_ctx()
        self.prev_weights = np.ones(self.cfg.num_assets, dtype=np.float32) / self.cfg.num_assets
        self.portfolio_returns: List[float] = []
        self.step_counter = 0
        self.cooldown_timer = 0
        self._last_state_signal = 0.0

    def _init_macro_state(self) -> Dict[str, float]:
        return {
            "vix": 18.0,
            "move": 90.0,
            "vvix": 100.0,
            "ted_spread": 0.3,
            "hy_ig_spread": 2.5,
            "spx_return": 0.0004,
            "nasdaq_return": 0.0005,
            "russell_return": 0.0003,
            "mean_corr": 0.2,
            "max_corr": 0.8,
            "eig1": 12.0,
            "eig2": 9.0,
        }

    def _init_portfolio_ctx(self) -> Dict[str, np.ndarray]:
        return {
            "weights": np.ones(self.cfg.num_assets, dtype=np.float32) / self.cfg.num_assets,
            "rolling_vol": 0.01,
            "drawdown": 0.0,
            "cash_ratio": 0.02,
            "turnover": 0.0,
            "win_rate": 0.5,
            "mdd_30d": 0.02,
            "exposure_var": 0.003,
            "hedge_ratio": 0.1,
            "beta_to_spx": 1.0,
            "beta_to_vix": -0.2,
            "port_skew": 0.0,
            "port_kurt": 3.0,
            "leverage": 1.0,
            "liquidity_score": 0.8,
        }

    def reset(self) -> Tuple[np.ndarray, np.ndarray, float]:
        self.asset_hist = np.random.normal(0.0005, 0.01, size=(self.history, self.cfg.num_assets))
        self.sector_hist = np.random.normal(0.0004, 0.008, size=(self.history, len(self.cfg.sector_etfs)))
        self.macro_state = self._init_macro_state()
        self.portfolio_ctx = self._init_portfolio_ctx()
        self.portfolio_returns.clear()
        self.prev_weights = self.portfolio_ctx["weights"].copy()
        self.step_counter = 0
        self.cooldown_timer = 0
        self._last_state_signal = 0.0
        self.asset_sector_ids = np.array(
            [
                feature_cfg.sector_names.index(feature_cfg.dow_sector_map.get(symbol, feature_cfg.sector_names[0]))
                for symbol in feature_cfg.dow_symbols
            ]
        )
        self.sector_masks = [
            self.asset_sector_ids == idx for idx in range(len(feature_cfg.sector_names))
        ]
        return self._observe()

    def _observe(self) -> Tuple[np.ndarray, np.ndarray, float]:
        state_vec, risk_feats, signal = self.assembler.build_state(
            self.asset_hist, self.sector_hist, self.macro_state, self.portfolio_ctx
        )
        if not np.isfinite(signal):
            signal = self._last_state_signal
        else:
            self._last_state_signal = signal
        return state_vec, risk_feats, signal

    def _sector_weights(self, weights: np.ndarray) -> np.ndarray:
        accum = np.zeros(len(self.sector_masks), dtype=np.float64)
        np.add.at(accum, self.asset_sector_ids, weights)
        return accum

    def _apply_sector_constraints(self, candidate: np.ndarray, prev: np.ndarray) -> np.ndarray:
        prev_sector = self._sector_weights(prev)
        cand_sector = self._sector_weights(candidate)
        diff_sector = cand_sector - prev_sector
        adjusted = candidate.copy()
        for idx, mask in enumerate(self.sector_masks):
            sector_diff = diff_sector[idx]
            if abs(sector_diff) < self.train_cfg.sector_no_trade_threshold:
                adjusted[mask] = prev[mask]
                continue
            max_change = self.train_cfg.sector_max_weight_change
            if max_change < 1.0 and abs(sector_diff) > max_change:
                scale = max_change / abs(sector_diff)
                adjusted[mask] = prev[mask] + (adjusted[mask] - prev[mask]) * scale
        total_sector_change = np.abs(adjusted - prev).sum()
        if (
            self.train_cfg.sector_max_total_weight_change < 1.0
            and total_sector_change > self.train_cfg.sector_max_total_weight_change
        ):
            scale = self.train_cfg.sector_max_total_weight_change / max(total_sector_change, 1e-8)
            adjusted = prev + (adjusted - prev) * scale
        adjusted = np.clip(adjusted, 0.0, None)
        norm = adjusted.sum()
        if norm <= 0.0:
            adjusted = np.ones_like(adjusted) / len(adjusted)
        else:
            adjusted /= norm
        return adjusted

    def _apply_action_constraints(self, action: np.ndarray) -> np.ndarray:
        prev = self.prev_weights
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return prev.copy()
        diff = action - prev
        if self.train_cfg.no_trade_threshold > 0.0:
            diff = np.where(np.abs(diff) < self.train_cfg.no_trade_threshold, 0.0, diff)
        if self.train_cfg.max_weight_change < 1.0:
            diff = np.clip(diff, -self.train_cfg.max_weight_change, self.train_cfg.max_weight_change)
        total_change = np.abs(diff).sum()
        if (
            self.train_cfg.max_total_weight_change < 1.0
            and total_change > self.train_cfg.max_total_weight_change
        ):
            scale = self.train_cfg.max_total_weight_change / max(total_change, 1e-8)
            diff *= scale
        constrained = prev + diff
        constrained = np.clip(constrained, 0.0, None)
        total = constrained.sum()
        if total <= 0.0:
            constrained = np.ones_like(constrained) / len(constrained)
        else:
            constrained /= total
        constrained = self._apply_sector_constraints(constrained, prev)
        return constrained

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, float, Dict]:
        action = self._apply_action_constraints(action)
        constrained_diff = action - self.prev_weights
        market_vol = np.random.gamma(shape=1.5, scale=0.01)
        systematic = np.random.normal(0.0, market_vol)
        asset_returns = np.random.normal(0.0004 + systematic, market_vol, size=self.cfg.num_assets)
        sector_returns = np.random.normal(0.0003 + systematic, market_vol * 0.8, size=len(self.cfg.sector_etfs))

        self.asset_hist = np.roll(self.asset_hist, -1, axis=0)
        self.asset_hist[-1] = asset_returns
        self.sector_hist = np.roll(self.sector_hist, -1, axis=0)
        self.sector_hist[-1] = sector_returns

        self.macro_state["vix"] = max(10.0, self.macro_state["vix"] * 0.97 + abs(systematic) * 100)
        self.macro_state["move"] = max(60.0, self.macro_state["move"] * 0.95 + market_vol * 4000)
        self.macro_state["vvix"] = max(70.0, self.macro_state["vvix"] * 0.95 + market_vol * 3000)
        self.macro_state["spx_return"] = systematic
        self.macro_state["nasdaq_return"] = systematic * 1.1
        self.macro_state["russell_return"] = systematic * 1.3
        corr = np.corrcoef(self.asset_hist.T)
        self.macro_state["mean_corr"] = float(np.mean(corr))
        self.macro_state["max_corr"] = float(np.max(corr))
        eigvals = np.linalg.eigvalsh(corr)
        eigvals.sort()
        self.macro_state["eig1"] = float(eigvals[-1])
        self.macro_state["eig2"] = float(eigvals[-2])

        portfolio_return = float(np.dot(action, asset_returns))
        risk_metric = float(np.abs(asset_returns).mean())
        l1_change = float(np.abs(constrained_diff).sum())
        l2_change = float(np.square(constrained_diff).sum())
        tc = float(l1_change * self.train_cfg.transaction_cost + l2_change * self.train_cfg.tc_penalty)
        reward = portfolio_return - tc - 0.5 * risk_metric

        self.prev_weights = action.copy()
        if l1_change > 0.0:
            self.cooldown_timer = self.train_cfg.cooldown_steps
        self.portfolio_returns.append(portfolio_return)
        if len(self.portfolio_returns) > 30:
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
                "mdd_30d": float(
                    np.min(
                        (equity_curve - np.maximum.accumulate(equity_curve))
                        / np.maximum.accumulate(equity_curve)
                    )
                ),
                "exposure_var": float(np.var(action)),
                "hedge_ratio": float(action[:5].sum()),
                "beta_to_spx": float(np.cov(action, asset_returns)[0, 1]) if len(action) > 1 else 1.0,
                "beta_to_vix": float(-market_vol),
                "port_skew": float(
                    np.mean((np.array(self.portfolio_returns) - np.mean(self.portfolio_returns)) ** 3)
                )
                if len(self.portfolio_returns) > 2
                else 0.0,
                "port_kurt": float(
                    np.mean((np.array(self.portfolio_returns) - np.mean(self.portfolio_returns)) ** 4)
                )
                if len(self.portfolio_returns) > 3
                else 3.0,
                "leverage": float(1.0 + 0.1 * random.random()),
                "liquidity_score": float(0.8 + 0.1 * random.random()),
            }
        )

        state, risk_features, signal = self._observe()
        self.step_counter += 1
        info = {
            "risk_features": risk_features,
            "signal": signal,
            "date": f"sim_{self.step_counter}",
            "done": False,
            "executed_action": action,
        }
        return state, reward, risk_metric, signal, info
