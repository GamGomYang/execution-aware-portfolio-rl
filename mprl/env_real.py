"""Environment that replays historical data fetched via yfinance."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import FeatureConfig, TrainingConfig
from .data_pipeline import MarketDataset
from .features import MarketFeatureAssembler
from .utils import safe_corrcoef


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
        self.cooldown_timer = 0
        self._last_state_signal = 0.0
        self._last_env_stress = 0.0
        self.asset_sector_ids = np.array(
            [
                feature_cfg.sector_names.index(feature_cfg.dow_sector_map.get(symbol, feature_cfg.sector_names[0]))
                for symbol in feature_cfg.dow_symbols
            ]
        )
        self.sector_masks = [
            self.asset_sector_ids == idx for idx in range(len(feature_cfg.sector_names))
        ]

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
        self.cooldown_timer = 0
        self._last_state_signal = 0.0
        self._last_env_stress = 0.0
        return self._observe()

    def _build_macro_state(self, asset_hist: np.ndarray) -> Dict[str, float]:
        macro = self.dataset.macro_state(self.index - 1)
        corr = safe_corrcoef(asset_hist)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals.sort()
        macro["mean_corr"] = float(np.mean(corr))
        macro["max_corr"] = float(np.max(corr))
        macro["eig1"] = float(eigvals[-1])
        macro["eig2"] = float(eigvals[-2])
        return macro

    def _observe(self) -> Tuple[np.ndarray, np.ndarray, float]:
        start = self.index - self.cfg.lookback
        asset_hist_raw = self.dataset.asset_returns[start:self.index]
        sector_hist_raw = self.dataset.sector_returns[start:self.index]
        asset_hist = self.assembler.sanitize_history(asset_hist_raw, "asset_history")
        sector_hist = self.assembler.sanitize_history(sector_hist_raw, "sector_history")
        macro_state = self._build_macro_state(asset_hist)
        state_vec, risk_feats, signal = self.assembler.build_state(
            asset_hist, sector_hist, macro_state, self.portfolio_ctx, precleaned=True
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
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return self.prev_weights.copy()
        prev = self.prev_weights
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
        constrained_action = self._apply_action_constraints(action)
        constrained_diff = constrained_action - self.prev_weights
        returns = self.dataset.asset_returns[self.index]
        date = str(self.dataset.dates[self.index])
        portfolio_return = float(np.dot(constrained_action, returns))
        risk_metric = float(np.abs(returns).mean())
        l1_change = float(np.abs(constrained_diff).sum())
        l2_change = float(np.square(constrained_diff).sum())
        tc = float(l1_change * self.train_cfg.transaction_cost + l2_change * self.train_cfg.tc_penalty)

        self.prev_weights = constrained_action.copy()
        if l1_change > 0.0:
            self.cooldown_timer = self.train_cfg.cooldown_steps
        self.portfolio_returns.append(portfolio_return)
        if len(self.portfolio_returns) > 60:
            self.portfolio_returns.pop(0)

        equity_curve = np.cumprod([1.0] + [r + 1.0 for r in self.portfolio_returns])
        peak = np.max(equity_curve)
        current = equity_curve[-1]
        drawdown = (current - peak) / max(peak, 1e-6)

        stress_inputs = np.array([risk_metric, drawdown], dtype=np.float32)
        stress_candidate = float(
            np.clip(0.5 * (risk_metric / 0.05) + 0.5 * np.clip(-drawdown, 0.0, 1.0), 0.0, 2.0)
        )
        if not np.isfinite(stress_candidate) or not np.isfinite(stress_inputs).all():
            stress_signal_now = self._last_env_stress
        else:
            stress_signal_now = stress_candidate
            self._last_env_stress = stress_signal_now
        risk_coeff = (
            self.train_cfg.risk_penalty_base
            + (self.train_cfg.risk_penalty_beta * stress_signal_now * 0.1)
        )
        turnover_penalty = (self.train_cfg.turnover_penalty_gamma * 0.25) * max(
            0.0, l1_change - self.train_cfg.turnover_penalty_threshold
        )
        tc = tc / 3.0
        reward = (
            portfolio_return * 100.0
            - tc
            - risk_coeff * risk_metric
            - turnover_penalty
        )

        self.portfolio_ctx.update(
            {
                "weights": constrained_action,
                "rolling_vol": float(np.std(self.portfolio_returns[-10:]) if self.portfolio_returns else 0.01),
                "drawdown": float(drawdown),
                "cash_ratio": float(max(0.0, 1.0 - constrained_action.sum())),
                "turnover": float(l1_change),
                "win_rate": float(np.mean(np.array(self.portfolio_returns) > 0)) if self.portfolio_returns else 0.5,
                "mdd_30d": float(np.min((equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve))),
                "exposure_var": float(np.var(action)),
                "hedge_ratio": float(constrained_action[:5].sum()),
                "beta_to_spx": float(np.cov(constrained_action, returns)[0, 1]) if len(constrained_action) > 1 else 1.0,
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
            "stress_now": stress_signal_now,
            "date": date,
            "done": done,
            "executed_action": constrained_action,
            "turnover": l1_change,
            "portfolio_return": portfolio_return,
        }
        return next_state, reward, risk_metric, signal, info
