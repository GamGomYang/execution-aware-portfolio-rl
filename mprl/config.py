"""Configuration dataclasses for feature engineering and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class FeatureConfig:
    dow_symbols: List[str] = field(
        default_factory=lambda: [
            "AAPL",
            "AMGN",
            "AXP",
            "BA",
            "CAT",
            "CRM",
            "CSCO",
            "CVX",
            "DIS",
            "DOW",
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
            "WMT",
        ]
    )
    dow_sector_map: Dict[str, str] = field(
        default_factory=lambda: {
            "AAPL": "Information Technology",
            "AMGN": "Health Care",
            "AXP": "Financials",
            "BA": "Industrials",
            "CAT": "Industrials",
            "CRM": "Information Technology",
            "CSCO": "Information Technology",
            "CVX": "Energy",
            "DIS": "Communication Services",
            "DOW": "Materials",
            "GS": "Financials",
            "HD": "Consumer Discretionary",
            "HON": "Industrials",
            "IBM": "Information Technology",
            "INTC": "Information Technology",
            "JNJ": "Health Care",
            "JPM": "Financials",
            "KO": "Consumer Staples",
            "MCD": "Consumer Discretionary",
            "MMM": "Industrials",
            "MRK": "Health Care",
            "MSFT": "Information Technology",
            "NKE": "Consumer Discretionary",
            "PG": "Consumer Staples",
            "TRV": "Financials",
            "UNH": "Health Care",
            "V": "Information Technology",
            "VZ": "Communication Services",
            "WMT": "Consumer Staples",
        }
    )
    sector_etfs: List[str] = field(
        default_factory=lambda: [
            "XLF",
            "XLK",
            "XLI",
            "XLE",
            "XLU",
            "XLV",
            "XLP",
            "XLY",
            "XLB",
            "XLRE",
            "XLC",
        ]
    )
    micro_features: List[str] = field(
        default_factory=lambda: [
            "ret_1d",
            "vol_5d",
            "vol_20d",
            "mom_5d",
            "mom_20d",
            "rsi_14d",
        ]
    )
    sector_features: List[str] = field(
        default_factory=lambda: ["ret_1d", "vol_5d", "vol_20d", "mom_20d"]
    )
    macro_features: List[str] = field(
        default_factory=lambda: [
            "vix",
            "move",
            "vvix",
            "ted_spread",
            "hy_ig_spread",
            "spx_return",
            "nasdaq_return",
            "russell_return",
            "mean_corr",
            "max_corr",
            "eig1",
            "eig2",
        ]
    )
    portfolio_features: List[str] = field(
        default_factory=lambda: [
            "weights",
            "rolling_vol",
            "drawdown",
            "cash_ratio",
            "turnover",
            "win_rate",
            "mdd_30d",
            "exposure_var",
            "hedge_ratio",
            "beta_to_spx",
            "beta_to_vix",
            "port_skew",
            "port_kurt",
            "leverage",
            "liquidity_score",
        ]
    )
    lookback: int = 20

    @property
    def sector_names(self) -> List[str]:
        return sorted(set(self.dow_sector_map.get(sym, "Unknown") for sym in self.dow_symbols))

    @property
    def sector_encoding_dim(self) -> int:
        return len(self.sector_names)

    def sector_one_hot(self, symbol: str) -> List[float]:
        encoding = [0.0] * self.sector_encoding_dim
        sector = self.dow_sector_map.get(symbol)
        if sector is None:
            return encoding
        idx = self.sector_names.index(sector)
        encoding[idx] = 1.0
        return encoding

    @property
    def assets_by_sector(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {name: [] for name in self.sector_names}
        for idx, symbol in enumerate(self.dow_symbols):
            sector = self.dow_sector_map.get(symbol, self.sector_names[0])
            mapping.setdefault(sector, []).append(idx)
        return mapping

    @property
    def num_assets(self) -> int:
        return len(self.dow_symbols)

    @property
    def micro_dim(self) -> int:
        return self.num_assets * (len(self.micro_features) + self.sector_encoding_dim)

    @property
    def sector_dim(self) -> int:
        return len(self.sector_etfs) * len(self.sector_features)

    @property
    def macro_dim(self) -> int:
        return len(self.macro_features)

    @property
    def portfolio_dim(self) -> int:
        # weights contribute num_assets dimensions; the remaining stats are scalar features
        return self.num_assets + max(len(self.portfolio_features) - 1, 0)

    @property
    def sector_summary_dim(self) -> int:
        # latest return, 5d volatility, 5d momentum per sector
        return self.sector_encoding_dim * 3

    @property
    def total_state_dim(self) -> int:
        return (
            self.micro_dim
            + self.sector_dim
            + self.macro_dim
            + self.portfolio_dim
            + self.sector_summary_dim
        )

    @property
    def risk_feature_dim(self) -> int:
        return 16

    @property
    def hebbian_dim(self) -> int:
        return 32

    @property
    def risk_feature_names(self) -> List[str]:
        return [
            "asset_vol_total",
            "asset_vol_5d",
            "asset_vol_last",
            "cvar_all",
            "vix",
            "move",
            "vvix",
            "ted_spread",
            "hy_ig_spread",
            "mean_corr",
            "max_corr",
            "eig1",
            "eig2",
            "portfolio_drawdown",
            "mean_abs_ret",
            "sector_vol_last",
        ]


@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gamma: float = 0.99
    alpha: float = 0.08
    k_alpha: float = 0.45
    k_q: float = 0.3
    risk_lambda: float = 0.9
    num_strategies: int = 4
    epsilon_base: float = 0.08
    epsilon_beta: float = 0.6
    epsilon_min: float = 0.02
    epsilon_decay: float = 8000.0
    plasticity_alpha_state: float = 0.6
    plasticity_alpha_reward: float = 0.4
    plasticity_alpha_uncertainty: float = 1.0
    plasticity_lr_beta: float = 1.2
    plasticity_lambda_entropy: float = 0.9
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_meta: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 100_000
    warmup_steps: int = 256
    total_steps: int = 1_000
    update_every: int = 2
    polyak: float = 0.995
    beta_min: float = 0.2
    beta_max: float = 1.1
    homeostat_target: float = 0.35
    homeostat_tau: float = 0.01
    tc_penalty: float = 0.0005
    transaction_cost: float = 0.0005
    replicator_min: float = 1e-3
    crisis_alpha_vol: float = 4.5
    crisis_alpha_dd: float = 6.0
    max_weight_change: float = 0.06
    no_trade_threshold: float = 0.01
    max_total_weight_change: float = 0.35
    cooldown_steps: int = 7
    sector_max_weight_change: float = 0.2
    sector_no_trade_threshold: float = 0.02
    sector_max_total_weight_change: float = 0.7
    risk_penalty_base: float = 0.3
    risk_penalty_beta: float = 0.4
    turnover_penalty_gamma: float = 1.5
    turnover_penalty_threshold: float = 0.0
