"""Configuration dataclasses for feature engineering and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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
            "WBA",
            "WMT",
        ]
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
    def num_assets(self) -> int:
        return len(self.dow_symbols)

    @property
    def micro_dim(self) -> int:
        return self.num_assets * len(self.micro_features)

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
    def total_state_dim(self) -> int:
        return self.micro_dim + self.sector_dim + self.macro_dim + self.portfolio_dim

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
    alpha: float = 0.1
    k_alpha: float = 0.6
    k_q: float = 0.4
    risk_lambda: float = 2.5
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_meta: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 100_000
    warmup_steps: int = 256
    total_steps: int = 1_000
    update_every: int = 2
    polyak: float = 0.995
    beta_min: float = 0.05
    beta_max: float = 1.5
    homeostat_target: float = 0.35
    homeostat_tau: float = 0.01
    tc_penalty: float = 0.0005
    transaction_cost: float = 0.001
