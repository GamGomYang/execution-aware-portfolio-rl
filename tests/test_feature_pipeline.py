import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)

    def _dummy_device(name: str) -> str:
        return name

    torch_stub.device = _dummy_device
    sys.modules["torch"] = torch_stub

from mprl.config import FeatureConfig
from mprl.features import MarketFeatureAssembler
from mprl.logging_utils import DecisionLogger


def _make_portfolio_ctx(cfg: FeatureConfig) -> dict:
    weights = np.ones(cfg.num_assets, dtype=np.float32) / cfg.num_assets
    return {
        "weights": weights,
        "rolling_vol": 0.01,
        "drawdown": -0.02,
        "cash_ratio": 0.0,
        "turnover": 0.0,
        "win_rate": 0.5,
        "mdd_30d": -0.02,
        "exposure_var": 0.001,
        "hedge_ratio": 0.1,
        "beta_to_spx": 1.0,
        "beta_to_vix": -0.2,
        "port_skew": 0.0,
        "port_kurt": 3.0,
        "leverage": 1.0,
        "liquidity_score": 0.8,
    }


def _make_macro_state(cfg: FeatureConfig, base: float = 0.1) -> dict:
    macro = {}
    for idx, key in enumerate(cfg.macro_features):
        macro[key] = base * (idx + 1)
    return macro


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = FeatureConfig()
        self.assembler = MarketFeatureAssembler(self.cfg)
        self.lookback = self.cfg.lookback
        rng = np.random.default_rng(42)
        self.asset_hist = rng.normal(0.0, 0.01, size=(self.lookback, self.cfg.num_assets)).astype(np.float32)
        self.sector_hist = rng.normal(0.0, 0.008, size=(self.lookback, len(self.cfg.sector_etfs))).astype(np.float32)

    def test_assembler_sanitizes_nan_inputs(self) -> None:
        asset_hist = self.asset_hist.copy()
        sector_hist = self.sector_hist.copy()
        asset_hist[0, 0] = np.nan
        sector_hist[1, 1] = np.nan
        macro_state = _make_macro_state(self.cfg)
        portfolio_ctx = _make_portfolio_ctx(self.cfg)
        portfolio_ctx["weights"][3] = np.nan
        portfolio_ctx["drawdown"] = np.nan

        state_vec, risk_feats, signal = self.assembler.build_state(
            asset_hist, sector_hist, macro_state, portfolio_ctx
        )
        self.assertTrue(np.isfinite(state_vec).all())
        self.assertTrue(np.isfinite(risk_feats).all())
        self.assertTrue(np.isfinite(signal))

    def test_macro_forward_fill_and_logger_dump(self) -> None:
        macro_state = _make_macro_state(self.cfg)
        portfolio_ctx = _make_portfolio_ctx(self.cfg)
        self.assembler.build_state(self.asset_hist, self.sector_hist, macro_state, portfolio_ctx)

        macro_nan = {key: float("nan") for key in self.cfg.macro_features}
        _, risk_feats, signal = self.assembler.build_state(
            self.asset_hist, self.sector_hist, macro_nan, portfolio_ctx
        )
        self.assertTrue(np.isfinite(risk_feats).all())
        self.assertTrue(np.isfinite(signal))

        with tempfile.TemporaryDirectory() as tmpdir:
            action = np.ones(self.cfg.num_assets, dtype=np.float32) / self.cfg.num_assets
            logger = DecisionLogger(tmpdir, self.cfg)
            zero_risk = np.zeros_like(risk_feats)
            logger.log_step(
                step=0,
                date="2024-01-01",
                beta=0.5,
                action=action,
                reward=0.0,
                risk_metric=0.0,
                stress_signal=0.0,
                risk_features=zero_risk,
                plasticity=0.0,
                portfolio_return=0.0,
            )
            logger.close()
            raw_path = Path(tmpdir) / "raw_features.jsonl"
            with raw_path.open("r", encoding="utf-8") as fp:
                lines = [json.loads(line) for line in fp if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["reason"], "zero_vector")
            self.assertEqual(len(lines[0]["risk_features"]), self.cfg.risk_feature_dim)


if __name__ == "__main__":
    unittest.main()
