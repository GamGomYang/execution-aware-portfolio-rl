from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.step6_build_reports import (
    LEGACY_KAPPA0_PAIRED_COLUMNS,
    MAIN_BASELINE_PAIRED_COLUMNS,
    RULEVOL_FIXED_PAIRED_COLUMNS,
    _build_paired_delta,
)


def test_paired_delta_main_vs_baseline_regression() -> None:
    runs = pd.DataFrame(
        [
            {
                "arm": "main",
                "kappa": 0.001,
                "seed": 0,
                "pair_eta": 0.1,
                "sharpe_net_lin": 0.4,
                "cagr": 0.2,
                "maxdd": -0.3,
                "run_id": "main_s0",
            },
            {
                "arm": "baseline",
                "kappa": 0.001,
                "seed": 0,
                "pair_eta": 1.0,
                "sharpe_net_lin": 0.3,
                "cagr": 0.1,
                "maxdd": -0.4,
                "run_id": "base_s0",
            },
        ]
    )

    paired = _build_paired_delta(runs)

    assert list(paired.columns) == MAIN_BASELINE_PAIRED_COLUMNS
    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["eta_main"] == 0.1
    assert row["eta_baseline"] == 1.0
    assert row["baseline_run_id"] == "base_s0"
    assert np.isclose(row["delta_sharpe"], 0.1)
    assert np.isclose(row["delta_cagr"], 0.1)
    assert np.isclose(row["delta_maxdd"], 0.1)


def test_paired_delta_rulevol_vs_fixed_comparison() -> None:
    runs = pd.DataFrame(
        [
            {
                "arm": "fixed_comparison",
                "kappa": 0.001,
                "seed": 0,
                "pair_eta": 0.1,
                "eta": 0.1,
                "rule_vol_a": np.nan,
                "sharpe_net_lin": 0.25,
                "cagr": 0.12,
                "maxdd": -0.22,
                "run_id": "fixed_s0",
            },
            {
                "arm": "fixed_comparison",
                "kappa": 0.001,
                "seed": 1,
                "pair_eta": 0.1,
                "eta": 0.1,
                "rule_vol_a": np.nan,
                "sharpe_net_lin": 0.20,
                "cagr": 0.10,
                "maxdd": -0.20,
                "run_id": "fixed_s1",
            },
            {
                "arm": "rule_vol",
                "kappa": 0.001,
                "seed": 0,
                "pair_eta": 0.5,
                "eta": 0.5,
                "rule_vol_a": 0.5,
                "sharpe_net_lin": 0.35,
                "cagr": 0.20,
                "maxdd": -0.18,
                "run_id": "rule_s0_a05",
            },
            {
                "arm": "rule_vol",
                "kappa": 0.001,
                "seed": 0,
                "pair_eta": 0.7,
                "eta": 0.7,
                "rule_vol_a": 1.0,
                "sharpe_net_lin": 0.33,
                "cagr": 0.18,
                "maxdd": -0.19,
                "run_id": "rule_s0_a10",
            },
            {
                "arm": "rule_vol",
                "kappa": 0.001,
                "seed": 1,
                "pair_eta": 0.6,
                "eta": 0.6,
                "rule_vol_a": 0.5,
                "sharpe_net_lin": 0.23,
                "cagr": 0.12,
                "maxdd": -0.17,
                "run_id": "rule_s1_a05",
            },
        ]
    )

    paired = _build_paired_delta(runs)

    assert list(paired.columns) == RULEVOL_FIXED_PAIRED_COLUMNS
    assert len(paired) == 3
    assert (paired["arm_pair"] == "rule_vol_vs_fixed").all()
    assert np.allclose(paired["eta"].to_numpy(), paired["eta_rule"].to_numpy(), atol=0.0, rtol=0.0)
    assert paired.sort_values(["seed", "kappa", "rule_vol_a", "eta_rule"]).index.tolist() == paired.index.tolist()

    s0_a05 = paired[(paired["seed"] == 0) & (np.isclose(paired["rule_vol_a"], 0.5))].iloc[0]
    assert s0_a05["baseline_run_id"] == "fixed_s0"
    assert np.isclose(s0_a05["delta_sharpe"], 0.10)
    assert np.isclose(s0_a05["delta_cagr"], 0.08)
    assert np.isclose(s0_a05["delta_maxdd"], 0.04)
    assert np.isclose(s0_a05["eta_fixed"], 0.1)


def test_paired_delta_kappa0_fallback_regression() -> None:
    runs = pd.DataFrame(
        [
            {
                "arm": "eta_sweep",
                "kappa": 0.0,
                "seed": 0,
                "pair_eta": 0.1,
                "sharpe_net_lin": 0.20,
                "cagr": 0.10,
                "maxdd": -0.30,
                "run_id": "k0_eta01",
            },
            {
                "arm": "eta_sweep",
                "kappa": 0.001,
                "seed": 0,
                "pair_eta": 0.1,
                "sharpe_net_lin": 0.25,
                "cagr": 0.14,
                "maxdd": -0.28,
                "run_id": "k1_eta01",
            },
        ]
    )

    paired = _build_paired_delta(runs)

    assert list(paired.columns) == LEGACY_KAPPA0_PAIRED_COLUMNS
    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["run_id"] == "k1_eta01"
    assert row["baseline_run_id"] == "k0_eta01"
    assert np.isclose(row["eta"], 0.1)
    assert np.isclose(row["delta_sharpe"], 0.05)
    assert np.isclose(row["delta_cagr"], 0.04)
    assert np.isclose(row["delta_maxdd"], 0.02)
