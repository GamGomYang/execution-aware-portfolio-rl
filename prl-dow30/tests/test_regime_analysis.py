import numpy as np
import pandas as pd

from prl.analysis import assign_vol_regimes, compute_regime_metrics


def test_regime_analysis_coverages():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    vol = pd.Series(np.linspace(0.1, 1.0, len(dates)), index=dates)
    portfolio_return = np.linspace(0.001, 0.002, len(dates))
    turnover = np.linspace(0.01, 0.02, len(dates))

    regimes = assign_vol_regimes(vol, n_bins=3)
    assert len(regimes) == len(dates)
    assert regimes.isna().sum() == 0

    metrics = compute_regime_metrics(
        model_type="baseline",
        dates=dates,
        portfolio_return=portfolio_return,
        turnover=turnover,
        vol_portfolio=vol.to_numpy(),
        n_bins=3,
    )
    assert len(metrics) == 3
    assert metrics["n_obs"].sum() == len(dates)
    for col in ["cumulative_return", "sharpe", "max_drawdown", "avg_turnover", "total_turnover"]:
        assert metrics[col].isna().sum() == 0
