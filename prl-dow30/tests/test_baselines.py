import numpy as np
import pandas as pd

from prl.baselines import (
    _inverse_vol_weights,
    buy_and_hold_equal_weight,
    daily_rebalanced_equal_weight,
    inverse_vol_risk_parity,
)


def _make_frames():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB", "CCC", "DDD"])
    vol = pd.DataFrame(
        {
            "AAA": np.linspace(0.01, 0.02, len(dates)),
            "BBB": np.linspace(0.02, 0.03, len(dates)),
            "CCC": np.linspace(0.03, 0.04, len(dates)),
            "DDD": np.linspace(0.04, 0.05, len(dates)),
        },
        index=dates,
    )
    return returns, vol


def test_buy_and_hold_turnover_zero():
    returns, vol = _make_frames()
    ts = buy_and_hold_equal_weight(returns=returns, volatility=vol, lookback=2)
    assert len(ts.turnover) == len(returns) - 2
    assert np.allclose(ts.turnover, 0.0)


def test_daily_equal_has_non_negative_turnover():
    returns, vol = _make_frames()
    ts = daily_rebalanced_equal_weight(returns=returns, volatility=vol, lookback=2)
    assert len(ts.turnover) == len(returns) - 2
    assert np.all(ts.turnover >= 0.0)


def test_inverse_vol_weights_properties():
    vol_row = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    weights = _inverse_vol_weights(vol_row, eps=1e-8)
    assert weights is not None
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)


def test_inverse_vol_risk_parity_turnover_nonzero():
    returns, vol = _make_frames()
    ts = inverse_vol_risk_parity(returns=returns, volatility=vol, lookback=2)
    assert len(ts.turnover) == len(returns) - 2
    assert np.any(ts.turnover > 0.0)
