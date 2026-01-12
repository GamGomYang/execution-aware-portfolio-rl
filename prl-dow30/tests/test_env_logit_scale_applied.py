import numpy as np
import pandas as pd

from prl.data import MarketData
from prl.features import VolatilityFeatures
from prl.train import build_env_for_range


def _make_market():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB", "CCC"])
    prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=returns.columns)
    return MarketData(prices=prices, returns=returns)


def test_logit_scale_applied_via_builder(tmp_path):
    market = _make_market()
    vol = pd.DataFrame(0.02, index=market.returns.index, columns=market.returns.columns)
    features = VolatilityFeatures(
        volatility=vol,
        portfolio_scalar=pd.Series(0.02, index=market.returns.index),
        stats_path=tmp_path / "vol_stats.json",
        mean=0.02,
        std=0.01,
    )
    start = market.returns.index[0].date().isoformat()
    end = market.returns.index[-1].date().isoformat()

    env_low = build_env_for_range(
        market=market,
        features=features,
        start=start,
        end=end,
        window_size=2,
        c_tc=0.0,
        seed=0,
        logit_scale=1.0,
    )
    env_high = build_env_for_range(
        market=market,
        features=features,
        start=start,
        end=end,
        window_size=2,
        c_tc=0.0,
        seed=0,
        logit_scale=10.0,
    )

    action = np.array([[0.2, -0.2, -0.2]], dtype=np.float32)
    env_low.reset()
    env_low.step(action)
    w_low = env_low.envs[0].prev_weights.copy()

    env_high.reset()
    env_high.step(action)
    w_high = env_high.envs[0].prev_weights.copy()

    assert np.isclose(w_low.sum(), 1.0)
    assert np.isclose(w_high.sum(), 1.0)
    assert np.all(w_low >= 0.0)
    assert np.all(w_high >= 0.0)
    assert w_high.max() > w_low.max() + 0.2
