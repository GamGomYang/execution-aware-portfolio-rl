import numpy as np
import pandas as pd
import pytest

from prl.data_sources import data_quality_check


def test_data_quality_rejects_flat_series():
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    prices = pd.DataFrame(100.0, index=dates, columns=["AAA"])
    log_returns = pd.DataFrame(0.0, index=dates, columns=["AAA"])
    with pytest.raises(RuntimeError, match="DATA_QUALITY_FAILED"):
        data_quality_check(prices, log_returns)


def test_data_quality_accepts_realistic_random_walk():
    dates = pd.date_range("2010-01-01", periods=700, freq="B")
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.0005, scale=0.01, size=len(dates))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps)), index=dates, columns=["AAA"])
    log_returns = pd.DataFrame(np.log(prices / prices.shift(1)), index=dates, columns=["AAA"]).dropna()
    data_quality_check(prices.loc[log_returns.index], log_returns)
