import numpy as np
import pandas as pd

from prl.signals import AVAILABLE_SIGNALS, compute_signal_frames


def _make_prices_and_returns(n_days: int = 420, n_assets: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    rng = np.random.default_rng(1234)
    log_ret = rng.normal(loc=0.0002, scale=0.01, size=(n_days, n_assets))
    price_arr = np.exp(np.cumsum(log_ret, axis=0))
    prices = pd.DataFrame(price_arr, index=dates, columns=cols)
    returns_log = np.log(prices / prices.shift(1)).dropna(how="any")
    prices = prices.loc[returns_log.index]
    return prices, returns_log


def test_signal_generation_has_no_lookahead_and_is_cross_section_zscored():
    prices, returns_log = _make_prices_and_returns()
    base = compute_signal_frames(prices, returns_log, signals="all")

    future_start = prices.index[320]
    prices_mod = prices.copy()
    returns_mod = returns_log.copy()
    returns_mod.loc[returns_mod.index >= future_start, "A0"] = (
        returns_mod.loc[returns_mod.index >= future_start, "A0"] + np.log(4.0)
    )
    mod = compute_signal_frames(prices_mod, returns_mod, signals="all")

    cutoff = prices.index[300]
    for signal_name in AVAILABLE_SIGNALS:
        base_pre = base[signal_name].loc[:cutoff]
        mod_pre = mod[signal_name].loc[:cutoff]
        common_idx = base_pre.index.intersection(mod_pre.index)
        left = base_pre.loc[common_idx].to_numpy(dtype=np.float64)
        right = mod_pre.loc[common_idx].to_numpy(dtype=np.float64)
        assert np.allclose(left, right, atol=1e-12, rtol=0.0, equal_nan=True), signal_name

        frame = base[signal_name]
        valid_rows = frame.notna().sum(axis=1) >= 2
        if not valid_rows.any():
            continue
        means = frame.loc[valid_rows].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
        stds = frame.loc[valid_rows].std(axis=1, ddof=0, skipna=True).to_numpy(dtype=np.float64)
        assert np.allclose(means, 0.0, atol=1e-10, rtol=0.0, equal_nan=True), signal_name
        assert np.allclose(stds, 1.0, atol=1e-8, rtol=0.0, equal_nan=True), signal_name
