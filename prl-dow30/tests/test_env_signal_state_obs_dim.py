import numpy as np
import pandas as pd

from prl.envs import Dow30PortfolioEnv, EnvConfig


def _build_frames():
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    assets = ["AAA", "BBB", "CCC"]
    returns = pd.DataFrame(0.001, index=dates, columns=assets, dtype=np.float32)
    volatility = pd.DataFrame(0.02, index=dates, columns=assets, dtype=np.float32)
    reversal_5d = pd.DataFrame(0.1, index=dates, columns=assets, dtype=np.float32)
    short_term_reversal = pd.DataFrame(-0.2, index=dates, columns=assets, dtype=np.float32)
    signal_features = pd.concat(
        {
            "reversal_5d": reversal_5d,
            "short_term_reversal": short_term_reversal,
        },
        axis=1,
    )
    return returns, volatility, signal_features


def test_signal_state_obs_dim_growth_matches_n_times_s():
    returns, volatility, signal_features = _build_frames()
    window_size = 3
    num_assets = returns.shape[1]
    num_signals = 2

    env_off = Dow30PortfolioEnv(
        EnvConfig(
            returns=returns,
            volatility=volatility,
            window_size=window_size,
            transaction_cost=0.0,
            logit_scale=1.0,
        )
    )
    obs_off, _ = env_off.reset()
    expected_off = window_size * num_assets + 2 * num_assets
    assert obs_off.shape == (expected_off,)

    env_on = Dow30PortfolioEnv(
        EnvConfig(
            returns=returns,
            volatility=volatility,
            window_size=window_size,
            transaction_cost=0.0,
            logit_scale=1.0,
            signal_features=signal_features,
        )
    )
    obs_on, _ = env_on.reset()
    expected_on = expected_off + num_assets * num_signals
    assert env_on.signal_state is True
    assert env_on.signal_names == ["reversal_5d", "short_term_reversal"]
    assert obs_on.shape == (expected_on,)

    signal_tail = obs_on[-(num_assets * num_signals) :]
    expected_signal_tail = signal_features.iloc[window_size - 1].to_numpy()
    np.testing.assert_allclose(signal_tail, expected_signal_tail.astype(np.float32))
