import numpy as np
import pandas as pd

from prl.data import load_market_data
from prl.features import compute_volatility_features
from prl.envs import Dow30PortfolioEnv, EnvConfig


def _fake_fetch(tickers, start, end, session_opts=None):
    ticker = list(tickers)[0]
    idx = pd.date_range(start=start, periods=20, freq="B")
    base = {"ZZZ": 100.0, "AAA": 200.0, "MMM": 300.0}[ticker]
    # 서로 다른 스케일을 줘서 변동성 크기가 다르게 유지되도록 함
    data = pd.DataFrame({ticker: base + pd.RangeIndex(len(idx)) * (1 if ticker == "ZZZ" else (2 if ticker == "AAA" else 3))}, index=idx)
    return data


def test_asset_order_propagates_to_env(tmp_path, monkeypatch):
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    tickers = ["ZZZ", "AAA", "MMM"]
    cfg = {
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-02-15",
            "test_start": "2020-02-18",
            "test_end": "2020-03-31",
        },
        "data": {
            "processed_dir": str(tmp_path / "processed"),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 1,
            "force_refresh": True,
            "offline": False,
            "require_cache": False,
            "paper_mode": False,
            "min_history_days": 5,
            "history_tolerance_days": 0,
            "ticker_substitutions": {},
            "tickers": tickers,
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_flat_fraction": 1.0,
                "max_missing_fraction": 1.0,
            },
        },
        "env": {"L": 3, "Lv": 3, "c_tc": 0.0},
    }

    prices, returns, manifest, _ = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )

    assert manifest["kept_tickers"] == tickers
    assert list(prices.columns) == tickers
    assert list(returns.columns) == tickers

    features = compute_volatility_features(
        returns=returns,
        lv=cfg["env"]["Lv"],
        train_start=cfg["dates"]["train_start"],
        train_end=cfg["dates"]["train_end"],
        processed_dir=cfg["data"]["processed_dir"],
    )
    assert list(features.volatility.columns) == tickers

    returns_slice = returns.loc[cfg["dates"]["train_start"] : cfg["dates"]["train_end"]]
    vol_slice = features.volatility.loc[cfg["dates"]["train_start"] : cfg["dates"]["train_end"]]
    idx = returns_slice.index.intersection(vol_slice.index)
    returns_aligned = returns_slice.loc[idx]
    vol_aligned = vol_slice.loc[idx]

    env = Dow30PortfolioEnv(
        EnvConfig(
            returns=returns_aligned,
            volatility=vol_aligned,
            window_size=cfg["env"]["L"],
            transaction_cost=cfg["env"]["c_tc"],
        )
    )
    assert list(env.volatility.columns) == tickers

    # vol_vector 순서가 returns/manifest 순서와 동일함을 간접 확인
    obs, _ = env.reset()
    vol_vector = env._get_vol_vector()
    expected_vol_vector = env.volatility.iloc[env.current_step - 1].to_numpy(copy=True)
    np.testing.assert_allclose(vol_vector, expected_vol_vector)
