import numpy as np
import pandas as pd
import pytest

from prl.envs import Dow30PortfolioEnv, EnvConfig
from prl.eval import assert_env_compatible
from prl.utils.signature import compute_env_signature


def test_eval_fails_on_mismatch():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
    volatility = pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"])
    env = Dow30PortfolioEnv(
        EnvConfig(
            returns=returns,
            volatility=volatility,
            window_size=2,
            transaction_cost=0.0,
            logit_scale=1.0,
        )
    )
    env.reset()

    expected_assets = ["AAA", "BBB", "CCC"]
    feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True}
    cost_params = {"transaction_cost": 0.0}
    expected_sig = compute_env_signature(expected_assets, 2, 2, feature_flags, cost_params, "v1")
    run_metadata = {
        "asset_list": expected_assets,
        "num_assets": 3,
        "obs_dim_expected": 3 * (2 + 2),
        "env_signature_hash": expected_sig,
    }

    with pytest.raises(ValueError, match="ENV_COMPATIBILITY_MISMATCH"):
        assert_env_compatible(env, run_metadata, Lv=2)
