from prl.utils.signature import compute_env_signature


def test_env_signature_deterministic():
    asset_list = ["AAA", "BBB"]
    feature_flags = {"returns_window": True, "volatility": True, "prev_weights": True}
    cost_params = {"transaction_cost": 0.001}
    sig1 = compute_env_signature(asset_list, 2, 2, feature_flags, cost_params, "v1")
    sig2 = compute_env_signature(list(asset_list), 2, 2, dict(feature_flags), dict(cost_params), "v1")
    assert sig1 == sig2
