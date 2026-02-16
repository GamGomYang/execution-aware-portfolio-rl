from prl.utils.signature import compute_env_signature


def test_env_signature_changes_with_rebalance_eta():
    asset_list = ["AAA", "BBB"]
    feature_flags = {
        "returns_window": True,
        "volatility": True,
        "prev_weights": True,
        "reward_type": "log_net_minus_r2",
        "action_smoothing": True,
    }
    cost_params_a = {"transaction_cost": 0.001, "risk_lambda": 0.0, "rebalance_eta": None}
    cost_params_b = {"transaction_cost": 0.001, "risk_lambda": 0.0, "rebalance_eta": 0.1}
    sig_a = compute_env_signature(asset_list, 2, 2, feature_flags, cost_params_a, "v1")
    sig_b = compute_env_signature(asset_list, 2, 2, feature_flags, cost_params_b, "v1")
    assert sig_a != sig_b
