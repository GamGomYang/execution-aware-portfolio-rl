from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.data import MarketData
from prl.envs import Dow30PortfolioEnv, EnvConfig, stable_softmax
from prl.eval import assert_env_compatible
from prl.features import VolatilityFeatures
from prl.train import (
    _compute_action_smoothing_flag,
    _is_step6_signature_extension_active,
    _parse_step6_env_params,
    build_env_for_range,
)
from prl.utils.signature import compute_env_signature


ATOL = 1e-12


def _assert_close(name: str, left: float, right: float, atol: float = ATOL) -> None:
    if not np.isclose(float(left), float(right), atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch: left={left}, right={right}, atol={atol}")


def _assert_true(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def _make_frames(*, n_steps: int = 20, n_assets: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="B")
    rng = np.random.default_rng(31)
    arithmetic_returns = rng.normal(loc=0.001, scale=0.01, size=(n_steps, n_assets)).astype(np.float32)
    arithmetic_returns = np.clip(arithmetic_returns, -0.2, 0.2)
    log_returns = np.log1p(arithmetic_returns)
    returns = pd.DataFrame(log_returns, index=dates, columns=[f"A{i}" for i in range(n_assets)])

    volatility = pd.DataFrame(
        rng.uniform(low=0.05, high=0.30, size=(n_steps, n_assets)).astype(np.float32),
        index=dates,
        columns=returns.columns,
    )
    prices = pd.DataFrame(
        np.exp(log_returns).cumprod(axis=0).astype(np.float32),
        index=dates,
        columns=returns.columns,
    )
    return returns, volatility, prices


def _make_market_features(returns: pd.DataFrame, volatility: pd.DataFrame, prices: pd.DataFrame) -> tuple[MarketData, VolatilityFeatures]:
    market = MarketData(prices=prices, returns=returns, manifest={}, quality_summary=None)
    portfolio_scalar = volatility.mean(axis=1)
    features = VolatilityFeatures(
        volatility=volatility,
        portfolio_scalar=portfolio_scalar,
        stats_path=Path("data/processed/step6_commit3_dummy_vol_stats.json"),
        mean=float(portfolio_scalar.mean()),
        std=float(max(portfolio_scalar.std(ddof=0), 1e-8)),
    )
    return market, features


def _build_env_via_training_path(
    *,
    env_yaml: dict[str, Any],
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    prices: pd.DataFrame,
    seed: int = 7,
) -> DummyVecEnv:
    market, features = _make_market_features(returns, volatility, prices)
    step6 = _parse_step6_env_params(env_yaml)
    return build_env_for_range(
        market=market,
        features=features,
        start=str(returns.index[0].date()),
        end=str(returns.index[-1].date()),
        window_size=3,
        c_tc=0.001,
        seed=seed,
        logit_scale=1.0,
        random_reset=False,
        risk_lambda=0.0,
        risk_penalty_type="r2",
        rebalance_eta=step6["rebalance_eta"],
        eta_mode=step6["eta_mode"],
        rule_vol_window=step6["rule_vol_window"],
        rule_vol_a=step6["rule_vol_a"],
        eta_clip_min=step6["eta_clip_min"],
        eta_clip_max=step6["eta_clip_max"],
    )


def _build_signature_from_step6(
    *,
    asset_list: list[str],
    L: int,
    Lv: int,
    transaction_cost: float,
    risk_lambda: float,
    reward_type: str,
    step6: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    feature_flags = {
        "returns_window": True,
        "volatility": True,
        "prev_weights": True,
        "reward_type": reward_type,
        "action_smoothing": _compute_action_smoothing_flag(
            eta_mode=str(step6["eta_mode"]),
            rebalance_eta=step6["rebalance_eta"],
        ),
    }
    cost_params = {
        "transaction_cost": float(transaction_cost),
        "risk_lambda": float(risk_lambda),
        "rebalance_eta": step6["rebalance_eta"],
    }
    if _is_step6_signature_extension_active(step6):
        cost_params.update(
            {
                "eta_mode": str(step6["eta_mode"]),
                "eta_clip_min": float(step6["eta_clip_min"]),
                "eta_clip_max": float(step6["eta_clip_max"]),
                "rule_vol_window": int(step6["rule_vol_window"]),
                "rule_vol_a": float(step6["rule_vol_a"]),
            }
        )
    sig = compute_env_signature(
        asset_list,
        L,
        Lv,
        feature_flags=feature_flags,
        cost_params=cost_params,
        schema_version="v1",
    )
    return sig, feature_flags, cost_params


def _build_old_legacy_signature(
    *,
    asset_list: list[str],
    L: int,
    Lv: int,
    transaction_cost: float,
    risk_lambda: float,
    reward_type: str,
    rebalance_eta: float | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    feature_flags = {
        "returns_window": True,
        "volatility": True,
        "prev_weights": True,
        "reward_type": reward_type,
        "action_smoothing": rebalance_eta is not None,
    }
    cost_params = {
        "transaction_cost": float(transaction_cost),
        "risk_lambda": float(risk_lambda),
        "rebalance_eta": rebalance_eta,
    }
    sig = compute_env_signature(
        asset_list,
        L,
        Lv,
        feature_flags=feature_flags,
        cost_params=cost_params,
        schema_version="v1",
    )
    return sig, feature_flags, cost_params


def _metadata_for_env(env: DummyVecEnv, *, Lv: int, reward_type: str = "log_net_minus_r2") -> dict[str, Any]:
    base_env = env.envs[0]
    asset_list = list(base_env.returns.columns)
    step6 = {
        "eta_mode": str(getattr(base_env.cfg, "eta_mode", EnvConfig.eta_mode)),
        "rebalance_eta": float(base_env.cfg.rebalance_eta) if getattr(base_env.cfg, "rebalance_eta", None) is not None else None,
        "rule_vol_window": int(getattr(base_env.cfg, "rule_vol_window", EnvConfig.rule_vol_window)),
        "rule_vol_a": float(getattr(base_env.cfg, "rule_vol_a", EnvConfig.rule_vol_a)),
        "eta_clip_min": float(getattr(base_env.cfg, "eta_clip_min", EnvConfig.eta_clip_min)),
        "eta_clip_max": float(getattr(base_env.cfg, "eta_clip_max", EnvConfig.eta_clip_max)),
    }
    signature, _, _ = _build_signature_from_step6(
        asset_list=asset_list,
        L=int(base_env.window_size),
        Lv=int(Lv),
        transaction_cost=float(base_env.cfg.transaction_cost),
        risk_lambda=float(getattr(base_env.cfg, "risk_lambda", 0.0)),
        reward_type=reward_type,
        step6=step6,
    )
    return {
        "env_signature_version": "v3",
        "env_signature_hash": signature,
        "env_params": {"reward_type": reward_type},
        "obs_dim_expected": int(base_env.observation_space.shape[0]),
        "num_assets": int(base_env.num_assets),
        "asset_list": asset_list,
    }


def test_yaml_propagation() -> None:
    returns, volatility, prices = _make_frames()
    env_yaml = {
        "eta_mode": "rule_vol",
        "rebalance_eta": 0.1,
        "rule_vol": {"window": 30, "a": 2.0, "eta_clip": [0.05, 0.4]},
    }
    env = _build_env_via_training_path(env_yaml=env_yaml, returns=returns, volatility=volatility, prices=prices)
    cfg = env.envs[0].cfg
    _assert_true("TEST1 eta_mode", cfg.eta_mode == "rule_vol")
    _assert_close("TEST1 rebalance_eta", float(cfg.rebalance_eta), 0.1)
    _assert_true("TEST1 rule_vol_window", int(cfg.rule_vol_window) == 30)
    _assert_close("TEST1 rule_vol_a", float(cfg.rule_vol_a), 2.0)
    _assert_close("TEST1 eta_clip_min", float(cfg.eta_clip_min), 0.05)
    _assert_close("TEST1 eta_clip_max", float(cfg.eta_clip_max), 0.4)


def test_legacy_signature_invariance() -> None:
    asset_list = ["A0", "A1", "A2", "A3"]
    L = 3
    Lv = 20
    transaction_cost = 0.001
    risk_lambda = 0.0
    reward_type = "log_net_minus_r2"

    step6 = _parse_step6_env_params({})
    new_sig, feature_flags, cost_params = _build_signature_from_step6(
        asset_list=asset_list,
        L=L,
        Lv=Lv,
        transaction_cost=transaction_cost,
        risk_lambda=risk_lambda,
        reward_type=reward_type,
        step6=step6,
    )
    old_sig, _, _ = _build_old_legacy_signature(
        asset_list=asset_list,
        L=L,
        Lv=Lv,
        transaction_cost=transaction_cost,
        risk_lambda=risk_lambda,
        reward_type=reward_type,
        rebalance_eta=None,
    )
    _assert_true("TEST2 legacy signature equality", new_sig == old_sig)

    forbidden = {"eta_mode", "eta_clip_min", "eta_clip_max", "rule_vol_window", "rule_vol_a"}
    _assert_true("TEST2 no step6 keys in cost_params", forbidden.isdisjoint(set(cost_params.keys())))
    _assert_true("TEST2 action_smoothing False", bool(feature_flags["action_smoothing"]) is False)


def test_step6_signature_difference() -> None:
    asset_list = ["A0", "A1", "A2", "A3"]
    L = 3
    Lv = 20
    transaction_cost = 0.001
    risk_lambda = 0.0
    reward_type = "log_net_minus_r2"

    legacy = _parse_step6_env_params({})
    legacy_sig, _, _ = _build_signature_from_step6(
        asset_list=asset_list,
        L=L,
        Lv=Lv,
        transaction_cost=transaction_cost,
        risk_lambda=risk_lambda,
        reward_type=reward_type,
        step6=legacy,
    )

    step6_mode_changed = _parse_step6_env_params(
        {"eta_mode": "rule_vol", "rule_vol": {"window": 30, "a": 1.0, "eta_clip": [0.02, 0.5]}}
    )
    sig_mode_changed, _, _ = _build_signature_from_step6(
        asset_list=asset_list,
        L=L,
        Lv=Lv,
        transaction_cost=transaction_cost,
        risk_lambda=risk_lambda,
        reward_type=reward_type,
        step6=step6_mode_changed,
    )
    _assert_true("TEST3 signature differs when eta_mode changes", sig_mode_changed != legacy_sig)

    step6_a_changed = _parse_step6_env_params({"eta_mode": "legacy", "rule_vol": {"a": 2.0}})
    sig_a_changed, _, _ = _build_signature_from_step6(
        asset_list=asset_list,
        L=L,
        Lv=Lv,
        transaction_cost=transaction_cost,
        risk_lambda=risk_lambda,
        reward_type=reward_type,
        step6=step6_a_changed,
    )
    _assert_true("TEST3 signature differs when rule_vol_a changes", sig_a_changed != legacy_sig)


def test_action_smoothing_flag() -> None:
    cases = [
        ("legacy", None, False),
        ("legacy", 0.1, True),
        ("fixed", 0.1, True),
        ("rule_vol", None, True),
        ("none", None, False),
    ]
    for eta_mode, rebalance_eta, expected in cases:
        actual = _compute_action_smoothing_flag(eta_mode=eta_mode, rebalance_eta=rebalance_eta)
        _assert_true(
            f"TEST4 action_smoothing case eta_mode={eta_mode}, rebalance_eta={rebalance_eta}",
            bool(actual) is bool(expected),
        )


def test_assert_env_compatible_mismatch() -> None:
    returns, volatility, prices = _make_frames()
    env = _build_env_via_training_path(env_yaml={"eta_mode": "rule_vol"}, returns=returns, volatility=volatility, prices=prices)
    metadata = _metadata_for_env(env, Lv=20)

    # Should pass with correct signature.
    assert_env_compatible(env, metadata, Lv=20)

    bad = dict(metadata)
    bad["env_signature_hash"] = "0" * 64
    try:
        assert_env_compatible(env, bad, Lv=20)
    except ValueError as exc:
        msg = str(exc)
        _assert_true("TEST5 mismatch error tag", "ENV_COMPATIBILITY_MISMATCH" in msg)
        _assert_true("TEST5 mismatch includes signature reason", "env_signature_expected=" in msg)
    else:
        raise AssertionError("TEST5 expected mismatch ValueError was not raised")


def test_freeze_integrity() -> None:
    src = inspect.getsource(Dow30PortfolioEnv.step)
    required_snippets = (
        "w_exec = (1.0 - eta_t) * prev_weights + eta_t * w_target",
        "turnover_target = turnover_l1(prev_weights, w_target)",
        "turnover_exec = turnover_l1(prev_weights, w_exec)",
        "cost_exec = self.cfg.transaction_cost * turnover_exec",
        "cost_target = self.cfg.transaction_cost * turnover_target",
        "portfolio_return = float(np.dot(w_exec, arithmetic_returns))",
        "reward = log_return_net - risk_penalty",
    )
    for snippet in required_snippets:
        _assert_true(f"TEST6 required snippet missing: {snippet}", snippet in src)
    forbidden_snippets = (
        "portfolio_return = float(np.dot(w_target, arithmetic_returns))",
        "reward = log_return_net_target",
    )
    for snippet in forbidden_snippets:
        _assert_true(f"TEST6 forbidden snippet found: {snippet}", snippet not in src)

    # Numeric freeze sanity: execution-based reward and convex execution update.
    returns, volatility, _ = _make_frames()
    cfg = EnvConfig(
        returns=returns,
        volatility=volatility,
        window_size=3,
        transaction_cost=0.001,
        logit_scale=1.0,
        eta_mode="fixed",
        rebalance_eta=0.1,
    )
    env = Dow30PortfolioEnv(cfg)
    env.reset(seed=123)
    action = np.array([1.2, -0.4, 0.7, -1.0], dtype=np.float32)

    prev_weights = env.prev_weights.astype(np.float64).copy()
    z = np.clip(action, env.action_space.low, env.action_space.high)
    w_target = stable_softmax(z, scale=env.cfg.logit_scale).astype(np.float64)
    step_idx = env.current_step
    arithmetic_returns = np.expm1(env.returns.iloc[step_idx].to_numpy(copy=False))

    _, reward, _, _, info = env.step(action)
    eta_t = float(info["eta_t"])
    w_exec_expected = env._safe_normalize_weights((1.0 - eta_t) * prev_weights + eta_t * w_target)
    w_exec_observed = env.prev_weights.astype(np.float64)

    _assert_true("TEST6 convex update", np.allclose(w_exec_observed, w_exec_expected, atol=1e-7, rtol=0.0))
    _assert_close(
        "TEST6 execution-based portfolio_return",
        float(info["portfolio_return"]),
        float(np.dot(w_exec_expected, arithmetic_returns)),
        atol=1e-12,
    )
    _assert_close(
        "TEST6 reward formula",
        float(reward),
        float(info["log_return_net"] - info["risk_penalty"]),
        atol=1e-12,
    )


def main() -> None:
    test_yaml_propagation()
    test_legacy_signature_invariance()
    test_step6_signature_difference()
    test_action_smoothing_flag()
    test_assert_env_compatible_mismatch()
    test_freeze_integrity()
    print("COMMIT-3 VERIFIED — SIGNATURE LAYER SAFE")


if __name__ == "__main__":
    main()
