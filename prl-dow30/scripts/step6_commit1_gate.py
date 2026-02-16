from __future__ import annotations

import inspect
import sys
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.envs import Dow30PortfolioEnv, EnvConfig, stable_softmax


EPS_STRICT = 1e-12
EPS_TINY = 1e-10
EPS_TRACK_ZERO = 1e-7

REQUIRED_OLD_KEYS = (
    "portfolio_return",
    "turnover",
    "turnover_exec",
    "turnover_target",
    "turnover_target_change",
    "rebalance_eta",
    "cost",
    "log_return_gross",
    "log_return_net",
)

REQUIRED_STEP6_KEYS = (
    "eta_mode",
    "eta_t",
    "lambda_t",
    "cost_exec",
    "cost_target",
    "net_return_lin_exec",
    "net_return_lin_target",
    "tracking_error_l2",
    "collapse_flag",
    "collapse_reason",
)


def _make_base_frames(*, n_steps: int = 12, n_assets: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="B")
    rng = np.random.default_rng(7)
    arithmetic_returns = rng.normal(loc=0.001, scale=0.01, size=(n_steps, n_assets)).astype(np.float32)
    arithmetic_returns = np.clip(arithmetic_returns, -0.2, 0.2)
    log_returns = np.log1p(arithmetic_returns)

    vol = rng.uniform(low=0.05, high=0.30, size=(n_steps, n_assets)).astype(np.float32)
    returns_df = pd.DataFrame(log_returns, index=dates, columns=[f"A{i}" for i in range(n_assets)])
    vol_df = pd.DataFrame(vol, index=dates, columns=returns_df.columns)
    return returns_df, vol_df


def _assert_close(name: str, left: float, right: float, atol: float = EPS_STRICT) -> None:
    if not np.isclose(left, right, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch: left={left}, right={right}, atol={atol}")


def _build_cfg(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    transaction_cost: float = 0.001,
    rebalance_eta: float | None = None,
    with_eta_mode: bool = True,
    eta_mode: str = "legacy",
    rule_vol_a: float = 1.0,
    eta_clip_min: float = 0.02,
    eta_clip_max: float = 0.5,
) -> EnvConfig:
    kwargs: dict[str, Any] = dict(
        returns=returns,
        volatility=volatility,
        window_size=3,
        transaction_cost=transaction_cost,
        logit_scale=1.0,
        rebalance_eta=rebalance_eta,
        rule_vol_a=rule_vol_a,
        eta_clip_min=eta_clip_min,
        eta_clip_max=eta_clip_max,
    )
    if with_eta_mode:
        kwargs["eta_mode"] = eta_mode
    return EnvConfig(**kwargs)


def _run_one_step(
    cfg: EnvConfig,
    action: np.ndarray,
    *,
    preset_prev_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    env = Dow30PortfolioEnv(cfg)
    env.reset(seed=123)
    if preset_prev_weights is not None:
        env.prev_weights = preset_prev_weights.astype(np.float32)

    prev_weights = env.prev_weights.astype(np.float64).copy()
    step_idx = env.current_step
    returns_t = env.returns.iloc[step_idx].to_numpy(copy=False)
    arithmetic_returns = np.expm1(returns_t)

    z = np.clip(action, env.action_space.low, env.action_space.high)
    w_target = stable_softmax(z, scale=env.cfg.logit_scale).astype(np.float64)

    _, reward, _, _, info = env.step(action)
    w_exec_observed = env.prev_weights.astype(np.float64)
    eta_t = float(info["eta_t"])
    w_exec_exact = env._safe_normalize_weights((1.0 - eta_t) * prev_weights + eta_t * w_target)

    return {
        "env": env,
        "reward": float(reward),
        "info": info,
        "prev_weights": prev_weights,
        "w_target": w_target,
        "w_exec_observed": w_exec_observed,
        "w_exec_exact": w_exec_exact,
        "arithmetic_returns": arithmetic_returns,
        "transaction_cost": float(env.cfg.transaction_cost),
    }


def _check_legacy_default_vs_explicit(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg_default = _build_cfg(returns, vol, rebalance_eta=None, with_eta_mode=False)
    cfg_legacy = _build_cfg(returns, vol, rebalance_eta=None, with_eta_mode=True, eta_mode="legacy")

    r_default = _run_one_step(cfg_default, action)
    r_legacy = _run_one_step(cfg_legacy, action)

    _assert_close("legacy replay reward", r_default["reward"], r_legacy["reward"])
    for key in REQUIRED_OLD_KEYS + REQUIRED_STEP6_KEYS:
        left = r_default["info"][key]
        right = r_legacy["info"][key]
        if isinstance(left, float) or isinstance(right, float):
            _assert_close(f"legacy replay {key}", float(left), float(right), atol=EPS_TINY)
        else:
            if left != right:
                raise AssertionError(f"legacy replay {key} mismatch: {left} != {right}")


def _test_legacy_replay_invariance(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    _check_legacy_default_vs_explicit(returns, vol, action)

    cfg = _build_cfg(returns, vol, rebalance_eta=None, with_eta_mode=False)
    result = _run_one_step(cfg, action)
    info = result["info"]
    w_exec_exact = result["w_exec_exact"]
    arithmetic_returns = result["arithmetic_returns"]

    if abs(float(info["tracking_error_l2"])) > EPS_TRACK_ZERO:
        raise AssertionError(
            f"tracking_error_l2 expected ~0 in legacy(rebalance_eta=None), got {info['tracking_error_l2']}"
        )
    _assert_close("eta_t (legacy,None)", float(info["eta_t"]), 1.0)
    _assert_close("cost_target==cost_exec", float(info["cost_target"]), float(info["cost_exec"]), atol=EPS_TINY)
    _assert_close(
        "net_return_lin_target==net_return_lin_exec",
        float(info["net_return_lin_target"]),
        float(info["net_return_lin_exec"]),
        atol=EPS_TINY,
    )

    recomputed = float(np.dot(w_exec_exact, arithmetic_returns))
    _assert_close("portfolio_return invariance", recomputed, float(info["portfolio_return"]), atol=EPS_STRICT)
    _assert_close(
        "log_return invariance",
        float(info["log_return_net"]),
        float(info["log_return_gross"]) - float(info["cost"]),
        atol=EPS_STRICT,
    )


def _test_reward_immutability(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, rebalance_eta=0.2, eta_mode="fixed")
    result = _run_one_step(cfg, action)
    info = result["info"]

    _assert_close(
        "net_return_lin_exec definition",
        float(info["net_return_lin_exec"]),
        float(info["portfolio_return"]) - float(info["cost_exec"]),
        atol=EPS_STRICT,
    )

    src = inspect.getsource(Dow30PortfolioEnv.step)
    if "portfolio_return_target" in src:
        raise AssertionError("Forbidden variable found in step(): portfolio_return_target")


def _test_info_key_completeness(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, rebalance_eta=0.2, eta_mode="fixed")
    result = _run_one_step(cfg, action)
    info = result["info"]

    for key in REQUIRED_OLD_KEYS + REQUIRED_STEP6_KEYS:
        if key not in info:
            raise AssertionError(f"Missing info key: {key}")


def _test_convex_execution_update(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, rebalance_eta=0.1, eta_mode="fixed")
    result = _run_one_step(cfg, action)
    info = result["info"]
    prev_weights = result["prev_weights"]
    w_target = result["w_target"]
    w_exec_observed = result["w_exec_observed"]
    w_exec_exact = result["w_exec_exact"]
    eta_t = float(info["eta_t"])

    expected = (1.0 - eta_t) * prev_weights + eta_t * w_target
    expected = expected / expected.sum()
    if not np.allclose(w_exec_exact, expected, atol=1e-12, rtol=0.0):
        raise AssertionError("w_exec exact path is not a convex update from prev_weights and w_target")
    if not np.allclose(w_exec_observed, expected, atol=1e-8, rtol=0.0):
        raise AssertionError("w_exec is not a convex update from prev_weights and w_target")


def _test_l1_turnover(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, rebalance_eta=0.1, eta_mode="fixed")
    result = _run_one_step(cfg, action)
    info = result["info"]
    prev_weights = result["prev_weights"]
    w_target = result["w_target"]
    w_exec_exact = result["w_exec_exact"]

    turnover_exec = float(np.sum(np.abs(w_exec_exact - prev_weights)))
    turnover_target = float(np.sum(np.abs(w_target - prev_weights)))

    _assert_close("turnover_exec L1", float(info["turnover_exec"]), turnover_exec, atol=EPS_STRICT)
    _assert_close("turnover_target L1", float(info["turnover_target"]), turnover_target, atol=EPS_STRICT)


def _test_cost_split(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, rebalance_eta=0.1, eta_mode="fixed", transaction_cost=0.007)
    result = _run_one_step(cfg, action)
    info = result["info"]
    c_tc = result["transaction_cost"]

    _assert_close(
        "cost_exec split",
        float(info["cost_exec"]),
        c_tc * float(info["turnover_exec"]),
        atol=EPS_STRICT,
    )
    _assert_close(
        "cost_target split",
        float(info["cost_target"]),
        c_tc * float(info["turnover_target"]),
        atol=EPS_STRICT,
    )
    _assert_close("cost alias", float(info["cost"]), float(info["cost_exec"]), atol=EPS_STRICT)


def _test_misalignment_definition(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, rebalance_eta=0.1, eta_mode="fixed")
    result = _run_one_step(cfg, action)
    info = result["info"]

    _assert_close(
        "net_return_lin_target definition",
        float(info["net_return_lin_target"]),
        float(info["portfolio_return"]) - float(info["cost_target"]),
        atol=EPS_STRICT,
    )
    _assert_close(
        "log_return_net_target definition",
        float(info["log_return_net_target"]),
        float(info["log_return_gross"]) - float(info["cost_target"]),
        atol=EPS_STRICT,
    )


def _test_eta_modes(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    legacy = _run_one_step(_build_cfg(returns, vol, eta_mode="legacy", rebalance_eta=None), action)
    info_legacy = legacy["info"]
    _assert_close("legacy eta_t", float(info_legacy["eta_t"]), 1.0)
    if abs(float(info_legacy["tracking_error_l2"])) > EPS_TRACK_ZERO:
        raise AssertionError("legacy tracking_error_l2 should be near zero")

    fixed = _run_one_step(_build_cfg(returns, vol, eta_mode="fixed", rebalance_eta=0.1), action)
    info_fixed = fixed["info"]
    _assert_close("fixed eta_t", float(info_fixed["eta_t"]), 0.1)
    if not float(info_fixed["tracking_error_l2"]) > 0.0:
        raise AssertionError("fixed mode tracking_error_l2 should be > 0")

    none = _run_one_step(_build_cfg(returns, vol, eta_mode="none", rebalance_eta=None), action)
    info_none = none["info"]
    _assert_close("none eta_t", float(info_none["eta_t"]), 1.0)
    if not np.allclose(none["w_exec_exact"], none["w_target"], atol=1e-7, rtol=0.0):
        raise AssertionError("none mode must execute target weights directly")

    rule_vol = _run_one_step(
        _build_cfg(
            returns,
            vol,
            eta_mode="rule_vol",
            rebalance_eta=None,
            rule_vol_a=1.0,
            eta_clip_min=0.02,
            eta_clip_max=0.5,
        ),
        action,
    )
    info_rule = rule_vol["info"]
    if info_rule["lambda_t"] is None:
        raise AssertionError("rule_vol lambda_t must not be None")
    eta_t = float(info_rule["eta_t"])
    if not (0.02 <= eta_t <= 0.5):
        raise AssertionError(f"rule_vol eta_t out of clip bounds: {eta_t}")


def _test_collapse_gate(base_returns: pd.DataFrame, base_vol: pd.DataFrame, action: np.ndarray) -> None:
    returns = base_returns.copy()
    # window_size=3 -> first stepped row index is 3
    returns.iloc[3, 0] = np.inf
    cfg = _build_cfg(returns, base_vol, eta_mode="legacy", rebalance_eta=None)
    result = _run_one_step(cfg, action)
    info = result["info"]
    if not bool(info["collapse_flag"]):
        raise AssertionError("collapse_flag must be True on non-finite portfolio return")
    if info["collapse_reason"] is None:
        raise AssertionError("collapse_reason must be set when collapse_flag is True")


def _test_sum_to_one(returns: pd.DataFrame, vol: pd.DataFrame, action: np.ndarray) -> None:
    cfg = _build_cfg(returns, vol, eta_mode="fixed", rebalance_eta=0.1)
    result = _run_one_step(cfg, action)
    w_exec_exact = result["w_exec_exact"]
    _assert_close("sum(w_exec)", float(np.sum(w_exec_exact)), 1.0, atol=1e-6)


def main() -> None:
    returns, vol = _make_base_frames()
    action = np.array([1.5, -0.3, 0.4, -1.2], dtype=np.float32)

    _test_legacy_replay_invariance(returns, vol, action)
    _test_reward_immutability(returns, vol, action)
    _test_info_key_completeness(returns, vol, action)
    _test_convex_execution_update(returns, vol, action)
    _test_l1_turnover(returns, vol, action)
    _test_cost_split(returns, vol, action)
    _test_misalignment_definition(returns, vol, action)
    _test_eta_modes(returns, vol, action)
    _test_collapse_gate(returns, vol, action)
    _test_sum_to_one(returns, vol, action)

    print("ALL CHECKS PASSED — Commit 1 SAFE")


if __name__ == "__main__":
    main()
