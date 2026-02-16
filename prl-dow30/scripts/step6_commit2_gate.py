from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.envs import Dow30PortfolioEnv, EnvConfig
from prl.eval import run_backtest_episode_detailed, trace_dict_to_frame


ATOL_STRICT = 1e-12
ATOL_GATE = 1e-10
ATOL_TRACK_ZERO = 1e-7


class DummyModel:
    def __init__(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1:
            raise ValueError("action must be 1-D")
        self._action = action.reshape(1, -1)

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002 - SB3-compatible signature
        return self._action, None


def _assert_allclose(name: str, left: np.ndarray | pd.Series, right: np.ndarray | pd.Series, atol: float) -> None:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        raise AssertionError(f"{name} shape mismatch: {left_arr.shape} vs {right_arr.shape}")
    if not np.allclose(left_arr, right_arr, atol=atol, rtol=0.0, equal_nan=True):
        diff = np.abs(left_arr - right_arr)
        max_diff = float(np.nanmax(diff))
        raise AssertionError(f"{name} mismatch: max_diff={max_diff}, atol={atol}")


def _make_frames(*, n_steps: int = 24, n_assets: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="B")
    rng = np.random.default_rng(17)
    arithmetic_returns = rng.normal(loc=0.001, scale=0.01, size=(n_steps, n_assets)).astype(np.float32)
    arithmetic_returns = np.clip(arithmetic_returns, -0.2, 0.2)
    log_returns = np.log1p(arithmetic_returns)
    vol = rng.uniform(low=0.05, high=0.30, size=(n_steps, n_assets)).astype(np.float32)

    returns = pd.DataFrame(log_returns, index=dates, columns=[f"A{i}" for i in range(n_assets)])
    volatility = pd.DataFrame(vol, index=dates, columns=returns.columns)
    return returns, volatility


def _build_cfg(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    with_eta_mode: bool = True,
    eta_mode: str = "legacy",
    rebalance_eta: float | None = None,
    window_size: int = 3,
    transaction_cost: float = 0.001,
    rule_vol_a: float = 1.0,
    eta_clip_min: float = 0.02,
    eta_clip_max: float = 0.5,
) -> EnvConfig:
    kwargs: dict[str, Any] = dict(
        returns=returns,
        volatility=volatility,
        window_size=window_size,
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


def _run_eval_df(
    cfg: EnvConfig,
    action: np.ndarray,
    *,
    eval_id: str = "step6c2",
    run_id: str = "gate",
    model_type: str = "dummy",
    seed: int = 0,
):
    env = DummyVecEnv([lambda: Dow30PortfolioEnv(cfg)])
    model = DummyModel(action)
    metrics, trace = run_backtest_episode_detailed(model, env)
    df = trace_dict_to_frame(trace, eval_id=eval_id, run_id=run_id, model_type=model_type, seed=seed)
    return metrics, trace, df


def test_a_reward_immutability(df: pd.DataFrame) -> None:
    _assert_allclose(
        "A/net_return_lin == portfolio_return - cost",
        df["net_return_lin"],
        df["portfolio_return"] - df["cost"],
        atol=ATOL_STRICT,
    )
    _assert_allclose(
        "A/log_return_net == log_return_gross - cost",
        df["log_return_net"],
        df["log_return_gross"] - df["cost"],
        atol=ATOL_STRICT,
    )


def test_b_misalignment_target_lin(df: pd.DataFrame) -> None:
    _assert_allclose(
        "B/net_return_lin_target == portfolio_return - cost_target",
        df["net_return_lin_target"],
        df["portfolio_return"] - df["cost_target"],
        atol=ATOL_STRICT,
    )


def test_c_misalignment_target_log(df: pd.DataFrame) -> None:
    if "log_return_net_target" not in df.columns:
        raise AssertionError("C/missing column: log_return_net_target")
    _assert_allclose(
        "C/log_return_net_target == log_return_gross - cost_target",
        df["log_return_net_target"],
        df["log_return_gross"] - df["cost_target"],
        atol=ATOL_STRICT,
    )


def test_d_equity(df: pd.DataFrame) -> None:
    expected = np.cumprod(1.0 + df["net_return_lin_target"].fillna(0.0).to_numpy(dtype=np.float64))
    _assert_allclose("D/equity_net_lin_target", df["equity_net_lin_target"], expected, atol=ATOL_STRICT)
    if "equity_net_lin" not in df.columns:
        raise AssertionError("D/missing column: equity_net_lin")
    if "equity_net_exp" not in df.columns:
        raise AssertionError("D/missing column: equity_net_exp")


def test_e_trace_columns(df: pd.DataFrame) -> None:
    required_cols = (
        "cost_target",
        "net_return_lin_target",
        "eta_t",
        "lambda_t",
        "tracking_error_l2",
        "collapse_flag",
        "collapse_reason",
        "equity_net_lin_target",
    )
    for col in required_cols:
        if col not in df.columns:
            raise AssertionError(f"E/missing required column: {col}")


def test_f_eta_mode_propagation(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    action: np.ndarray,
) -> None:
    _, _, legacy_df = _run_eval_df(
        _build_cfg(returns, volatility, eta_mode="legacy", rebalance_eta=None),
        action,
        run_id="legacy",
    )
    _assert_allclose("F/legacy eta_t", legacy_df["eta_t"], np.ones(len(legacy_df)), atol=ATOL_STRICT)

    _, _, none_df = _run_eval_df(
        _build_cfg(returns, volatility, eta_mode="none", rebalance_eta=None),
        action,
        run_id="none",
    )
    _assert_allclose("F/none eta_t", none_df["eta_t"], np.ones(len(none_df)), atol=ATOL_STRICT)
    tracking_abs = np.abs(none_df["tracking_error_l2"].to_numpy(dtype=np.float64))
    if np.nanmax(tracking_abs) > ATOL_TRACK_ZERO:
        raise AssertionError(f"F/none tracking_error_l2 should be ~0, got max={np.nanmax(tracking_abs)}")

    rebalance_eta = 0.1
    _, _, fixed_df = _run_eval_df(
        _build_cfg(returns, volatility, eta_mode="fixed", rebalance_eta=rebalance_eta),
        action,
        run_id="fixed",
    )
    _assert_allclose(
        "F/fixed eta_t",
        fixed_df["eta_t"],
        np.full(len(fixed_df), rebalance_eta, dtype=np.float64),
        atol=ATOL_STRICT,
    )

    eta_clip_min = 0.02
    eta_clip_max = 0.5
    _, _, rule_df = _run_eval_df(
        _build_cfg(
            returns,
            volatility,
            eta_mode="rule_vol",
            rebalance_eta=None,
            rule_vol_a=1.0,
            eta_clip_min=eta_clip_min,
            eta_clip_max=eta_clip_max,
        ),
        action,
        run_id="rule_vol",
    )
    eta = rule_df["eta_t"].to_numpy(dtype=np.float64)
    if np.isnan(eta).any():
        raise AssertionError("F/rule_vol eta_t contains NaN")
    if not np.all((eta >= eta_clip_min) & (eta <= eta_clip_max)):
        raise AssertionError("F/rule_vol eta_t out of clip bounds")


def test_g_collapse_propagation(
    base_returns: pd.DataFrame,
    base_vol: pd.DataFrame,
    action: np.ndarray,
) -> None:
    returns = base_returns.copy()
    window_size = 3
    returns.iloc[window_size, 0] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, df = _run_eval_df(
            _build_cfg(returns, base_vol, eta_mode="legacy", rebalance_eta=None, window_size=window_size),
            action,
            run_id="collapse",
        )
    if not df["collapse_flag"].astype(bool).any():
        raise AssertionError("G/collapse_flag should contain at least one True")
    collapsed = df[df["collapse_flag"].astype(bool)]
    if collapsed["collapse_reason"].isna().any():
        raise AssertionError("G/collapse rows must include collapse_reason")


def test_h_legacy_replay_invariance(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    action: np.ndarray,
) -> None:
    metrics_default, _, df_default = _run_eval_df(
        _build_cfg(returns, volatility, with_eta_mode=False, rebalance_eta=None),
        action,
        run_id="legacy_default",
    )
    metrics_legacy, _, df_legacy = _run_eval_df(
        _build_cfg(returns, volatility, with_eta_mode=True, eta_mode="legacy", rebalance_eta=None),
        action,
        run_id="legacy_explicit",
    )

    _assert_allclose("H/reward", df_default["reward"], df_legacy["reward"], atol=ATOL_GATE)
    _assert_allclose("H/net_return_lin", df_default["net_return_lin"], df_legacy["net_return_lin"], atol=ATOL_GATE)
    _assert_allclose("H/equity_net_lin", df_default["equity_net_lin"], df_legacy["equity_net_lin"], atol=ATOL_GATE)
    if not np.isclose(float(metrics_default.sharpe), float(metrics_legacy.sharpe), atol=ATOL_GATE, rtol=0.0):
        raise AssertionError(
            f"H/sharpe mismatch: default={metrics_default.sharpe}, legacy={metrics_legacy.sharpe}, atol={ATOL_GATE}"
        )


def main() -> None:
    returns, volatility = _make_frames()
    action = np.array([1.2, -0.4, 0.7, -1.0], dtype=np.float32)

    _, _, fixed_df = _run_eval_df(
        _build_cfg(returns, volatility, eta_mode="fixed", rebalance_eta=0.1),
        action,
        run_id="fixed_main",
    )

    test_a_reward_immutability(fixed_df)
    test_b_misalignment_target_lin(fixed_df)
    test_c_misalignment_target_log(fixed_df)
    test_d_equity(fixed_df)
    test_e_trace_columns(fixed_df)

    test_f_eta_mode_propagation(returns, volatility, action)
    test_g_collapse_propagation(returns, volatility, action)
    test_h_legacy_replay_invariance(returns, volatility, action)

    print("COMMIT-2 VERIFIED — TRACE LAYER SAFE")


if __name__ == "__main__":
    main()
