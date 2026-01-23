import math

import numpy as np
import pandas as pd
import pytest

from prl.baselines import run_baseline_strategy_detailed
from prl.eval import summarize_regime_metrics, trace_dict_to_frame
from prl.metrics import compute_metrics


def _sample_trace():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    rewards = [math.log(1.01) - 0.001, math.log(1.02) - 0.002, math.log(1.0)]
    portfolio_returns = [0.01, 0.02, 0.0]
    turnovers = [0.1, 0.2, 0.0]
    costs = [0.001, 0.002, 0.0]
    net_returns_exp = [math.exp(r) - 1.0 for r in rewards]
    net_returns_lin = [r - c for r, c in zip(portfolio_returns, costs)]
    trace = {
        "dates": list(dates),
        "rewards": rewards,
        "portfolio_returns": portfolio_returns,
        "turnovers": turnovers,
        "turnover_target_changes": [0.05, 0.05, 0.05],
        "costs": costs,
        "net_returns_exp": net_returns_exp,
        "net_returns_lin": net_returns_lin,
    }
    return dates, trace


def test_compute_metrics_net_populates_fields_and_cost_monotonicity():
    _, trace = _sample_trace()
    metrics = compute_metrics(
        trace["rewards"],
        trace["portfolio_returns"],
        trace["turnovers"],
        net_returns_exp=trace["net_returns_exp"],
        net_returns_lin=trace["net_returns_lin"],
    )
    assert metrics.cumulative_return_net_exp is not None
    assert metrics.cumulative_return_net_lin is not None
    assert metrics.cumulative_return_net_exp <= metrics.cumulative_return
    assert metrics.cumulative_return_net_lin <= metrics.cumulative_return


def test_trace_dict_to_frame_includes_net_columns():
    dates, trace = _sample_trace()
    df = trace_dict_to_frame(trace, eval_id="eval", run_id="run", model_type="baseline_sac", seed=0)
    expected_cols = {"cost", "net_return_exp", "net_return_lin", "equity_gross", "equity_net_exp", "equity_net_lin"}
    assert expected_cols.issubset(df.columns)
    assert len(df) == len(dates)
    assert df["equity_net_exp"].iloc[-1] != df["equity_gross"].iloc[-1]


def test_summarize_regime_metrics_includes_net_fields():
    _, trace = _sample_trace()
    df = trace_dict_to_frame(trace, eval_id="eval", run_id="run", model_type="baseline_sac", seed=0)
    df["regime"] = ["low", "mid", "high"]
    metrics_rows = summarize_regime_metrics(df, include_all=True)
    regimes = {(row["regime"], row["model_type"]) for row in metrics_rows}
    assert ("all", "baseline_sac") in regimes
    for row in metrics_rows:
        assert "cumulative_return_net_exp" in row
        assert "sharpe_net_exp" in row


def test_baseline_strategy_net_vs_gross():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    returns = pd.DataFrame(
        [[0.02, 0.0], [0.0, 0.02], [0.015, 0.005]],
        index=dates,
        columns=["A", "B"],
    )
    volatility = pd.DataFrame(0.1, index=dates, columns=["A", "B"])
    tc = 0.01
    bh_metrics, _ = run_baseline_strategy_detailed(
        returns, volatility, "buy_and_hold_equal_weight", transaction_cost=tc
    )
    reb_metrics, _ = run_baseline_strategy_detailed(
        returns, volatility, "daily_rebalanced_equal_weight", transaction_cost=tc
    )
    assert bh_metrics.cumulative_return_net_exp == pytest.approx(bh_metrics.cumulative_return)
    assert reb_metrics.cumulative_return_net_exp < reb_metrics.cumulative_return


def test_metrics_recomputed_from_trace_matches():
    _, trace = _sample_trace()
    df = trace_dict_to_frame(trace, eval_id="eval", run_id="run", model_type="baseline_sac", seed=0)
    metrics = compute_metrics(
        df["reward"],
        df["portfolio_return"],
        df["turnover"],
        net_returns_exp=df["net_return_exp"],
        net_returns_lin=df["net_return_lin"],
    )
    expected_cumret = float(np.prod(1.0 + df["portfolio_return"]) - 1.0)
    expected_cumret_net = float(np.prod(1.0 + df["net_return_exp"]) - 1.0)
    assert metrics.cumulative_return == pytest.approx(expected_cumret)
    assert metrics.cumulative_return_net_exp == pytest.approx(expected_cumret_net)
