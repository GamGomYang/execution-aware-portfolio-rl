import pandas as pd

from prl.diagnostics import select_signals_by_screening


def test_signal_ic_screening_requires_both_thresholds():
    ic_summary = pd.DataFrame(
        [
            {"signal": "cs_mom_3_12", "tstat": 2.1},
            {"signal": "vol_scaled_mom", "tstat": -2.5},
            {"signal": "residual_mom_beta_neutral", "tstat": 2.0},
            {"signal": "short_term_reversal", "tstat": 4.0},
        ]
    )
    ls_summary = pd.DataFrame(
        [
            {"signal": "cs_mom_3_12", "ls_sharpe": 0.3},
            {"signal": "vol_scaled_mom", "ls_sharpe": -0.1},
            {"signal": "residual_mom_beta_neutral", "ls_sharpe": 0.5},
            {"signal": "short_term_reversal", "ls_sharpe": 0.0},
        ]
    )

    selected = select_signals_by_screening(ic_summary, ls_summary)
    assert selected == ["cs_mom_3_12"]


def test_signal_ic_screening_handles_empty_input():
    selected = select_signals_by_screening(pd.DataFrame(), pd.DataFrame())
    assert selected == []
