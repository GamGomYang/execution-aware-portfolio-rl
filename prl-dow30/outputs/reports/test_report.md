# Test Report

- pytest -q : 9 passed in 7.31s (see log above).
- Anti-lookahead behavior verified via tests/test_env.py::test_reward_uses_previous_weights_with_turnover_penalty (reward relied on w_{t-1} and differed from using w_t).
- Fixes performed during Stage 3:
  * Added prl/__init__.py plus tests/conftest.py for import stability.
  * Extended prl/metrics.py and scripts/run_eval.py to include Sharpe + max drawdown reporting.
  * Exposed PRLAlphaScheduler.prl_probability and ensured numpy is imported in prl/data.py.
