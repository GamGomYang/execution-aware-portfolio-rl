# Step 8: External Baseline Expansion (2026-03-24)

## Goal

Add one to two stronger heuristic comparators without changing the locked validation/test protocol.

## Implemented

- Added `minimum_variance` baseline.
- Added `mean_variance_long_only` baseline.
- Kept all heuristics under the same held-out test window, same `kappa * executed_turnover` cost definition, same `sqrt(252)` Sharpe annualization, same `rf=0`, and same executed-path metric definitions.

## Heuristic specification

- window: `2024-01-01` to `2025-12-31`
- lookback: `252`
- minimum warm-up: `30`
- mean-variance risk aversion: `10.0`
- allocation domain: long-only simplex

## Files changed

- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/baselines.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/eval.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/run_external_heuristic_baselines.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/build_control_eta_validation_first_tables.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/diagnostics.py`
- `/workspace/execution-aware-portfolio-rl/02.17.01.tex`

## Locked-run outputs updated

- `/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/external_baselines/aggregate.csv`
- `/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/external_baselines/protocol.json`
- `/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv`

## Key results

Selected `eta=0.2` keeps higher net Sharpe than all five heuristics in all three cost regimes.

### Versus minimum-variance

- `kappa=0.0`: `delta_sharpe = +0.1596`, `delta_cagr = +0.0162`, `delta_turnover = -0.04180`
- `kappa=5e-4`: `delta_sharpe = +0.2067`, `delta_cagr = +0.0221`, `delta_turnover = -0.04180`
- `kappa=1e-3`: `delta_sharpe = +0.2542`, `delta_cagr = +0.0280`, `delta_turnover = -0.04180`

### Versus mean-variance (long-only)

- `kappa=0.0`: `delta_sharpe = +0.0679`, `delta_cagr = -0.0315`, `delta_turnover = -0.09434`
- `kappa=5e-4`: `delta_sharpe = +0.1495`, `delta_cagr = -0.0177`, `delta_turnover = -0.09434`
- `kappa=1e-3`: `delta_sharpe = +0.2315`, `delta_cagr = -0.0040`, `delta_turnover = -0.09434`

## Interpretation lock

- The expanded baseline set strengthens the paper's safe claim on net Sharpe and turnover efficiency.
- It does not justify a blanket absolute-return superiority claim.
- The mean-variance baseline is especially useful as a contrast case: higher turnover can preserve or chase raw growth while degrading cost-aware net Sharpe.

## Verification

- `python3 -m py_compile` passed for the changed modules.
- Targeted pytest pass:
  - `tests/test_baselines.py`
  - `tests/test_fast_skip_baselines_changes_row_count.py`
  - `tests/test_paper_gate_artifacts.py`
  - `tests/test_regime_metrics_has_three_regimes.py`
  - `tests/test_eval_period_alignment.py`
  - `tests/test_ids_consistent_across_metrics.py`
- Locked-run external baseline artifacts and paper-pack tables were regenerated.
