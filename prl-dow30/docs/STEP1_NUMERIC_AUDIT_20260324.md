# Step 1 Numeric Audit

## Scope

- Manuscript: `02.17.01.tex`
- Locked run: `prl-dow30/outputs/paper_rebuild_20260324T065755Z`
- Audit target:
  - table/prose numeric consistency
  - `eta=0.2` operating-point consistency
  - paired-median interpretation in the held-out selected-vs-baseline table
  - diagnostic-table `kappa=0` anomaly

## Checks Completed

- Re-verified that the held-out selected-vs-eta1 table still matches:
  - `selected_eta_vs_eta1_stats.csv`
  - the manuscript values for Sharpe, CAGR, turnover, win-rate, and sign-test `p`
- Re-verified that the validation-selected operating point is consistently `eta=0.2`.
- Re-verified that the external-baseline table still matches the generated held-out baseline summary.
- Added a manuscript note clarifying that marginal arm medians and paired medians are different objects.

## Main Finding

The original `kappa=0` diagnostic row did **not** collapse to zero because of rounding or print precision.
It collapsed because the stored target trace reused the executed gross return and only changed the cost deduction:

- old behavior:
  - `net_return_lin_target = portfolio_return_exec - cost_target`
- implication:
  - at `kappa=0`, return-gap and equity-gap columns were structurally zero even when tracking error was non-zero
  - at `kappa>0`, the reported target-path gaps mainly reflected cost differences rather than true target-vs-executed path divergence

## Fix Applied

- Patched target-trace construction in:
  - `prl/envs.py`
  - `prl/eval.py`
- Added regression test:
  - `tests/test_target_trace_uses_target_weights.py`
- Re-ran:
  - held-out final evaluation
  - paper-pack stats/diagnostics/tables/figures
  - on the existing locked run root, without retraining

## Regression Status

- `python3 -m py_compile` passed for the touched Python files.
- Targeted regression tests passed:
  - `tests/test_target_trace_uses_target_weights.py`
  - `tests/test_cost_uses_turnover_exec.py`
  - `tests/test_env_turnover_rebalance.py`

## Impact on Paper Results

- Core held-out selected-vs-baseline metrics did **not** change.
  - These metrics are defined on the executed path and were already correct.
- The trace-based diagnostic table changed.

Updated diagnostic medians for `eta=0.2`:

- `kappa=0`
  - mean absolute return gap: `4.99e-05`
  - final equity gap: `7.89e-04`
  - max daily gap: `3.25e-04`
- `kappa=5e-4`
  - mean absolute return gap: `5.07e-05`
  - final equity gap: `0.00265`
  - max daily gap: `3.31e-04`
- `kappa=1e-3`
  - mean absolute return gap: `5.26e-05`
  - final equity gap: `0.00518`
  - max daily gap: `3.46e-04`

## Manuscript Updates Required

- Keep the paired-vs-marginal median note in the held-out selected-vs-baseline table.
- Use the corrected diagnostic values in the diagnostic paragraph and table.
- Describe gap columns as executed-path vs hypothetical target-weight-path quantities.
