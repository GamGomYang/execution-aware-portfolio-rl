# Step 6: Statistical Enrichment Pass (2026-03-24)

## Goal

Strengthen the held-out selected-vs-immediate comparison with uncertainty reporting beyond IQR and exact sign tests.

## Implemented

- Added two-sided Wilcoxon signed-rank reporting for paired `delta_sharpe_net_lin` and paired `delta_cagr`.
- Added percentile-bootstrap 95% confidence intervals for the paired median `delta_sharpe_net_lin` and paired median `delta_cagr`.
- Kept the existing paired-seed framing, exact one-sided sign test, and IQR reporting.

## Files changed

- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/build_selected_eta_stats.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/build_control_eta_validation_first_tables.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/tests/test_build_selected_eta_stats.py`
- `/workspace/execution-aware-portfolio-rl/02.17.01.tex`

## Key held-out results for selected eta = 0.2 vs eta = 1.0

Source CSV:

- `/workspace/execution-aware-portfolio-rl/prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv`

### Net Sharpe

- `kappa=0.0`: paired median `delta_sharpe_net_lin = +0.00541`, exact sign test `p = 0.37695`, Wilcoxon `p = 0.27539`, bootstrap 95% CI `[-0.00568, 0.01614]`
- `kappa=5e-4`: paired median `delta_sharpe_net_lin = +0.02218`, exact sign test `p = 0.00098`, Wilcoxon `p = 0.00195`, bootstrap 95% CI `[0.01281, 0.03682]`
- `kappa=1e-3`: paired median `delta_sharpe_net_lin = +0.03896`, exact sign test `p = 0.00098`, Wilcoxon `p = 0.00195`, bootstrap 95% CI `[0.03156, 0.05798]`

### CAGR

- `kappa=0.0`: paired median `delta_cagr = +0.00055`, Wilcoxon `p = 0.37500`, bootstrap 95% CI `[-0.00083, 0.00189]`
- `kappa=5e-4`: paired median `delta_cagr = +0.00263`, Wilcoxon `p = 0.00195`, bootstrap 95% CI `[0.00140, 0.00440]`
- `kappa=1e-3`: paired median `delta_cagr = +0.00470`, Wilcoxon `p = 0.00195`, bootstrap 95% CI `[0.00380, 0.00699]`

## Interpretation lock

- Positive-cost evidence is materially stronger after the statistical pass: directionality, paired rank test, and bootstrap CI all agree.
- `kappa=0` remains weak/modest evidence only: both Wilcoxon and bootstrap CI fail to support a strong claim.
- This strengthens the cost-aligned execution-control claim without expanding the paper into a general training-superiority claim.

## Verification

- `python3 -m py_compile` passed for the updated scripts and tests.
- `pytest` passed for:
  - `tests/test_build_selected_eta_stats.py`
  - `tests/test_analyze_wilcoxon_skip.py`
  - `tests/test_bootstrap_ci_shape.py`
- Rebuilt stats and paper-pack tables for the locked run root:
  - `/workspace/execution-aware-portfolio-rl/prl-dow30/outputs/paper_rebuild_20260324T065755Z`
