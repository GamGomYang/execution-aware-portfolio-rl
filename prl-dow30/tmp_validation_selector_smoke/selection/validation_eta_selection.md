# Validation Eta Selection Report

- root: prl-dow30/tmp_validation_selector_smoke
- baseline_eta: 1.0
- positive_kappas: 0.0005, 0.001
- relative_threshold: 0.95
- selected_eta: 0.082

## Rule

Select the largest eta whose validation score is within the configured fraction of the best score.
The score is the mean of per-kappa median `sharpe_net_lin` over positive transaction-cost regimes.

## Summary

| eta | n_pos_kappa | n_pairs_vs_eta1 | score_mean_median_sharpe_pos_kappa | score_mean_median_delta_sharpe_vs_eta1_pos_kappa | median_turnover_exec_pos_kappa_mean | qualifies | selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 2 | 4 | 0.700000 | 0.000000 | 0.300000 | False | False |
| 0.082 | 2 | 4 | 0.960000 | 0.260000 | 0.120000 | True | True |
| 0.05 | 2 | 4 | 0.980000 | 0.280000 | 0.100000 | True | False |
