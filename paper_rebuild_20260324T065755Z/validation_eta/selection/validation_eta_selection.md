# Validation Eta Selection Report

- root: outputs/paper_rebuild_20260324T065755Z/validation_eta
- baseline_eta: 1.0
- positive_kappas: 0.0005, 0.001
- relative_threshold: 0.95
- selected_eta: 0.5

## Rule

Select the largest eta whose validation score is within the configured fraction of the best score.
The score is the mean of per-kappa median `sharpe_net_lin` over positive transaction-cost regimes.

## Summary

| eta | n_pos_kappa | n_pairs_vs_eta1 | score_mean_median_sharpe_pos_kappa | score_mean_median_delta_sharpe_vs_eta1_pos_kappa | median_turnover_exec_pos_kappa_mean | qualifies | selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 2 | 20 | 0.493267 | 0.000000 | 0.021603 | False | False |
| 0.5 | 2 | 20 | 0.509072 | 0.014093 | 0.010870 | True | True |
| 0.2 | 2 | 20 | 0.513589 | 0.022900 | 0.005012 | True | False |
| 0.1 | 2 | 20 | 0.517044 | 0.026777 | 0.002714 | True | False |
| 0.082 | 2 | 20 | 0.517905 | 0.027684 | 0.002263 | True | False |
| 0.05 | 2 | 20 | 0.519583 | 0.029190 | 0.001427 | True | False |
| 0.02 | 2 | 20 | 0.521258 | 0.030935 | 0.000600 | True | False |
