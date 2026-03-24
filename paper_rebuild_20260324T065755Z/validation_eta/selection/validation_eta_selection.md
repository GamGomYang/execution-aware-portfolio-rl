# Validation Eta Selection Report

- root: outputs/paper_rebuild_20260324T065755Z/validation_eta
- baseline_eta: 1.0
- positive_kappas: 0.0005, 0.001
- relative_threshold: 0.95
- selected_eta: 0.2

## Rule

Select the largest eta whose validation score is within the configured fraction of the best score.
The score is the mean of per-kappa median `sharpe_net_lin` over positive transaction-cost regimes.

## Summary

| eta | n_pos_kappa | n_pairs_vs_eta1 | score_mean_median_sharpe_pos_kappa | score_mean_median_delta_sharpe_vs_eta1_pos_kappa | median_turnover_exec_pos_kappa_mean | qualifies | selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 2 | 20 | -0.101469 | 0.000000 | 0.021582 | False | False |
| 0.5 | 2 | 20 | -0.088627 | 0.010610 | 0.010945 | False | False |
| 0.2 | 2 | 20 | -0.079947 | 0.018130 | 0.005071 | True | True |
| 0.1 | 2 | 20 | -0.078257 | 0.020835 | 0.002752 | True | False |
| 0.082 | 2 | 20 | -0.078464 | 0.020650 | 0.002295 | True | False |
| 0.05 | 2 | 20 | -0.079064 | 0.020193 | 0.001447 | True | False |
| 0.02 | 2 | 20 | -0.078811 | 0.020509 | 0.000618 | True | False |
