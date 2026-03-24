# External Heuristic Baselines

- config: outputs/paper_rebuild_smoke_materialize/configs/final_eta.yaml
- eval_window: 2024-01-01 to 2025-12-31
- kappas: 0.0, 0.0005
- matched definitions:
  same window, same kappa, same Sharpe annualization sqrt(252), rf=0, same executed-path net-linear metrics.

| eval_window | eval_start | eval_end | kappa | strategy | seed | sharpe_net_lin | cumulative_return_net_lin | cagr | maxdd | avg_turnover_exec | total_turnover_exec | sharpe_annualization | risk_free_rate | cost_definition | primary_metric | trace_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2024-01-01 | 2025-12-31 | 0.0 | buy_and_hold_equal_weight | 0 | 1.2002591401978349 | 0.35573169385763026 | 0.1650650351288936 | -0.15988055619499209 | 0.0 | 0.0 | sqrt(252) | 0.0 | kappa * executed_turnover | executed_path_sharpe_net_lin | outputs/paper_rebuild_smoke_materialize/external_baselines/traces/kappa_0_buy_and_hold_equal_weight.parquet |
| test | 2024-01-01 | 2025-12-31 | 0.0 | daily_rebalanced_equal_weight | 0 | 1.1674592500701553 | 0.3530542120060276 | 0.1639094192276509 | -0.15627262748893878 | 0.009851861488385114 | 4.945634467169327 | sqrt(252) | 0.0 | kappa * executed_turnover | executed_path_sharpe_net_lin | outputs/paper_rebuild_smoke_materialize/external_baselines/traces/kappa_0_daily_rebalanced_equal_weight.parquet |
| test | 2024-01-01 | 2025-12-31 | 0.0 | inverse_vol_risk_parity | 0 | 1.109770288251139 | 0.30753835790123407 | 0.14408739304301088 | -0.151595589968749 | 0.032904681631226736 | 16.51815017887582 | sqrt(252) | 0.0 | kappa * executed_turnover | executed_path_sharpe_net_lin | outputs/paper_rebuild_smoke_materialize/external_baselines/traces/kappa_0_inverse_vol_risk_parity.parquet |
| test | 2024-01-01 | 2025-12-31 | 0.0005 | buy_and_hold_equal_weight | 0 | 1.2002591401978349 | 0.35573169385763026 | 0.1650650351288936 | -0.15988055619499209 | 0.0 | 0.0 | sqrt(252) | 0.0 | kappa * executed_turnover | executed_path_sharpe_net_lin | outputs/paper_rebuild_smoke_materialize/external_baselines/traces/kappa_0.0005_buy_and_hold_equal_weight.parquet |
| test | 2024-01-01 | 2025-12-31 | 0.0005 | daily_rebalanced_equal_weight | 0 | 1.1584673626446993 | 0.34971373042077847 | 0.1624660510639082 | -0.15646405451287837 | 0.009851861488385114 | 4.945634467169327 | sqrt(252) | 0.0 | kappa * executed_turnover | executed_path_sharpe_net_lin | outputs/paper_rebuild_smoke_materialize/external_baselines/traces/kappa_0.0005_daily_rebalanced_equal_weight.parquet |
| test | 2024-01-01 | 2025-12-31 | 0.0005 | inverse_vol_risk_parity | 0 | 1.0774550960436042 | 0.2967839715919296 | 0.13935392528203727 | -0.1521447030193792 | 0.032904681631226736 | 16.51815017887582 | sqrt(252) | 0.0 | kappa * executed_turnover | executed_path_sharpe_net_lin | outputs/paper_rebuild_smoke_materialize/external_baselines/traces/kappa_0.0005_inverse_vol_risk_parity.parquet |
