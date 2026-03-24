# Control Eta Rebuild Spec

## Paper Lock

- The eta grid is fixed a priori: `1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02`.
- Eta is selected using the validation window only.
- The validation score is the mean, over positive-cost kappas, of the seed-median `sharpe_net_lin`.
- The selected eta is the largest eta within the configured relative threshold of the best validation score.
- The implemented threshold is sign-aware: `best_score * 0.95` when the best score is positive, `best_score / 0.95` when the best score is negative.
- The test window is used only for final held-out evaluation of the selected operating point and is not used for eta selection.
- External heuristic baselines must use the same test window, the same transaction-cost definition `kappa * executed_turnover`, the same Sharpe annualization `sqrt(252)`, the same `rf=0` assumption, and the same executed-path net-linear metric definitions.

## Claim Scope

- Keep the main claim at the frozen-policy level.
- Safe claims:
  - execution-aware formulation is cost-aligned
  - the validation-selected eta operating point remains valid on the held-out test window
  - the selected operating point is competitive with matched heuristic baselines
- Do not upgrade the main claim to general execution-aware retraining superiority until retraining evidence is complete.

## Table Layout

- Validation table: eta frontier and selection report.
- Test table A: selected eta vs immediate-execution baseline `eta=1.0`, with paired dispersion and exact sign test.
- Test table B: selected eta vs external heuristic baselines.
- Diagnostic table: selected-eta turnover, turnover ratio, tracking, and trace-based absolute gap summaries.

## Run Commands

### Full rebuild

```bash
cd /workspace/execution-aware-portfolio-rl
PYTHON_CMD='uv run --with-requirements requirements.txt python' \
RUN_ROOT="outputs/paper_rebuild_$(date -u +%Y%m%dT%H%M%SZ)" \
SAC_TOTAL_TIMESTEPS=0 \
RUN_TRAIN=1 \
RUN_VALIDATION=1 \
RUN_SELECT=1 \
RUN_FINAL=1 \
RUN_BASELINES=1 \
RUN_PACK=1 \
FINAL_MODE=selected_plus_baseline \
prl-dow30/scripts/run_u27_control_eta_validation_first.sh
```

### Resume after training

```bash
cd /workspace/execution-aware-portfolio-rl
PYTHON_CMD='uv run --with-requirements requirements.txt python' \
RUN_ROOT="outputs/paper_rebuild_<existing_timestamp>" \
RUN_TRAIN=0 \
RUN_VALIDATION=1 \
RUN_SELECT=1 \
RUN_FINAL=1 \
RUN_BASELINES=1 \
RUN_PACK=1 \
FINAL_MODE=selected_plus_baseline \
prl-dow30/scripts/run_u27_control_eta_validation_first.sh
```

## Required Outputs

- `train_control/`
- `validation_eta/aggregate.csv`
- `validation_eta/selection/validation_eta_selection.json`
- `final_eta/aggregate.csv`
- `external_baselines/aggregate.csv`
- `paper_pack/tables/validation_selection.csv`
- `paper_pack/tables/test_selected_vs_eta1.csv`
- `paper_pack/tables/test_selected_vs_external_baselines.csv`
- `paper_pack/tables/diagnostic_selected_eta.csv`
- `paper_pack/stats/selected_eta_vs_eta1_stats.csv`
- `paper_pack/diagnostics/diagnostic_selected_eta_v2.csv`
- `paper_pack/figures/fig_selected_trace.png`
- `paper_pack/figures/fig_validation_frontier.png`
- `paper_pack/protocol_lock.md`
