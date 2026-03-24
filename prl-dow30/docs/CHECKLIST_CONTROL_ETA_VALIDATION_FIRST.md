# Control Eta Validation-First Checklist

## Implementation Checklist

- [x] Add a validation-first orchestration script that separates validation selection from final test evaluation.
- [x] Standardize the output tree under `train_control`, `validation_eta`, `final_eta`, `external_baselines`, and `paper_pack`.
- [x] Extend config materialization so the frozen control snapshot can emit both validation and final eval configs.
- [x] Add an automatic `selected eta` report so eta choice is reproducible and not post-hoc.
- [x] Add an execution checker that writes markdown/json PASS/FAIL checklists for generated artifacts.
- [x] Keep Python executable selection configurable so the workflow can run in different environments.
- [x] Allow dependency-injected execution via `PYTHON_CMD`, so the workflow can run with `uv run --with-requirements requirements.txt python`.
- [x] Add matched external heuristic baseline evaluation under the same window, kappa, annualization, rf, and executed-path metric definitions.
- [x] Build validation/test paper tables around selection and held-out evaluation rather than only raw frontier dumps.
- [x] Lock the paper protocol in a reproducible spec document and pack-level protocol summary.
- [x] Add paired selected-vs-eta1 statistics with IQR, win-rate, and exact sign test outputs.
- [x] Add trace-based misalignment diagnostics with absolute/pathwise gap summaries.
- [x] Add paper-oriented figures for the selected trace, validation frontier, and seed-wise deltas.

## Execution Checklist

- [ ] `run_root` was created under `outputs/paper_rebuild_<timestamp>`.
- [ ] Frozen training config exists.
- [ ] Validation eval config exists.
- [ ] Final eval config exists.
- [ ] Seed `0..9` frozen-control retraining completed.
- [ ] Validation eta frontier finished.
- [ ] Validation `aggregate.csv` and `paired_delta.csv` were written.
- [ ] Validation eta selection report was written.
- [ ] Final/test eta evaluation finished.
- [ ] Final/test `aggregate.csv` and `paired_delta.csv` were written.
- [ ] External heuristic baseline `aggregate.csv` was written under `external_baselines`.
- [ ] Paper pack contains `validation_selection`, `test_selected_vs_eta1`, `test_selected_vs_external_baselines`, and `diagnostic_selected_eta` tables.
- [ ] Paper pack contains selected-vs-eta1 stats and seed-wise delta outputs.
- [ ] Paper pack contains `diagnostic_selected_eta_v2` and representative-seed metadata.
- [ ] Paper pack contains paper figures for the selected trace, validation frontier, and seed scatter.
- [ ] Paper pack contains configs, validation reports, final reports, baseline reports, protocol lock, and checklist outputs.

## Notes

- The intended selection rule is: choose the largest eta whose validation score is within a configured fraction of the best score.
- When the best validation score is negative, the implemented threshold uses the same relative rule in signed form, i.e. the qualifying cutoff is `best_score / relative_threshold`.
- The validation score is computed from positive-cost regimes only.
- External heuristic baselines must use the same window, the same kappa, the same Sharpe annualization, the same rf assumption, and the same executed-path net-linear metric definition.
- In `paper_mode`, training enforces `sac.total_timesteps >= 100000`; use `SAC_TOTAL_TIMESTEPS=0` to keep the frozen default or a value at/above that floor.
