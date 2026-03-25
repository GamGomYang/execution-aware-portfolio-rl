# Validation-First Paper Protocol

- selected_eta: 0.5
- baseline_eta: 1.0
- eta grid fixed a priori.
- eta selected on validation only.
- test used only for final held-out evaluation of the selected operating point.
- heuristic baselines matched on window, kappa, annualization, rf, and executed-path metrics.

## Table Layout

- Validation table: frontier plus selection report.
- Test table A: selected eta vs immediate-execution baseline with paired dispersion, bootstrap CI, sign test, and Wilcoxon reporting.
- Test table B: selected eta vs external heuristic baselines.
- Diagnostic table: turnover, tracking, and trace-based gap summaries at the selected eta.
