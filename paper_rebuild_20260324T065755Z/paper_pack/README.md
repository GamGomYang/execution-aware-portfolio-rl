# Control Eta Validation-First Pack

- run_root: outputs/paper_rebuild_20260324T065755Z
- train_root: outputs/paper_rebuild_20260324T065755Z/train_control
- validation_root: outputs/paper_rebuild_20260324T065755Z/validation_eta
- final_root: outputs/paper_rebuild_20260324T065755Z/final_eta
- baselines_root: outputs/paper_rebuild_20260324T065755Z/external_baselines
- selected_eta_json: outputs/paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.json
- notes:
  - eta grid fixed a priori
  - validation first, then eta selection, then held-out test evaluation
  - test results are not used for eta selection
  - heuristic baselines are matched on window, kappa, annualization, rf, and executed-path metrics
