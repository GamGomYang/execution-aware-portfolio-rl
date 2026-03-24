# Control Eta Validation-First Checklist

- run_root: prl-dow30/outputs/paper_rebuild_smoke_materialize

## Implementation Checklist

| status | description | path |
| --- | --- | --- |
| PASS | Run root created | prl-dow30/outputs/paper_rebuild_smoke_materialize |
| PASS | train_control directory created | prl-dow30/outputs/paper_rebuild_smoke_materialize/train_control |
| PASS | validation_eta directory created | prl-dow30/outputs/paper_rebuild_smoke_materialize/validation_eta |
| PASS | final_eta directory created | prl-dow30/outputs/paper_rebuild_smoke_materialize/final_eta |
| PASS | external_baselines directory created | prl-dow30/outputs/paper_rebuild_smoke_materialize/external_baselines |
| PASS | paper_pack directory created | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack |
| PASS | Frozen training config emitted | prl-dow30/outputs/paper_rebuild_smoke_materialize/configs/snapshot_control.yaml |
| PASS | Validation eval config emitted | prl-dow30/outputs/paper_rebuild_smoke_materialize/configs/validation_eta.yaml |
| PASS | Final eval config emitted | prl-dow30/outputs/paper_rebuild_smoke_materialize/configs/final_eta.yaml |
| PASS | Materialization metadata emitted | prl-dow30/outputs/paper_rebuild_smoke_materialize/configs/materialization_meta.json |

## Execution Checklist

| status | description | path |
| --- | --- | --- |
| FAIL | Validation aggregate report written | prl-dow30/outputs/paper_rebuild_smoke_materialize/validation_eta/aggregate.csv |
| FAIL | Validation paired report written | prl-dow30/outputs/paper_rebuild_smoke_materialize/validation_eta/paired_delta.csv |
| SKIP | Validation frontier figure written | prl-dow30/outputs/paper_rebuild_smoke_materialize/validation_eta/fig_frontier.png |
| FAIL | Validation eta selection JSON written | prl-dow30/outputs/paper_rebuild_smoke_materialize/validation_eta/selection/validation_eta_selection.json |
| FAIL | Validation eta selection markdown written | prl-dow30/outputs/paper_rebuild_smoke_materialize/validation_eta/selection/validation_eta_selection.md |
| SKIP | Final/test aggregate report written | prl-dow30/outputs/paper_rebuild_smoke_materialize/final_eta/aggregate.csv |
| SKIP | Final/test paired report written | prl-dow30/outputs/paper_rebuild_smoke_materialize/final_eta/paired_delta.csv |
| PASS | External baseline aggregate report written | prl-dow30/outputs/paper_rebuild_smoke_materialize/external_baselines/aggregate.csv |
| PASS | External baseline protocol JSON written | prl-dow30/outputs/paper_rebuild_smoke_materialize/external_baselines/protocol.json |
| SKIP | Paper pack README written | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack/README.md |
| SKIP | Pack protocol lock markdown written | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack/protocol_lock.md |
| SKIP | Validation selection table written | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack/tables/validation_selection.csv |
| SKIP | Test selected-vs-eta1 table written | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack/tables/test_selected_vs_eta1.csv |
| SKIP | Test selected-vs-external-baselines table written | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack/tables/test_selected_vs_external_baselines.csv |
| SKIP | Diagnostic selected-eta table written | prl-dow30/outputs/paper_rebuild_smoke_materialize/paper_pack/tables/diagnostic_selected_eta.csv |
