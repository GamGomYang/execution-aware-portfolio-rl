# Control Eta Validation-First Checklist

- run_root: prl-dow30/outputs/paper_rebuild_smoke_full

## Implementation Checklist

| status | description | path |
| --- | --- | --- |
| PASS | Run root created | prl-dow30/outputs/paper_rebuild_smoke_full |
| PASS | train_control directory created | prl-dow30/outputs/paper_rebuild_smoke_full/train_control |
| PASS | validation_eta directory created | prl-dow30/outputs/paper_rebuild_smoke_full/validation_eta |
| PASS | final_eta directory created | prl-dow30/outputs/paper_rebuild_smoke_full/final_eta |
| PASS | external_baselines directory created | prl-dow30/outputs/paper_rebuild_smoke_full/external_baselines |
| PASS | paper_pack directory created | prl-dow30/outputs/paper_rebuild_smoke_full/paper_pack |
| PASS | Frozen training config emitted | prl-dow30/outputs/paper_rebuild_smoke_full/configs/snapshot_control.yaml |
| PASS | Validation eval config emitted | prl-dow30/outputs/paper_rebuild_smoke_full/configs/validation_eta.yaml |
| PASS | Final eval config emitted | prl-dow30/outputs/paper_rebuild_smoke_full/configs/final_eta.yaml |
| PASS | Materialization metadata emitted | prl-dow30/outputs/paper_rebuild_smoke_full/configs/materialization_meta.json |

## Execution Checklist

| status | description | path |
| --- | --- | --- |
| FAIL | Validation aggregate report written | prl-dow30/outputs/paper_rebuild_smoke_full/validation_eta/aggregate.csv |
| FAIL | Validation paired report written | prl-dow30/outputs/paper_rebuild_smoke_full/validation_eta/paired_delta.csv |
| SKIP | Validation frontier figure written | prl-dow30/outputs/paper_rebuild_smoke_full/validation_eta/fig_frontier.png |
| FAIL | Validation eta selection JSON written | prl-dow30/outputs/paper_rebuild_smoke_full/validation_eta/selection/validation_eta_selection.json |
| FAIL | Validation eta selection markdown written | prl-dow30/outputs/paper_rebuild_smoke_full/validation_eta/selection/validation_eta_selection.md |
| SKIP | Final/test aggregate report written | prl-dow30/outputs/paper_rebuild_smoke_full/final_eta/aggregate.csv |
| SKIP | Final/test paired report written | prl-dow30/outputs/paper_rebuild_smoke_full/final_eta/paired_delta.csv |
| SKIP | Paper pack README written | prl-dow30/outputs/paper_rebuild_smoke_full/paper_pack/README.md |
