# Control Eta Validation-First Checklist

- run_root: outputs/paper_rebuild_20260324T065755Z

## Implementation Checklist

| status | description | path |
| --- | --- | --- |
| PASS | Run root created | outputs/paper_rebuild_20260324T065755Z |
| PASS | train_control directory created | outputs/paper_rebuild_20260324T065755Z/train_control |
| PASS | validation_eta directory created | outputs/paper_rebuild_20260324T065755Z/validation_eta |
| PASS | final_eta directory created | outputs/paper_rebuild_20260324T065755Z/final_eta |
| PASS | external_baselines directory created | outputs/paper_rebuild_20260324T065755Z/external_baselines |
| PASS | paper_pack directory created | outputs/paper_rebuild_20260324T065755Z/paper_pack |
| PASS | Frozen training config emitted | outputs/paper_rebuild_20260324T065755Z/configs/snapshot_control.yaml |
| PASS | Validation eval config emitted | outputs/paper_rebuild_20260324T065755Z/configs/validation_eta.yaml |
| PASS | Final eval config emitted | outputs/paper_rebuild_20260324T065755Z/configs/final_eta.yaml |
| PASS | Materialization metadata emitted | outputs/paper_rebuild_20260324T065755Z/configs/materialization_meta.json |

## Execution Checklist

| status | description | path |
| --- | --- | --- |
| PASS | Validation aggregate report written | outputs/paper_rebuild_20260324T065755Z/validation_eta/aggregate.csv |
| PASS | Validation paired report written | outputs/paper_rebuild_20260324T065755Z/validation_eta/paired_delta.csv |
| PASS | Validation frontier figure written | outputs/paper_rebuild_20260324T065755Z/validation_eta/fig_frontier.png |
| PASS | Validation eta selection JSON written | outputs/paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.json |
| PASS | Validation eta selection markdown written | outputs/paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.md |
| PASS | Final/test aggregate report written | outputs/paper_rebuild_20260324T065755Z/final_eta/aggregate.csv |
| PASS | Final/test paired report written | outputs/paper_rebuild_20260324T065755Z/final_eta/paired_delta.csv |
| PASS | External baseline aggregate report written | outputs/paper_rebuild_20260324T065755Z/external_baselines/aggregate.csv |
| PASS | External baseline protocol JSON written | outputs/paper_rebuild_20260324T065755Z/external_baselines/protocol.json |
| PASS | Paper pack README written | outputs/paper_rebuild_20260324T065755Z/paper_pack/README.md |
| PASS | Pack protocol lock markdown written | outputs/paper_rebuild_20260324T065755Z/paper_pack/protocol_lock.md |
| PASS | Validation selection table written | outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/validation_selection.csv |
| PASS | Test selected-vs-eta1 table written | outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_eta1.csv |
| PASS | Selected-vs-eta1 stats CSV written | outputs/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv |
| PASS | Selected-vs-eta1 seedwise deltas CSV written | outputs/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_seedwise_deltas.csv |
| PASS | Test selected-vs-external-baselines table written | outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv |
| PASS | Diagnostic selected-eta table written | outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/diagnostic_selected_eta.csv |
| PASS | Diagnostic selected-eta v2 table written | outputs/paper_rebuild_20260324T065755Z/paper_pack/diagnostics/diagnostic_selected_eta_v2.csv |
| PASS | Representative selected-eta seed metadata written | outputs/paper_rebuild_20260324T065755Z/paper_pack/diagnostics/representative_seed_metrics.json |
| PASS | Selected-trace figure written | outputs/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_selected_trace.png |
| PASS | Validation frontier figure written | outputs/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_validation_frontier.png |
| PASS | Seed scatter figure written | outputs/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_seed_scatter.png |
