# FAST Mode (output isolation)

- Purpose: run gating experiments repeatedly without mixing artifacts. Every run uses its own `--output-root`, and downstream analysis reads `run_index.json` instead of ad-hoc file globs.

## Running
- Example: `python -m scripts.run_all --config configs/paper.yaml --output-root outputs/run_2024w12 --offline`
- The runner creates `<output_root>/{reports,traces,models,logs}` up front; reports stay under `reports/` for compatibility, while traces can be mirrored into `traces/` if needed.
- Train/eval metadata, metrics, and regimes are written under `reports/`; models and training logs are routed to `models/` and `logs/`.

## run_index.json (reports/run_index.json)
- Fixed fields: `exp_name`, `timestamp`, `config_path`, `model_types`, `seeds`, `eval_windows`, `run_ids`, `metrics_path`, `regime_metrics_path`, `reports_dir`, `traces_dir`, `models_dir`, `logs_dir`, `output_root`.
- `run_ids` is order-preserving and deduplicated for the session (includes strategy baselines).
- Use the recorded `metrics_path`/`regime_metrics_path` for all aggregation; avoid mixing legacy outputs by always pointing tooling at the chosen `output_root`.

## Policy
- Do not delete or overwrite previous runs; pick a fresh `--output-root` per gate cycle.
- Any analysis, dashboards, or reports should resolve artifacts via `run_index.json` rather than hardcoded `outputs/reports`.
- Net_exp metrics remain the primary decision signal; gross/net_lin are secondary diagnostics.

## Gate run commands (templates)
- Gate0 smoke (W1): `python -m scripts.run_all --config configs/exp/gate0_smoke_W1.yaml --output-root outputs/exp_runs/gate0_smoke/<ts>`
- Gate1 reference baseline (W1, seeds=1, force_refresh=false): `python -m scripts.run_all --config configs/exp/gate1_reference_baseline_sac_W1.yaml --model-types baseline --output-root outputs/exp_runs/gate1/reference_baseline_sac/<ts>`
- Gate1 screen (A-series knobs): `python -m scripts.run_all --config configs/exp/exp_A1_smooth_a010.yaml --output-root outputs/exp_runs/A1_a010/<ts>`
- Gate2 confirm (W1+W2): `python -m scripts.run_all --config configs/exp/gate2_confirm_W1W2.yaml --output-root outputs/exp_runs/gate2_confirm/<ts>`
- Gate3 risk align: `python -m scripts.run_all --config configs/exp/gate3_riskalign.yaml --output-root outputs/exp_runs/gate3/<ts>`
- Final: `python -m scripts.run_all --config configs/exp/final_confirm.yaml --output-root outputs/exp_runs/final/<ts>`

## Post-run analysis
- Metrics/regime filter by run_index: `python -m scripts.analyze_paper_results --metrics <out>/reports/metrics.csv --regime-metrics <out>/reports/regime_metrics.csv --run-index <out>/reports/run_index.json --output-dir <out>/reports`
- Diagnosis decomposition: `python -m scripts.diagnosis_decomposition --metrics <out>/reports/metrics.csv --regime-metrics <out>/reports/regime_metrics.csv --output-dir <out>/reports`
- Gate1 leaderboard: `python -m scripts.build_gate1_leaderboard --reference-run-index outputs/exp_runs/gate1/reference_baseline_sac/<ts_ref>/reports/run_index.json --candidate-run-indexes "outputs/exp_runs/gate1/*/*/reports/run_index.json" --output-dir outputs/exp_runs/gate1`
