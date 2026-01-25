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
