## Test Report

- Commands run:
  - `pytest -q` (27 passed)
  - `PYTHONPATH=. python3 scripts/build_cache.py --config configs/paper.yaml` → SUCCESS (WBA fetched via substitution CVS, DOW via DD; manifest records substitutions)
  - `PYTHONPATH=. python3 - <<...>>` short smoke run (200 steps, offline cache) to get turnover sanity: avg_turnover ≈ 0.0607

- Acceptance checklist:
  1) pytest passed (27/27), covering expressivity, CLI defaults, cache-only guards, per-run metadata.
  2) build_cache (paper config) now succeeds: writes prices/returns/manifest/quality_summary; manifest includes substitutions `{"WBA":"CVS","DOW":"DD"}`.
  3) paper/offline/require_cache enforce cache-only (`cache_only` includes offline|paper|require_cache; load_market_data logs cache-only, raises `CACHE_MISSING` if missing).
  4) Action expressivity verified: `tests/test_action_expressivity.py` (N=30, logit_scale) and turnover uses `sum(|w_t - w_{t-1}|)` across env/metrics/docs.
  5) Model naming & CLI defaults: canonical `outputs/models/{model_type}_seed{seed}_final.zip`; `tests/test_model_naming_and_eval_defaults.py` covers training save + run_eval default resolution.
  6) run_metadata per-run: `outputs/reports/run_metadata_*.json` include run_id/seed/mode/model_type/config_hash/git_commit/python & torch/pandas/yfinance/sb3 versions, data_manifest_hash, artifact paths.
  7) yfinance fallback: session retry, substitution map applied on failures/late starts; `tests/test_yfinance_session_fallback.py` still valid.
