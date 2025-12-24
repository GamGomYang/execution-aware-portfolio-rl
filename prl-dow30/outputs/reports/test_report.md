## Test Report

- Commands:
  - `pytest -q`
- Result: 26 passed
- Acceptance checklist:
  1) pytest passed (26/26).
  2) build_cache (paper config) downloads and creates prices/returns/manifest/quality_summary (see `tests/test_build_cache_downloads.py`; real run recommended before paper).
  3) paper/offline/require_cache paths are cache-only via `cache_only` flag; build_cache always online.
  4) Action expressivity verified (logit_scale softmax, `tests/test_action_expressivity.py`).
  5) Model naming canonical: `outputs/models/{model_type}_seed{seed}_final.zip`; run_eval default uses this.
  6) run_metadata.json present (`outputs/reports/run_metadata.json`) with seed/mode/config hash/git commit/python & torch/pandas/yfinance/sb3 versions plus data_manifest_hash.
  7) yfinance session fallback retries full ticker set and errors on missing columns (`tests/test_yfinance_session_fallback.py`).
