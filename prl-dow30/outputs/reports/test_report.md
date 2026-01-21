## Test Report (Stage 2.1 Paper-Gate)

- Commands run:
  - `pytest -q` (42 passed)
  - `PYTHONPATH=. python3 scripts/build_cache.py --config configs/paper.yaml` → SUCCESS (27 tickers, 4023 rows)
  - `PYTHONPATH=. python3 scripts/run_all.py --config configs/paper_gate.yaml --seeds 0 --offline` → SUCCESS (cache-only; no download logs)

- GATE-G0 build_cache (paper.yaml):
  - manifest: `universe_policy=availability_filtered`, `N_assets_final=27`, `min_assets=20`
  - kept_tickers: `AAPL, AMGN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GS, HD, HON, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PG, UNH, V, VZ, WMT`
  - outputs: `data/processed/prices.parquet`, `data/processed/returns.parquet`, `data/processed/data_manifest.json`, `outputs/reports/data_quality_summary.csv`

- GATE-G1 paper_gate offline (paper_gate.yaml):
  - cache-only evidence: `Loading processed cache from data/processed` logs only; no yfinance fetch lines
  - artifacts:
    - `outputs/reports/metrics.csv`
    - `outputs/reports/summary.csv`
    - `outputs/models/baseline_seed0_final.zip`
    - `outputs/models/prl_seed0_final.zip`
    - `outputs/logs/baseline_seed0_train_log.csv`
    - `outputs/logs/prl_seed0_train_log.csv`
    - `outputs/reports/run_metadata_*.json`
  - turnover (metrics.csv):
    - baseline avg_turnover=0.803929, total_turnover=781.419201
    - prl avg_turnover=1.122940, total_turnover=1091.497964

- Exit criteria status:
  - pytest: PASS
  - build_cache (paper.yaml): PASS (Option B manifest populated)
  - paper_gate offline: PASS (metrics/summary/models/logs/metadata present; turnover non-zero)

## Test Report (Stage 3 Step 2 Smoke)

- Commands run:
  - `pytest -q` (45 passed)
  - `python scripts/run_all.py --config configs/smoke.yaml --seeds 0 --offline` → FAILED (`python` not found)
  - `PYTHONPATH=. python3 scripts/run_all.py --config configs/smoke.yaml --seeds 0 --offline` → SUCCESS (cache-only; no download logs)

- Smoke outputs:
  - `outputs/reports/metrics.csv` present
  - metrics.csv rows:
    - baseline avg_turnover=0.2139756441, total_turnover=11.3407091383
    - prl avg_turnover=0.2311949085, total_turnover=12.2533301513
  - `daily_rebalanced_equal_weight` row: NOT FOUND
  - `buy_and_hold_equal_weight` row: NOT FOUND

## Test Report (Stage 3 Step 3 Baselines)

- Commands run:
  - `pytest -q` (47 passed)
  - `PYTHONPATH=. python scripts/run_all.py --config configs/smoke.yaml --seeds 0 --offline` → SUCCESS (cache-only; no download logs)

- Smoke outputs:
  - `outputs/reports/metrics.csv` present
  - `outputs/reports/summary.csv` present
  - `outputs/reports/regime_metrics.csv` present
  - metrics.csv rows:
    - baseline_sac avg_turnover=0.2139756441, total_turnover=11.3407091383
    - prl_sac avg_turnover=0.2311949085, total_turnover=12.2533301513
    - buy_and_hold_equal_weight avg_turnover=0.0, total_turnover=0.0
    - daily_rebalanced_equal_weight avg_turnover=0.0092573079, total_turnover=0.5832103986
    - inverse_vol_risk_parity avg_turnover=0.0789984104, total_turnover=4.9768998547
