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
