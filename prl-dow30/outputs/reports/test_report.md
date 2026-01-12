## Test Report (Stage 3 Gate)

- Commands run:
  - `pytest -q` (47 passed)
  - `PYTHONPATH=. python3 scripts/build_cache.py --config configs/paper.yaml` → SUCCESS (27 tickers, 4023 rows)
  - `PYTHONPATH=. python3 scripts/run_all.py --config configs/smoke.yaml --seeds 0 --offline` → SUCCESS (cache-only; no download logs)
  - `PYTHONPATH=. python3 scripts/run_all.py --config configs/paper_gate.yaml --seeds 0 --offline` → SUCCESS (cache-only; no download logs)

- GATE-G0 build_cache (paper.yaml):
  - manifest: `universe_policy=availability_filtered`, `N_assets_final=27`, `min_assets=20`
  - kept_tickers: `AAPL, AMGN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GS, HD, HON, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PG, UNH, V, VZ, WMT`
  - outputs: `data/processed/prices.parquet`, `data/processed/returns.parquet`, `data/processed/data_manifest.json`, `outputs/reports/data_quality_summary.csv`

- G3-1 offline smoke (smoke.yaml):
  - cache-only evidence: `Loading processed cache from data/processed` logs only; no yfinance fetch lines
  - artifacts: `outputs/reports/metrics.csv`, `outputs/reports/summary.csv`, `outputs/reports/regime_metrics.csv`
  - model_type rows: baseline, prl, buy_and_hold_equal_weight, daily_rebalanced_equal_weight, inverse_vol_risk_parity
  - turnover (metrics.csv):
    - baseline avg_turnover=0.242282, total_turnover=12.840965
    - prl avg_turnover=0.230922, total_turnover=12.238845
    - inverse_vol_risk_parity avg_turnover=0.078001, total_turnover=4.134045

- G3-2 offline paper_gate (paper_gate.yaml):
  - cache-only evidence: `Loading processed cache from data/processed` logs only; no yfinance fetch lines
  - artifacts:
    - `outputs/reports/metrics.csv`
    - `outputs/reports/summary.csv`
    - `outputs/reports/regime_metrics.csv`
    - `outputs/models/baseline_seed0_final.zip`
    - `outputs/models/prl_seed0_final.zip`
    - `outputs/logs/baseline_seed0_train_log.csv`
    - `outputs/logs/prl_seed0_train_log.csv`
    - `outputs/reports/run_metadata_*.json`
  - turnover (metrics.csv):
    - baseline avg_turnover=0.803929, total_turnover=781.419201
    - prl avg_turnover=1.122940, total_turnover=1091.497964
    - inverse_vol_risk_parity avg_turnover=0.028740, total_turnover=27.934875

- Stage 3 acceptance status:
  - pytest: PASS
  - baselines/regime metrics: PASS (metrics/summary/regime files + 5 model_type rows)
  - cache-only offline gates: PASS (no download logs, artifacts present)
