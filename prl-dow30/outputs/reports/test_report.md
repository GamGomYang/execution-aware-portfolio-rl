## Test Report (Stage 2.1-B)

- Commands run:
  - `pytest -q` (41 passed)
  - `PYTHONPATH=. python3 scripts/build_cache.py --config configs/paper.yaml` → SUCCESS (27 tickers, 4023 rows)
  - `PYTHONPATH=. python3 scripts/run_all.py --config configs/smoke.yaml --seeds 0 --offline` → SUCCESS
  - `PYTHONPATH=. python3 scripts/run_all.py --config configs/paper.yaml --seeds 0 --offline` → TIMEOUT (180s, 420s; not completed)

- G1 build_cache (paper.yaml):
  - manifest: `universe_policy=availability_filtered`, `N_assets_final=27`, `min_assets=20`, `kept_tickers=27`, `substitutions_used=None`
  - outputs: prices/returns/manifest/quality_summary produced

- G2 smoke offline (smoke.yaml):
  - no download logs observed; cache-only load used
  - `outputs/reports/metrics.csv` rows=2 (baseline/prl), turnover non-zero:
    - baseline avg_turnover=0.212996, total_turnover=11.288813
    - prl avg_turnover=0.209125, total_turnover=11.083638
  - `outputs/models/*_final.zip` created

- G3 paper offline (paper.yaml):
  - run did not finish within 180s or 420s; only partial training progress logged
  - final artifacts not confirmed (no pass)

- Exit criteria status:
  - pytest: PASS
  - G1/G2: PASS
  - G3: NOT COMPLETED (timeout)
  - turnover definition unified & tested: PASS (`test_turnover_definition.py`)
  - logit_scale wiring test: PASS (`test_env_logit_scale_applied.py`)
  - missing_fraction raw gate test: PASS (`test_missing_fraction_gate.py`)
