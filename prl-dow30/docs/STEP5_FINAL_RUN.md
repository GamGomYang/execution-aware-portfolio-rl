# STEP5 Final Run Guide

## Fixed assumptions

- `env.risk_lambda = 0.0`
- `env.random_reset_train = false`
- Mainline execution smoothing is `env.rebalance_eta = 0.10`
- PRL gate(B) is evaluated only for PRL scheduling runs. Baseline PRL-off runs are `SKIP`.
- Turnover primary metric is `avg_turnover_exec` (`avg_turnover_target` is analysis-only).
- Primary performance family is `*_net_exp`.

## Output root convention

```bash
OUT=outputs/exp_runs/step5_final/$(date +%Y%m%d_%H%M%S)
```

## 1) Required main comparison (2 runs, seeds 0..4)

```bash
python3 -m scripts.run_all \
  --config configs/exp/step5/exp_S5_final_baseline_eta010.yaml \
  --model-types prl \
  --seeds 0 1 2 3 4 \
  --output-root $OUT

python3 -m scripts.run_all \
  --config configs/exp/step5/exp_S5_final_prl_eta010.yaml \
  --model-types prl \
  --seeds 0 1 2 3 4 \
  --output-root $OUT
```

## 2) Recommended eta ablation (2 extra runs, seeds 0..4)

```bash
python3 -m scripts.run_all \
  --config configs/exp/step5/exp_S5_ablate_baseline_etaNone.yaml \
  --model-types prl \
  --seeds 0 1 2 3 4 \
  --output-root $OUT

python3 -m scripts.run_all \
  --config configs/exp/step5/exp_S5_ablate_prl_etaNone.yaml \
  --model-types prl \
  --seeds 0 1 2 3 4 \
  --output-root $OUT
```

## 3) Generate paper artifacts

```bash
python3 -m analysis.analyze_step5_final \
  --input-root $OUT \
  --out-dir $OUT/reports/paper/step5
```

## 4) Final gate decision

```bash
python3 -m analysis.step5_final_gate \
  --input-root $OUT
```

## Archive behavior (`run_all` / `run_eval`)

Each run keeps latest files and also appends immutable archives.

- Latest (overwritten):
  - `reports/metrics.csv`
  - `reports/summary.csv`
  - `reports/regime_metrics.csv`
- Archive (accumulated):
  - `reports/archive/metrics_<exp_id>.csv`
  - `reports/archive/summary_<exp_id>.csv`
  - `reports/archive/regime_metrics_<exp_id>.csv`

If the same config is rerun in the same output root, a numeric suffix is appended (for example `__2`) to prevent overwrite.

## Expected Step5 artifacts

Primary output directory: `$OUT/reports/paper/step5`

- `table_main.csv`
- `table_main_robust.tex`
- `robust_stats_summary.csv`
- `robust_delta_prl_minus_base.csv`
- `table_regime.csv`
- `table_ablation.csv` (only when all 4 Step5 configs exist)
- `stats_tests.csv`
- `fig_equity_curve_net_exp.png`
- `fig_drawdown_net_exp.png`
- `fig_turnover_exec_vs_target.png`
- `summary_step5.md`
- `step5_gate_result.json` (written by gate script)
