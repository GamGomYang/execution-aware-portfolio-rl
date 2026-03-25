# Execution-Aware Portfolio Reinforcement Learning
## A Two-Stage Decomposition with Execution Timescale Control

Execution-aware RL framework for **multi-asset portfolio management** with a paper-audited **fixed 27-name large-cap U.S. equity snapshot**.

Some repository paths retain legacy `prl-dow30` naming, but the locked paper rebuild itself uses the fixed 27-name universe documented in `paper.tex` and `data/processed_u27`.

Most portfolio-RL formulations implicitly assume the policy's target weights are the same as what gets traded:

```
w_target = w_exec
```

This repository **separates** the two:

- **Target portfolio (`w_target`)**: policy output
- **Executed portfolio (`w_exec`)**: weights actually applied to the portfolio path

This separation enables:
- **Execution-consistent transaction cost accounting**
- **Execution-timescale control** via a single parameter `eta`
- **Turnover-Tracking-Performance frontier** analysis
- Stabilization under weak/overtrading regimes

---

## Core Idea: Two-Stage Control

### Stage 1: Target Policy
The policy outputs portfolio logits and produces target weights:

```
w_target,t = softmax(logits_t)
```

### Stage 2: Execution Layer (Timescale Control)
Executed weights follow a controlled update:

```
w_exec,t = (1 - eta_t) * w_exec,t-1 + eta_t * w_target,t
```

Only `w_exec` is used for realized portfolio dynamics and evaluation.

**Return**

`r_t = dot(w_exec,t, arithmetic_returns_t)`

**Turnover (L1)**

`TO_target = || w_exec,t-1 - w_target,t ||_1`  
`TO_exec = || w_exec,t-1 - w_exec,t ||_1`

**Cost**

`cost_exec = kappa * TO_exec`  
`cost_target = kappa * TO_target`

> **Note:** Throughout this repo, `prev_weights` refers to `w_exec,t-1`.

For deeper protocol details and definitions, see `docs/spec.md`.

---

## Paper Rebuild Snapshot

The locked paper rebuild is a **frozen-policy execution study**, not a retraining comparison.

- Universe: fixed 27-name large-cap U.S. equity snapshot
- Train / validation / test splits: `2010--2021`, `2022--2023`, `2024--2025`
- Effective realized windows after 30-day rolling features:
  - validation: `2022-02-15` to `2023-12-29`
  - held-out test: `2024-02-14` to `2025-12-31`
- Locked execution grid: `eta in {1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02}`
- Validation-selected operating point in the corrected full-window rebuild: `eta = 0.5`

Held-out selected-vs-immediate (`eta=0.5` vs `eta=1.0`) summary:

- Median executed turnover falls from `0.02200` to `0.01095`
- Paired median net Sharpe improves by `+0.0105` at `kappa = 0.0005`
- Paired median net Sharpe improves by `+0.0213` at `kappa = 0.001`
- At `kappa = 0`, evidence is negligible rather than strongly favorable

Matched heuristic baseline summary:

- Stronger than inverse-volatility risk parity and minimum-variance across reported cost regimes
- Mixed against daily-rebalanced equal weight and long-only mean-variance
- Does not beat buy-and-hold equal weight on this held-out window

Interpretation:

- The main claim is about **cost-aligned execution control**
- The selected operating point looks most turnover-efficient relative to **higher-churn heuristic comparators**
- The paper does **not** claim universal superiority of execution-aware retraining

Primary paper artifacts:

- `paper.tex`
- `paper.pdf`
- `paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.md`
- `paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.md`
- `paper_rebuild_20260324T065755Z/external_baselines/report.md`
- `fig_frontier.png`, `fig_misalignment.png`, `fig_seed_scatter.png`

## Broader Repository Experiments

The repository also contains broader exploratory sweeps, including fixed-eta frontiers and adaptive execution schedules. Those experiments are useful for development, but the locked paper claims should be read from the paper rebuild artifacts above rather than from older exploratory runs.

---

## Repository Structure

```
prl-dow30/
- prl/
  - data.py
  - features.py
  - envs.py
  - prl.py
  - sb3_prl_sac.py
  - train.py
  - eval.py
  - metrics.py
- scripts/
  - run_train.py
  - run_eval.py
  - run_all.py
  - run_matrix.py
  - build_reports.py
  - sanity_checks.py
- configs/
  - default.yaml
  - paper.yaml
  - main_experiment.yaml
  - eta_sweep.yaml
  - rule_vol.yaml
- outputs/
- reports/
- docs/spec.md
```

## Validation & Reproducibility

Step-wise recomputation checks (max absolute error):

- `w_exec`: `1e-9`
- `turnover`: `1e-8`
- `reward`: `1e-10`
- `cost (kappa=0)`: `0.0`

Paired-delta exactness: `~1e-16`  
Signature hashes are unique across eta / rule-vol configurations.

---

## Quickstart

### Install

```
pip install -r requirements.txt
```

### Build Cache (Online)

```
python scripts/build_cache.py --config configs/paper.yaml
```

Paper mode is designed to run without external downloads during training/evaluation.

### Train

```
python scripts/run_train.py \
  --config configs/default.yaml \
  --model-type prl \
  --seed 0
```

### Evaluate

```
python scripts/run_eval.py \
  --config configs/default.yaml \
  --model-type prl \
  --seed 0
```

### Run Experiment Suites

```
python scripts/run_matrix.py --config configs/main_experiment.yaml
python scripts/run_matrix.py --config configs/eta_sweep.yaml
python scripts/run_matrix.py --config configs/rule_vol.yaml
```

---

## Citation / Reference

If you use this codebase in research, please cite the accompanying paper (coming soon) and link to this repository.
