# Execution-Aware Portfolio Reinforcement Learning  
## A Two-Stage Decomposition with Execution Timescale Control

Execution-aware RL framework for **multi-asset portfolio management** on the **Dow30** universe.

Most portfolio-RL formulations implicitly assume the policy’s target weights are the same as what gets traded:

w_target = w_exec


This repository **separates** the two:

- **Target portfolio (w_target)**: policy output  
- **Executed portfolio (w_exec)**: weights actually applied to the portfolio path

This separation enables:
- **Execution-consistent transaction cost accounting**
- **Execution-timescale control** via a single parameter `eta`
- **Turnover–tracking–performance frontier** analysis
- Stabilization under weak/overtrading regimes

---

## Core Idea: Two-Stage Control

### Stage 1 — Target Policy
The policy outputs portfolio logits and produces target weights:

w_target,t = softmax(logits_t)


### Stage 2 — Execution Layer (Timescale Control)
Executed weights follow a controlled update:

w_exec,t = (1 - eta_t) * w_exec,t-1 + eta_t * w_target,t


Only `w_exec` is used for realized portfolio dynamics and evaluation:

- **Return**
r_t = dot(w_exec,t, arithmetic_returns_t)


- **Turnover (L1)**
TO_target = || w_exec,t-1 - w_target,t ||_1
TO_exec = || w_exec,t-1 - w_exec,t ||_1


- **Cost**
cost_exec = kappa * TO_exec
cost_target = kappa * TO_target


> **Note:** Throughout this repo, `prev_weights` refers to `w_exec,t-1`.

For deeper protocol details and definitions, see `docs/spec.md`.

---

## Key Findings (Dow30)

### 1) Baseline vs Controlled Execution
- **Baseline:** `eta = 1.0` (fully reactive execution)
- **Controlled:** `eta = 0.10`
- Transaction cost levels: `kappa ∈ {0.0, 0.0005, 0.001}`
- Seeds: `{0, 1, 2}`

Observed in our runs:
- When `kappa > 0`, controlled execution typically improves **Sharpe**
- When `kappa = 0`, improvements are not guaranteed but were often observed
- Collapse rate: `0.0` (under the reported settings)

Artifacts:
- `aggregate.csv`, `paired_delta.csv`, `collapse_report.md`
- `fig_frontier.png`, `fig_misalignment.png`

### 2) Execution Timescale Frontier (eta Sweep)
eta grid:

```text
[1.0, 0.5, 0.2, 0.1, 0.05,
0.02, 0.01, 0.005, 0.002,
0.001, 0.0005, 0.0002]
Across kappa ∈ {0, 0.0005, 0.001}, we observe monotonic behavior:

eta ↓ => TO_exec ↓

eta ↓ => tracking error ↑

eta ↓ => misalignment gap ↑

Within the tested range, Sharpe often improves as eta decreases; plateau was not observed in our sweep.

3) Adaptive Execution (Volatility-Based eta)
Rule:

eta_t = clip(a / vol_t, eta_min, eta_max)
Parameters:

a ∈ {0.5, 1.0, 2.0}

clip range: [0.02, 1.0]

Findings:

Larger a reduces effective eta

High eta -> higher turnover -> lower Sharpe (in our settings)

Configuration signatures cleanly separate eta / rule-vol regimes

Repository Structure
prl-dow30/
├── prl/
│   ├── data.py
│   ├── features.py
│   ├── envs.py
│   ├── prl.py
│   ├── sb3_prl_sac.py
│   ├── train.py
│   ├── eval.py
│   └── metrics.py
├── scripts/
│   ├── run_train.py
│   ├── run_eval.py
│   ├── run_all.py
│   ├── run_matrix.py
│   ├── build_reports.py
│   └── sanity_checks.py
├── configs/
│   ├── default.yaml
│   ├── paper.yaml
│   ├── main_experiment.yaml
│   ├── eta_sweep.yaml
│   └── rule_vol.yaml
├── outputs/
├── reports/
└── docs/spec.md
Validation & Reproducibility
Step-wise recomputation checks (max absolute error):

w_exec: 1e-9

turnover: 1e-8

reward: 1e-10

cost (kappa=0): 0.0

Paired-delta exactness: ~1e-16
Signature hashes are unique across eta / rule-vol configurations.

Quickstart
Install
pip install -r requirements.txt
Build Cache (Online)
python scripts/build_cache.py --config configs/paper.yaml
Paper mode is designed to run without external downloads during training/evaluation.

Train
python scripts/run_train.py \
  --config configs/default.yaml \
  --model-type prl \
  --seed 0
Evaluate
python scripts/run_eval.py \
  --config configs/default.yaml \
  --model-type prl \
  --seed 0
Run Experiment Suites
python scripts/run_matrix.py --config configs/main_experiment.yaml
python scripts/run_matrix.py --config configs/eta_sweep.yaml
python scripts/run_matrix.py --config configs/rule_vol.yaml


Citation / Reference
If you use this codebase in research, please cite the accompanying paper (coming soon) and link to this repository.

