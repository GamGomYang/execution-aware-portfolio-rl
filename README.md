# Forecasting Actionability Report Card

Code and reusable results for **[From Forecasting Leaderboards to Deployment Decisions: A Fail-Closed Certification Protocol](https://arxiv.org/abs/2606.24996)**.

This repository asks a deployment-facing model-selection question: when a forecasting metric selects a winner, is that winner also supported as top-1 advice after the same fixed forecast-to-decision interface and friction model are applied?

<p align="center">
  <img src="results/figures/fig1_fixed_interface_inversion.png" alt="Fixed-interface evaluation can change model selection" width="860">
</p>

## What This Audits

Forecasting benchmarks usually rank candidates by predictive quality. Deployment-facing use can add a fixed interface before utility is measured:

- threshold alerts
- hysteresis rules
- budgeted top-k actions
- residual-warning screens
- replenishment rules

The report card keeps forecast-side metrics intact and adds a fail-closed check for whether the selected winner remains deployment-actionable under the specified interface. Cases that are not sufficiently supported should stay diagnostic rather than being promoted as certified reversals.

## Repository Layout

- `results/tables/`: reusable CSV outputs from the actionability checks.
- `results/figures/`: retained PNG figures for inspecting the main result patterns.
- `scripts/forecast_eval/`: experiment and event-micro support scripts.
- `scripts/reporting/`: lightweight report-card summary utilities.

## Result Snapshot

| Task | Interface | Friction | Forecast-side winner | Deployed-side winner | Agreement | Suboptimal cases |
| --- | --- | ---: | --- | --- | ---: | ---: |
| Synthetic | zero-friction anchor | 0.00 | Naive last | Naive last | 1.00 | 0/20 |
| Event warning | threshold tau=0.55 | 0.50 | Reactive sharp | Calibrated | 0.31 | 69/100 |
| Event warning | threshold tau=0.55 | 1.00 | Reactive sharp | Smoother | 0.01 | 99/100 |
| Budgeted traffic alert | budget k=249 | 0.50 | Reactive short | Smoother | 0.00 | 100/100 |
| Budgeted traffic alert | budget k=249 | 1.00 | Reactive short | Smoother | 0.00 | 100/100 |
| Inventory replenishment | replenishment | 1.00 | Small MLP | MA(7) | 0.01 | 99/100 |

Event warning and Traffic-Hourly provide the main prediction-to-decision checks. Inventory is retained as operational corroboration.

## Useful Commands

```powershell
python scripts\reporting\compute_share_intervals.py
python -m py_compile (Get-ChildItem -Recurse -Filter *.py).FullName
```

The first command writes `results/tables/share_interval_audit.csv`. The second command checks that retained Python files parse.

## Scope

This is a reporting diagnostic, not a new forecasting benchmark suite, forecaster, or universal deployed metric. Deployed utility is specific to the chosen interface, simulator, and friction model.

## Paper

This repository accompanies the arXiv paper:

**From Forecasting Leaderboards to Deployment Decisions: A Fail-Closed Certification Protocol**  
Geumyoung Kim, 2026  
[arXiv:2606.24996](https://arxiv.org/abs/2606.24996) | [DOI: 10.48550/arXiv.2606.24996](https://doi.org/10.48550/arXiv.2606.24996)

## Citation

```bibtex
@misc{kim2026forecastingleaderboardsdeploymentdecisions,
  title={From Forecasting Leaderboards to Deployment Decisions: A Fail-Closed Certification Protocol},
  author={Geumyoung Kim},
  year={2026},
  eprint={2606.24996},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2606.24996},
  url={https://arxiv.org/abs/2606.24996}
}
```
