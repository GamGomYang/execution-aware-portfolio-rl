# Experiment Report

## Dataset & Setup
- Universe: Dow30 (30 equities), yfinance Adjusted Close daily bars cached via `scripts/build_cache.py` for reproducibility. Business-day alignment covers 2010-01-01 to 2025-12-31.
- Train range: 2010-01-01 to 2021-12-31. Test/backtest range: 2022-01-01 to 2025-12-31.
- Rolling windows: L=30 for returns, Lv=30 for volatility. Transaction cost coefficient c_tc=0.0005 (turnover = sum |w_t - w_{t-1}|).
- Anti-lookahead enforced by using w_{t-1} * r_t in rewards; actions converted to weights via softmax inside the environment.

## Hyperparameters
### PRL (volatility-only scheduler)
- alpha0=0.2, beta=1.0, lambdav=2.0, bias=0.0, alpha_min=0.01, alpha_max=1.0.
### SAC (shared for baseline & PRL-SAC)
- learning_rate=0.0003, batch_size=256, gamma=0.99, tau=0.005.
- buffer_size=5000, total_timesteps=2000, ent_coef=0.2. Eval frequency set to 500 steps.

## Results
### Per-seed metrics (2010-2021 train, 2022-2025 backtest)
| Model | Seed | Return | Sharpe | MDD | Turnover |
|---|---|---|---|---|---|
| baseline | 0 | 0.0061 | 1.9059 | -0.000680 | 0.000051 |
| baseline | 1 | 0.0062 | 1.9166 | -0.000694 | 0.000073 |
| baseline | 2 | 0.0061 | 1.8999 | -0.000709 | 0.000055 |
| prl | 0 | 0.0060 | 1.8530 | -0.000669 | 0.000292 |
| prl | 1 | 0.0066 | 1.9849 | -0.000703 | 0.000195 |
| prl | 2 | 0.0062 | 1.8989 | -0.000732 | 0.000350 |

### Mean +/- Std (per model)
| Model | Return | Sharpe | MDD | Turnover |
|---|---|---|---|---|
| baseline | 0.0062 +/- 0.0001 | 1.9075 +/- 0.0069 | -0.000695 +/- 0.000012 | 0.000060 +/- 0.000009 |
| prl | 0.0063 +/- 0.0002 | 1.9123 +/- 0.0547 | -0.000701 +/- 0.000026 | 0.000279 +/- 0.000064 |

## Issues & Next Steps
- Training budget was trimmed to 2k steps per run so experiments could finish on this workstation; performance numbers therefore reflect very early training. Increase total_timesteps and replay buffer size once access to more compute is available.
- PRL-SAC achieved slightly higher expected return/Sharpe but incurred higher turnover under the L1 definition. A turnover-aware penalty schedule and hyperparameter sweep (beta, lambdav) should be explored.
- Monitor stability across seeds: Sharpe std for PRL (~0.055) is non-trivial even with short runs; consider adding more seeds or longer evaluation windows.
