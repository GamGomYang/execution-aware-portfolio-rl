## PRL Dow30 v1 Design Freeze

### Scope
- Assets: Dow30 components only, strictly 30 risky assets (no cash proxy).  
- Timeline: Training 2010-01-01 to 2021-12-31, backtest 2022-01-01 to 2025-12-31.  
- Libraries: Base RL agent will rely on Stable-Baselines3 SAC, but VecNormalize and any update/loss scaling tricks are explicitly excluded in v1.  
- Anti-lookahead: every reward at index `t` must consume realized returns using previous weights `w_{t-1}`.

### Data Pipeline
1. **Source**: yfinance-only Adjusted Close. `build_cache.py` always downloads a fresh snapshot (no stooq, no synthetic), runs quality checks, and writes `data/processed/prices.parquet`, `returns.parquet`, and `data_manifest.json` plus `outputs/reports/data_quality_summary.csv`.
2. **Cache-only modes**: when `paper_mode=true` and `require_cache=true` (or `offline=true`), train/eval/run_all load only the processed cache; if missing they raise `CACHE_MISSING` instructing `build_cache.py`.
3. **Calendar alignment**: combine all tickers on the union of trading days, forward/backward fill once, drop remaining NaNs, and persist aligned matrices.
4. **Split**: maintain explicit metadata for train vs. test periods (2010-2021 vs. 2022-2025) and hashes inside the manifest.
5. **Returns**: compute daily log returns after alignment; rolling windows of length `L` are applied after the split to avoid leakage.

### Environment Specification
- **Observation**: concatenate `(returns_window(L, N) -> flattened L*N vector)` + `(vol_vector of size N)` + `(previous_weights of size N)`. The resulting observation dimension is `obs_dim = L*N + 2*N`.
- **Action space**: gym `Box(low=-1, high=1, shape=(N,))`. Actions represent unconstrained logits `z`.
- **Softmax inside env**: `env.step()` must convert `z` to portfolio weights via `softmax(z)` so that `w >= 0` and `sum(w) = 1`. No other location may apply the softmax.
- **State progression**: the environment stores `w_{t-1}` after each step for use in rewards and in the next observation’s `previous_weights` slice.

### Reward Definition
- Primary reward: `r_t = log(1 + sum(w_{t-1} * return_t))` with a small `rp` clamp (e.g., min value) to keep the log argument positive even in stress scenarios.
- Transaction cost penalty: `turnover_t = sum(|w_t - w_{t-1}|)` and reward subtracts `c_tc * turnover_t`.
- Anti-lookahead enforcement: all calculations for `r_t` use `return_t` (current realized returns) but `w_{t-1}` (previous action) to avoid peeking ahead.

### Volatility & PRL Logic
1. **Rolling volatility**: for each asset `i`, `vol_i(t)` is the rolling standard deviation of returns over a window of length `Lv`.  
2. **Aggregate scalar**: `V_t = mean_i vol_i(t)` for downstream use.  
3. **Normalization**: any z-scoring of volatility (`Vz_t`) is computed using statistics (mean, std) from the **training** subset only; the same parameters are reused in evaluation/backtests.  
4. **PRL modulation**:  
   - Policy regularization logit: `P_t = sigmoid(lambdav * Vz_t + bias)`.  
   - Learning-rate scaling term: `alpha_t = alpha0 * (1 + beta * P_t)` with clamping `[alpha_min, alpha_max]`.  
   - No separate loss or optimizer scaling beyond this alpha schedule in v1.  
5. **Alpha slices**: both `alpha_obs` and `alpha_next` derive their volatility component from the observation slice `vol_slice = obs[L*N : L*N + N]`, ensuring local volatility drives the adaptive alpha signal.

### Evaluation & Metrics Notes
- Training: 2010-01-01 through 2021-12-31, with validation/backtest strictly 2022-01-01 through 2025-12-31.  
- Performance metrics (Sharpe, max drawdown, turnover) consume the processed data and environment outputs described above; turnover is the full L1 distance between consecutive weight vectors.  
- Scripts `scripts/run_train.py`, `scripts/run_eval.py`, and `scripts/run_all.py` orchestrate workflows; `build_cache.py` must be run online once to freeze the reproducible snapshot used in paper/offline modes.
