## PRL Dow30 v1 Design Freeze

### Scope
- Assets: Dow30 components only, strictly 30 risky assets (no cash proxy).  
- Timeline: Training 2010-01-01 to 2021-12-31, backtest 2022-01-01 to 2025-12-31.  
- Libraries: Base RL agent will rely on Stable-Baselines3 SAC, but VecNormalize and any update/loss scaling tricks are explicitly excluded in v1.  
- Anti-lookahead: every reward at index `t` must consume realized returns using previous weights `w_{t-1}`.

### Data Pipeline
1. **Source hierarchy**: pull Adjusted Close daily bars via `yfinance` and cache to `data/raw`. If the network call fails or cache is missing for a symbol, fall back to a user-provided local CSV under `data/raw`.
2. **Calendar alignment**: combine all tickers on the union of trading days, then reindex to the intersection of valid dates across the entire Dow30 universe over the full 2010-2025 interval.
3. **Missing data**: forward fill once, then backward fill once, per symbol. Any rows still containing NaNs after the two-pass fill are dropped entirely.
4. **Split**: persist aligned price matrices into `data/processed` and maintain explicit metadata for train vs. test periods (2010-2021 vs. 2022-2025).
5. **Returns**: compute daily log returns and rolling windows of length `L` for features after split to avoid leakage.

### Environment Specification
- **Observation**: concatenate `(returns_window(L, N) -> flattened L*N vector)` + `(vol_vector of size N)` + `(previous_weights of size N)`. The resulting observation dimension is `obs_dim = L*N + 2*N`.
- **Action space**: gym `Box(low=-1, high=1, shape=(N,))`. Actions represent unconstrained logits `z`.
- **Softmax inside env**: `env.step()` must convert `z` to portfolio weights via `softmax(z)` so that `w >= 0` and `sum(w) = 1`. No other location may apply the softmax.
- **State progression**: the environment stores `w_{t-1}` after each step for use in rewards and in the next observation’s `previous_weights` slice.

### Reward Definition
- Primary reward: `r_t = log(1 + sum(w_{t-1} * return_t))` with a small `rp` clamp (e.g., min value) to keep the log argument positive even in stress scenarios.
- Transaction cost penalty: `turnover_t = 0.5 * sum(|w_t - w_{t-1}|)` (exact formula TBD later) and reward subtracts `c_tc * turnover_t`.
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
- Training: 2010-01-01 through 2021-12-31, with validation/backtest left strictly to 2022-01-01 through 2025-12-31.  
- Performance metrics (Sharpe, max drawdown, etc.) and diagnostic plots will be defined later but must consume the processed data and environment outputs described above.  
- Scripts `scripts/run_train.py`, `scripts/run_eval.py`, and `scripts/run_all.py` will orchestrate workflows once implementation begins; this document freezes interfaces for those future steps.
