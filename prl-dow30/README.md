# PRL Dow30

Stage 2 delivers an end-to-end Dow30 portfolio reinforcement learning stack that adheres to the fixed requirements in `docs/spec.md`. The pipeline covers yfinance-only data ingestion with a reproducibility cache, volatility-aware feature engineering, a custom gymnasium environment with strict anti-lookahead enforcement, a volatility-only PRL alpha scheduler, SB3 SAC integration (baseline + PRL variants), and CLI scripts for training/evaluation.

## Repository Layout
- `data/raw` – yfinance Adjusted Close downloads (when building the cache).
- `data/processed` – aligned parquet matrices (`prices.parquet`, `returns.parquet`) and `vol_stats.json`.
- `prl/`
  - `data.py` – data download/cache + return-frame helpers.
  - `features.py` – rolling volatility + portfolio scalar stats (train-only mean/std).
  - `envs.py` – `Dow30PortfolioEnv` (`obs = returns_window + vol_vector + prev_weights`, softmax inside `step`, reward uses `w_{t-1}`).
  - `prl.py` – volatility-only `PRLAlphaScheduler`.
  - `sb3_prl_sac.py` – Method-A SAC subclass injecting `alpha_obs` / `alpha_next`.
  - `train.py` / `eval.py` / `metrics.py` – orchestration helpers, evaluation loop, and reporting utilities.
- `scripts/` – CLI entrypoints (`run_train.py`, `run_eval.py`, `run_all.py`).
- `configs/default.yaml` – hyperparameters, data paths, and schedule knobs.
- `docs/spec.md` – frozen design contract from Stage 1.

## Usage
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build reproducibility cache (online)**  
   ```bash
   python scripts/build_cache.py --config configs/paper.yaml
   ```
   This downloads yfinance Adj Close for the full Dow30, runs quality checks, and writes `data/processed` parquet + `data_manifest.json` + `outputs/reports/data_quality_summary.csv`.

3. **Configure hyperparameters**  
   Edit `configs/default.yaml` or switch to `configs/paper.yaml` (paper mode requires the processed cache and never downloads).

4. **Train**  
   ```bash
   python scripts/run_train.py --config configs/default.yaml --model-type baseline --seed 0
   python scripts/run_train.py --config configs/default.yaml --model-type prl --seed 0
   ```
   Models are written to `outputs/models/{model_type}_seed{seed}_final.zip`.

5. **Evaluate / Backtest (2022-2025)**  
   ```bash
   python scripts/run_eval.py --config configs/default.yaml --model-type prl --seed 0
   ```
   Metrics append to `outputs/reports/metrics.csv`.

6. **Batch workflow (seeds 0/1/2 for both baselines)**  
   ```bash
   python scripts/run_all.py --config configs/default.yaml
   ```
   Produces `outputs/reports/summary.csv` while also persisting per-run models/metrics.

Notes:
- Paper mode (`paper.yaml` or `--offline`) is cache-only; if cache files are missing it raises `CACHE_MISSING` and never downloads.
- The environment applies softmax with `logit_scale` inside `env.step`, clamps log arguments for numerical safety, enforces turnover penalties (`sum(|w_t - w_{t-1}|)`), and exposes `portfolio_return` + `turnover` via `info` for downstream reporting.
