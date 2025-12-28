import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_cache as build_cache


def test_build_cache_downloads_in_paper_config(tmp_path, monkeypatch):
    # Patch cwd to isolated tmp
    monkeypatch.chdir(tmp_path)

    calls = {"count": 0}

    def _fake_fetch(tickers, start, end, session_opts=None):
        calls["count"] += 1
        ticker = list(tickers)[0]
        dates = pd.date_range(start=start, end=end, freq="B")
        data = pd.DataFrame({("Adj Close", ticker): 100.0 + pd.RangeIndex(len(dates))}, index=dates)
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data

    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch)
    monkeypatch.setattr("prl.data.DOW30_TICKERS", ("AAA", "BBB"))

    cfg_path = tmp_path / "paper.yaml"
    cfg_path.write_text(
        """
mode: paper
dates:
  train_start: '2020-01-01'
  train_end: '2020-01-10'
  test_start: '2020-01-01'
  test_end: '2020-01-10'
data:
  raw_dir: data/raw
  processed_dir: data/processed
  source: yfinance_only
  universe_policy: availability_filtered
  min_assets: 2
  paper_mode: true
  require_cache: true
  offline: false
  force_refresh: true
  history_tolerance_days: 0
  min_history_days: 5
  ticker_substitutions: {}
  quality_params:
    min_vol_std: 0.0
    min_max_abs_return: 0.0
    max_flat_fraction: 1.0
    max_missing_fraction: 1.0
env: {L: 5, Lv: 5, c_tc: 0.0}
prl: {alpha0: 0.2, beta: 1.0, lambdav: 2.0, bias: 0.0, alpha_min: 0.01, alpha_max: 1.0}
sac: {learning_rate: 0.001, batch_size: 32, gamma: 0.99, tau: 0.005, buffer_size: 1000, total_timesteps: 10, ent_coef: 0.2}
"""
    )

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["build_cache.py", "--config", str(cfg_path)])

    build_cache.main()

    assert calls["count"] > 0  # download attempted
    processed_dir = tmp_path / "data" / "processed"
    assert processed_dir.joinpath("prices.parquet").exists()
    assert processed_dir.joinpath("returns.parquet").exists()
    assert processed_dir.joinpath("data_manifest.json").exists()
    assert tmp_path.joinpath("outputs", "reports", "data_quality_summary.csv").exists()
