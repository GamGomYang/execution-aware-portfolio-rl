from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prl import data as data_module
from prl.eval import load_model, run_backtest_episode
from prl.train import (
    build_env_for_range,
    prepare_market_and_features,
    run_training,
)
from scripts import run_eval as run_eval_script


@pytest.fixture
def fake_download(monkeypatch):
    def _fake_fetch(tickers, start, end, session_opts=None):
        ticker = tickers[0]
        idx = pd.date_range(start=start, end=end, freq="B")
        base = sum(ord(c) for c in ticker) % 30 + 100
        values = base + np.linspace(0, len(idx) - 1, len(idx)) * 0.5
        return pd.DataFrame({ticker: values}, index=idx)

    monkeypatch.setattr(data_module, "fetch_yfinance", _fake_fetch)
    monkeypatch.setattr(data_module, "DOW30_TICKERS", ("AAA", "BBB"))


def test_short_train_and_eval_pipeline(tmp_path, fake_download):
    config = {
        "dates": {
            "train_start": "2010-01-01",
            "train_end": "2010-02-15",
            "test_start": "2010-02-16",
            "test_end": "2010-03-31",
        },
        "data": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 2,
            "force_refresh": True,
            "offline": False,
            "require_cache": False,
            "paper_mode": False,
            "history_tolerance_days": 0,
            "min_history_days": 50,
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_missing_fraction": 1.0,
                "max_flat_fraction": 1.0,
            },
            "ticker_substitutions": {},
        },
        "env": {"L": 5, "Lv": 5, "c_tc": 0.0001},
        "prl": {
            "alpha0": 0.2,
            "beta": 1.0,
            "lambdav": 2.0,
            "bias": 0.0,
            "alpha_min": 0.01,
            "alpha_max": 1.0,
        },
        "sac": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "gamma": 0.95,
            "tau": 0.02,
            "buffer_size": 500,
            "total_timesteps": 20,
            "eval_freq": 10,
            "ent_coef": 0.2,
        },
    }

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    outputs_root = tmp_path / "outputs"
    models_dir = outputs_root / "models"

    model_path = run_training(
        config=config,
        model_type="baseline",
        seed=0,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=models_dir,
        force_refresh=True,
    )

    assert model_path.exists()

    market, features = prepare_market_and_features(
        config=config,
        lv=config["env"]["Lv"],
        force_refresh=False,
        offline=False,
        require_cache=False,
        paper_mode=False,
        cache_only=False,
    )

    env = build_env_for_range(
        market=market,
        features=features,
        start=config["dates"]["test_start"],
        end=config["dates"]["test_end"],
        window_size=config["env"]["L"],
        c_tc=config["env"]["c_tc"],
        seed=0,
    )

    model = load_model(model_path, "baseline", env, scheduler=None)
    metrics = run_backtest_episode(model, env)

    metrics_path = outputs_root / "reports" / "metrics.csv"
    run_eval_script.write_metrics(
        metrics_path,
        {
            "model_type": "baseline",
            "seed": 0,
            **metrics.to_dict(),
        },
    )

    assert metrics_path.exists()
    contents = metrics_path.read_text().strip().splitlines()
    assert len(contents) >= 2  # header + row
