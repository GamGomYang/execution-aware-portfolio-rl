import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from prl.data import MarketData
from prl.data import slice_frame
from prl.eval import trace_dict_to_frame
from prl.features import VolatilityFeatures
from prl.metrics import PortfolioMetrics, compute_metrics
from prl.train import _write_run_metadata


def _default_config(processed_dir: Path, seeds: Iterable[int]) -> dict:
    return {
        "mode": "smoke",
        "dates": {
            "train_start": "2020-01-01",
            "train_end": "2020-01-10",
            "test_start": "2020-01-01",
            "test_end": "2020-01-10",
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": str(processed_dir),
            "source": "yfinance_only",
            "universe_policy": "availability_filtered",
            "min_assets": 1,
            "history_tolerance_days": 0,
            "min_history_days": 5,
            "require_cache": True,
            "paper_mode": True,
            "offline": True,
            "quality_params": {
                "min_vol_std": 0.0,
                "min_max_abs_return": 0.0,
                "max_missing_fraction": 1.0,
                "max_flat_fraction": 1.0,
            },
            "ticker_substitutions": {},
        },
        "env": {"L": 2, "Lv": 2, "c_tc": 0.0, "logit_scale": 1.0},
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
            "batch_size": 8,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 1000,
            "total_timesteps": 5,
            "ent_coef": 0.2,
        },
        "seeds": list(seeds),
    }


def run_stubbed_run_all(tmp_path: Path, monkeypatch, *, output_root: str | Path | None = None, model_types: Iterable[str] | None = None, seeds: Iterable[int] | None = None, eval_cfg: dict | None = None) -> dict:
    """Run scripts.run_all.main with lightweight stubs and return context."""
    monkeypatch.chdir(tmp_path)
    seeds = list(seeds) if seeds is not None else [0]
    model_types = list(model_types) if model_types is not None else ["baseline", "prl"]
    output_root_path = Path(output_root) if output_root is not None else tmp_path / "outputs"

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cfg = _default_config(processed_dir, seeds)
    if eval_cfg:
        cfg["eval"] = eval_cfg
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
    prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    volatility = pd.DataFrame(0.02, index=dates, columns=["AAA", "BBB"])
    market = MarketData(prices=prices, returns=returns)

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": 0.0, "std": 1.0}))
    features = VolatilityFeatures(
        volatility=volatility,
        portfolio_scalar=pd.Series(0.0, index=dates),
        mean=0.0,
        std=1.0,
        stats_path=stats_path,
    )

    def _fake_prepare_market_and_features(*args, **kwargs):
        return market, features

    def _fake_run_training(*args, **kwargs):
        config = kwargs.get("config", cfg)
        model_type = kwargs.get("model_type", "baseline")
        seed = kwargs.get("seed", 0)
        reports_dir = Path(kwargs.get("reports_dir", output_root_path / "reports"))
        logs_dir = Path(kwargs.get("logs_dir", output_root_path / "logs"))
        output_dir = Path(kwargs.get("output_dir", output_root_path / "models"))
        processed_dir_local = Path(config["data"]["processed_dir"])
        processed_dir_local.mkdir(parents=True, exist_ok=True)
        manifest = {
            "asset_list": ["AAA", "BBB"],
            "num_assets": 2,
            "L": config["env"]["L"],
            "Lv": config["env"]["Lv"],
            "obs_dim_expected": 2 * (config["env"]["L"] + 2),
            "env_schema_version": "v1",
        }
        (processed_dir_local / "data_manifest.json").write_text(json.dumps(manifest))
        run_id = f"runid_{model_type}_seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{run_id}_final.zip"
        model_path.write_text("stub")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"train_{run_id}.csv"
        log_path.write_text("schema_version,run_id,model_type,seed,timesteps\n1.1,run,baseline,0,1\n")
        reports_dir.mkdir(parents=True, exist_ok=True)
        _write_run_metadata(reports_dir, config, seed, config.get("mode", ""), model_type, run_id, model_path, log_path)
        return model_path

    def _fake_build_env_for_range(*, market, start, end, **kwargs):
        class DummyEnv:
            def __init__(self, returns):
                self.returns = returns
                self.num_assets = self.returns.shape[1]
                self.window_size = cfg["env"]["L"]
                self.observation_space = type("Space", (), {"shape": (self.num_assets * (self.window_size + 2),)})
                self.cfg = type("Cfg", (), {"transaction_cost": cfg["env"]["c_tc"]})

        returns_slice = slice_frame(market.returns, start, end)
        return DummyEnv(returns_slice)

    def _fake_run_backtest_episode_detailed(model, env):
        portfolio_returns = [0.01, 0.0, -0.005]
        costs = [0.001, 0.001, 0.0]
        rewards = [math.log(1.0 + r) - c for r, c in zip(portfolio_returns, costs)]
        net_returns_exp = [math.exp(r) - 1.0 for r in rewards]
        net_returns_lin = [r - c for r, c in zip(portfolio_returns, costs)]
        turnovers = [0.1, 0.1, 0.05]
        metrics = compute_metrics(
            rewards,
            portfolio_returns,
            turnovers,
            net_returns_exp=net_returns_exp,
            net_returns_lin=net_returns_lin,
        )
        trace = {
            "dates": list(env.returns.index[: len(portfolio_returns)]),
            "rewards": rewards,
            "portfolio_returns": portfolio_returns,
            "turnovers": turnovers,
            "turnover_target_changes": [0.05, 0.05, 0.05],
            "costs": costs,
            "net_returns_exp": net_returns_exp,
            "net_returns_lin": net_returns_lin,
        }
        return metrics, trace

    def _fake_eval_strategies_to_trace(returns, volatility, *, transaction_cost: float, eval_id: str, run_id: str, seed: int, **kwargs):
        portfolio_returns = [0.008, -0.004, 0.0]
        costs = [0.0005, 0.0005, 0.0]
        rewards = [math.log(1.0 + r) - c for r, c in zip(portfolio_returns, costs)]
        net_returns_exp = [math.exp(r) - 1.0 for r in rewards]
        net_returns_lin = [r - c for r, c in zip(portfolio_returns, costs)]
        turnovers = [0.05, 0.05, 0.05]
        dates_local = list(returns.index[: len(portfolio_returns)])
        metrics = compute_metrics(
            rewards,
            portfolio_returns,
            turnovers,
            net_returns_exp=net_returns_exp,
            net_returns_lin=net_returns_lin,
        )
        trace = {
            "dates": dates_local,
            "rewards": rewards,
            "portfolio_returns": portfolio_returns,
            "turnovers": turnovers,
            "turnover_target_changes": [0.02, 0.02, 0.02],
            "costs": costs,
            "net_returns_exp": net_returns_exp,
            "net_returns_lin": net_returns_lin,
        }
        df = trace_dict_to_frame(trace, eval_id=eval_id, run_id=run_id, model_type="buy_and_hold_equal_weight", seed=seed)
        return {"buy_and_hold_equal_weight": metrics}, df

    monkeypatch.setattr("scripts.run_all.prepare_market_and_features", _fake_prepare_market_and_features)
    monkeypatch.setattr("scripts.run_all.run_training", _fake_run_training)
    monkeypatch.setattr("scripts.run_all.build_env_for_range", _fake_build_env_for_range)
    monkeypatch.setattr("scripts.run_all.load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("prl.eval.run_backtest_episode_detailed", _fake_run_backtest_episode_detailed)
    monkeypatch.setattr("scripts.run_all.eval_strategies_to_trace", _fake_eval_strategies_to_trace)
    monkeypatch.setattr("scripts.run_all.create_scheduler", lambda *args, **kwargs: None)

    argv = ["run_all.py", "--config", str(config_path), "--output-root", str(output_root_path), "--seeds", *[str(s) for s in seeds], "--model-types", *model_types, "--offline"]
    monkeypatch.setattr("sys.argv", argv)

    from scripts import run_all as run_all_script

    run_all_script.main()

    reports_dir = output_root_path / "reports"
    run_index_path = reports_dir / "run_index.json"
    run_index = json.loads(run_index_path.read_text())
    return {"output_root": output_root_path, "reports_dir": reports_dir, "run_index": run_index}
