import pytest

from prl.train import run_training


def _base_config():
    return {
        "mode": "paper",
        "dates": {
            "train_start": "2010-01-01",
            "train_end": "2010-02-01",
            "test_start": "2010-02-02",
            "test_end": "2010-03-01",
        },
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed", "paper_mode": True, "require_cache": True},
        "env": {"L": 5, "Lv": 5, "c_tc": 0.0, "logit_scale": 1.0},
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
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 100000,
            "total_timesteps": 100000,
            "ent_coef": 0.2,
        },
    }


def test_guard_raises_on_small_timesteps(tmp_path, monkeypatch):
    cfg = _base_config()
    cfg["sac"]["total_timesteps"] = 10
    cfg["data"]["processed_dir"] = str(tmp_path / "processed")
    with pytest.raises(ValueError):
        run_training(cfg, "baseline", seed=0, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", force_refresh=False)


def test_guard_raises_on_small_buffer(tmp_path):
    cfg = _base_config()
    cfg["sac"]["buffer_size"] = 10
    cfg["data"]["processed_dir"] = str(tmp_path / "processed")
    with pytest.raises(ValueError):
        run_training(cfg, "baseline", seed=0, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", force_refresh=False)


def test_cache_missing_in_paper_mode(tmp_path):
    cfg = _base_config()
    cfg["data"]["processed_dir"] = str(tmp_path / "processed")
    with pytest.raises(RuntimeError, match="CACHE_MISSING"):
        run_training(cfg, "baseline", seed=0, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", force_refresh=False)
