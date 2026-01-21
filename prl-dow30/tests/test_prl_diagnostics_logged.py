from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch as th

from prl.prl import PRLAlphaScheduler, PRLConfig
from prl.sb3_prl_sac import PRLSAC
from prl.train import TrainLoggingCallback
from stable_baselines3.common.logger import configure


class DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


def test_prl_diagnostics_logged(tmp_path):
    cfg = PRLConfig(
        alpha0=0.2,
        beta=1.0,
        lambdav=1.0,
        bias=0.0,
        alpha_min=0.05,
        alpha_max=0.5,
        vol_mean=0.0,
        vol_std=1.0,
        window_size=1,
        num_assets=2,
    )
    scheduler = PRLAlphaScheduler(cfg)
    obs_dim = cfg.window_size * cfg.num_assets + 2 * cfg.num_assets
    env = DummyEnv(obs_dim=obs_dim, action_dim=cfg.num_assets)

    model = PRLSAC(
        "MlpPolicy",
        env,
        scheduler=scheduler,
        learning_rate=0.001,
        buffer_size=10,
        batch_size=2,
        gamma=0.99,
        tau=0.005,
    )
    model.set_logger(configure(folder=str(tmp_path / "sb3_logs"), format_strings=["csv"]))

    batch_size = 2
    device = model.device
    observations = th.zeros((batch_size, obs_dim), device=device)
    observations[:, scheduler.vol_slice] = 0.5
    next_observations = observations.clone()
    actions = th.zeros((batch_size, cfg.num_assets), device=device)
    rewards = th.zeros((batch_size, 1), device=device)
    dones = th.zeros((batch_size, 1), device=device)

    replay_data = SimpleNamespace(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    model.replay_buffer.sample = lambda *args, **kwargs: replay_data

    model.train(gradient_steps=1, batch_size=batch_size)

    required_keys = [
        "train/prl_prob_mean",
        "train/vz_mean",
        "train/alpha_raw_mean",
        "train/alpha_clamped_mean",
        "train/emergency_rate",
        "train/beta_effective_mean",
    ]
    for key in required_keys:
        assert key in model.logger.name_to_value

    log_path = tmp_path / "train.csv"
    cb = TrainLoggingCallback(log_path, run_id="run123", model_type="prl", seed=0, log_interval=1)
    cb.model = model
    cb.num_timesteps = 1
    cb._on_step()
    cb._on_training_end()

    df = pd.read_csv(log_path)
    assert "prl_prob_mean" in df.columns
    assert df["prl_prob_mean"].notna().any()
