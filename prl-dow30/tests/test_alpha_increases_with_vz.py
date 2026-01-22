import torch as th

from prl.prl import PRLAlphaScheduler, PRLConfig


def _make_obs(scheduler: PRLAlphaScheduler, vol_values) -> th.Tensor:
    obs_dim = scheduler.cfg.window_size * scheduler.cfg.num_assets + 2 * scheduler.cfg.num_assets
    obs = th.zeros((1, obs_dim), dtype=th.float32)
    obs[0, scheduler.vol_slice] = th.tensor(vol_values, dtype=th.float32)
    return obs


def test_alpha_increases_with_vz():
    cfg = PRLConfig(
        alpha0=0.2,
        beta=1.0,
        lambdav=2.0,
        bias=-2.0,
        alpha_min=0.01,
        alpha_max=1.0,
        vol_mean=0.0,
        vol_std=1.0,
        window_size=2,
        num_assets=3,
        center_prob=True,
    )
    scheduler = PRLAlphaScheduler(cfg)
    obs_low = _make_obs(scheduler, [0.0, 0.0, 0.0])
    obs_high = _make_obs(scheduler, [2.0, 2.0, 2.0])
    alpha_low = scheduler.alpha_from_obs(obs_low)
    alpha_high = scheduler.alpha_from_obs(obs_high)
    assert alpha_high.item() > alpha_low.item() + 0.05
