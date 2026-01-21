import torch as th

from prl.prl import PRLAlphaScheduler, PRLConfig


def _make_scheduler(**overrides) -> PRLAlphaScheduler:
    cfg = PRLConfig(
        alpha0=0.2,
        beta=10.0,
        lambdav=1.0,
        bias=0.0,
        alpha_min=0.05,
        alpha_max=0.5,
        vol_mean=0.0,
        vol_std=1.0,
        window_size=2,
        num_assets=3,
        **overrides,
    )
    return PRLAlphaScheduler(cfg)


def _make_obs(scheduler: PRLAlphaScheduler, values) -> th.Tensor:
    obs_dim = scheduler.cfg.window_size * scheduler.cfg.num_assets + 2 * scheduler.cfg.num_assets
    obs = th.zeros((1, obs_dim), dtype=th.float32)
    obs[0, scheduler.vol_slice] = th.tensor(values, dtype=th.float32)
    return obs


def test_emergency_clamp_triggered():
    scheduler = _make_scheduler(emergency_mode="clamp")
    obs = _make_obs(scheduler, [5.0, 5.0, 5.0])
    _, diag = scheduler.alpha_from_obs(obs, return_diagnostics=True)
    assert bool(diag.emergency.item()) is True


def test_emergency_vz_mode_triggered():
    scheduler = _make_scheduler(emergency_mode="vz", emergency_vz_threshold=1.0)
    obs = _make_obs(scheduler, [2.0, 2.0, 2.0])
    _, diag = scheduler.alpha_from_obs(obs, return_diagnostics=True)
    assert bool(diag.emergency.item()) is True
