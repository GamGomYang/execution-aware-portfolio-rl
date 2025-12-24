import torch as th

from prl.prl import PRLAlphaScheduler, PRLConfig


def make_scheduler(vol_mean: float = 0.1, vol_std: float = 0.05) -> PRLAlphaScheduler:
    cfg = PRLConfig(
        alpha0=0.2,
        beta=1.0,
        lambdav=2.0,
        bias=0.0,
        alpha_min=0.05,
        alpha_max=0.5,
        vol_mean=vol_mean,
        vol_std=vol_std,
        window_size=2,
        num_assets=3,
    )
    return PRLAlphaScheduler(cfg)


def make_obs(scheduler: PRLAlphaScheduler, values) -> th.Tensor:
    obs_dim = scheduler.cfg.window_size * scheduler.cfg.num_assets + 2 * scheduler.cfg.num_assets
    obs = th.zeros((1, obs_dim), dtype=th.float32)
    obs[0, scheduler.vol_slice] = th.tensor(values, dtype=th.float32)
    return obs


def test_probabilities_and_alpha_bounds():
    scheduler = make_scheduler()
    obs = make_obs(scheduler, [0.15, 0.2, 0.25])
    P = scheduler.prl_probability(obs)
    alpha = scheduler.alpha_from_obs(obs)
    assert P.min().item() >= 0.0 and P.max().item() <= 1.0
    assert scheduler.cfg.alpha_min <= alpha.item() <= scheduler.cfg.alpha_max


def test_alpha_increases_with_higher_volatility():
    scheduler = make_scheduler()
    low_obs = make_obs(scheduler, [0.05, 0.06, 0.07])
    high_obs = make_obs(scheduler, [0.3, 0.35, 0.4])
    alpha_low = scheduler.alpha_from_obs(low_obs)
    alpha_high = scheduler.alpha_from_obs(high_obs)
    assert alpha_high.item() > alpha_low.item()


def test_scheduler_uses_provided_train_stats():
    scheduler_train = make_scheduler(vol_mean=0.1, vol_std=0.05)
    scheduler_shifted = make_scheduler(vol_mean=0.2, vol_std=0.05)
    obs = make_obs(scheduler_train, [0.1, 0.1, 0.1])
    alpha_train = scheduler_train.alpha_from_obs(obs)
    alpha_shifted = scheduler_shifted.alpha_from_obs(obs)
    assert not th.isclose(alpha_train, alpha_shifted)
