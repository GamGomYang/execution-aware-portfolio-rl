import torch as th

from prl.prl import PRLAlphaScheduler, PRLConfig


def test_beta_effective_safe_when_prob_near_zero():
    cfg = PRLConfig(
        alpha0=0.2,
        beta=1.0,
        lambdav=1.0,
        bias=-100.0,
        alpha_min=0.05,
        alpha_max=0.5,
        vol_mean=0.0,
        vol_std=1.0,
        window_size=2,
        num_assets=2,
    )
    scheduler = PRLAlphaScheduler(cfg)
    obs_dim = cfg.window_size * cfg.num_assets + 2 * cfg.num_assets
    obs = th.zeros((1, obs_dim), dtype=th.float32)

    _, diag = scheduler.alpha_from_obs(obs, return_diagnostics=True)

    assert th.isfinite(diag.beta_effective).all()
    assert diag.beta_effective.item() == 0.0
