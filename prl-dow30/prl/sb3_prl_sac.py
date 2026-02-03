from __future__ import annotations

from typing import Type

import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import polyak_update

from .prl import PRLAlphaScheduler


class PRLSAC(SAC):
    """Stable-Baselines3 SAC with Method-A PRL alpha injection."""

    def __init__(
        self,
        policy: Type[BasePolicy] | str,
        env: GymEnv,
        scheduler: PRLAlphaScheduler | None = None,
        **kwargs,
    ):
        kwargs.setdefault("ent_coef", 1.0)  # unused but required by base class
        super().__init__(policy, env, **kwargs)
        self.scheduler = scheduler
        # Disable SB3 entropy auto update
        self.log_ent_coef = None
        self.ent_coef_optimizer = None

    def train(self, gradient_steps: int, batch_size: int) -> None:
        if self.scheduler is None:
            raise ValueError("PRL scheduler is required for training.")
        self.policy.set_training_mode(True)
        def _record_tensor_stats(name: str, tensor: th.Tensor) -> None:
            values = tensor.detach().float().view(-1)
            if values.numel() == 0:
                return
            self.logger.record(f"train/{name}_min", values.min().item())
            self.logger.record(f"train/{name}_max", values.max().item())
            self.logger.record(f"train/{name}_p05", th.quantile(values, 0.05).item())
            self.logger.record(f"train/{name}_p50", th.quantile(values, 0.50).item())
            self.logger.record(f"train/{name}_p95", th.quantile(values, 0.95).item())
            self.logger.record(f"train/{name}_std", values.std(unbiased=False).item())

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_actions, next_log_prob = self.policy.actor.action_log_prob(replay_data.next_observations)
                if next_log_prob.dim() == 1:
                    next_log_prob = next_log_prob.unsqueeze(-1)
                assert next_log_prob.shape[1] == 1, "next_log_prob must be [B,1]"
                q_next = self.critic_target(replay_data.next_observations, next_actions)
                q_next_cat = th.cat(q_next, dim=1)
                min_q_next = th.min(q_next_cat, dim=1, keepdim=True)[0]
                alpha_next = self.scheduler.alpha_from_obs(replay_data.next_observations).detach()
                assert alpha_next.shape == next_log_prob.shape, "alpha_next shape mismatch"
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (
                    min_q_next - alpha_next * next_log_prob
                )

            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(th.nn.functional.mse_loss(q, target_q) for q in current_q)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actions_pi, log_prob = self.policy.actor.action_log_prob(replay_data.observations)
            if log_prob.dim() == 1:
                log_prob = log_prob.unsqueeze(-1)
            assert log_prob.shape[1] == 1, "log_prob must be [B,1]"
            q_pi = self.critic(replay_data.observations, actions_pi)
            min_q_pi = th.min(th.cat(q_pi, dim=1), dim=1, keepdim=True)[0]
            alpha_obs, diagnostics = self.scheduler.alpha_from_obs(replay_data.observations, return_diagnostics=True)
            alpha_obs = alpha_obs.detach()
            assert alpha_obs.shape == log_prob.shape, "alpha_obs shape mismatch"
            actor_loss_base = (alpha_obs * log_prob - min_q_pi).mean()

            # CVaR/variance penalty hooks to down-weight tail risk or reduce variance of returns.
            # penalty_raw terms keep gradients; penalty_weighted scales by gamma/beta.
            gamma = getattr(self.scheduler.cfg, "cvar_penalty_gamma", None)
            beta = getattr(self.scheduler.cfg, "var_penalty_beta", None)
            if beta is None and gamma not in (None, 0):
                # Fallback: reuse gamma as variance weight when beta is unspecified to ensure penalty has effect.
                beta = gamma
            # CVaR-style penalty: emphasize lower-tail Q-values (25% quantile)
            tail_q = th.quantile(min_q_pi.detach(), 0.25)
            cvar_penalty_raw = th.relu(tail_q - min_q_pi).mean()
            var_penalty_raw = th.var(min_q_pi, unbiased=False)
            penalty_weighted = th.zeros_like(actor_loss_base)
            if gamma is not None and gamma != 0:
                penalty_weighted = penalty_weighted + cvar_penalty_raw * float(gamma)
            if beta is not None and beta != 0:
                penalty_weighted = penalty_weighted + var_penalty_raw * float(beta)

            # Clip penalty magnitude to avoid loss blow-ups when variance term explodes.
            clamp_ratio = getattr(self.scheduler.cfg, "penalty_clip_ratio", 0.2)
            max_penalty = clamp_ratio * actor_loss_base.abs()
            penalty_weighted = th.clamp(penalty_weighted, min=-max_penalty, max=max_penalty)

            actor_loss = actor_loss_base + penalty_weighted

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor.optimizer.step()

            self._n_updates += 1
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            self.logger.record("train/critic_loss", critic_loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.logger.record("train/alpha_obs_mean", alpha_obs.mean().item())
            self.logger.record("train/alpha_next_mean", alpha_next.mean().item())
            self.logger.record("train/prl_prob_mean", diagnostics.prl_prob.mean().item())
            self.logger.record("train/vz_mean", diagnostics.vz.mean().item())
            self.logger.record("train/alpha_raw_mean", diagnostics.alpha_raw.mean().item())
            self.logger.record("train/alpha_clamped_mean", diagnostics.alpha_clamped.mean().item())
            self.logger.record("train/emergency_rate", diagnostics.emergency.float().mean().item())
            self.logger.record("train/beta_effective_mean", diagnostics.beta_effective.mean().item())
            _record_tensor_stats("prl_prob", diagnostics.prl_prob)
            _record_tensor_stats("vz", diagnostics.vz)
            _record_tensor_stats("alpha_raw", diagnostics.alpha_raw)
            _record_tensor_stats("alpha_obs", alpha_obs)
            self.logger.record("train/entropy_loss", float((alpha_obs * log_prob).mean().item()))
            self.logger.record("train/actor_loss_base", actor_loss_base.item())
            self.logger.record("train/cvar_penalty_raw_mean", cvar_penalty_raw.item())
            self.logger.record("train/cvar_penalty_weighted_mean", (cvar_penalty_raw * float(gamma or 0.0)).item())
            self.logger.record("train/var_penalty_raw_mean", var_penalty_raw.item())
            self.logger.record("train/var_penalty_weighted_mean", (var_penalty_raw * float(beta or 0.0)).item())
            penalty_ratio = abs(penalty_weighted.item()) / (abs(actor_loss_base.item()) + 1e-8)
            self.logger.record("train/penalty_ratio", penalty_ratio)
