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
            alpha_obs = self.scheduler.alpha_from_obs(replay_data.observations).detach()
            assert alpha_obs.shape == log_prob.shape, "alpha_obs shape mismatch"
            actor_loss = (alpha_obs * log_prob - min_q_pi).mean()

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor.optimizer.step()

            self._n_updates += 1
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            self.logger.record("train/critic_loss", critic_loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.logger.record("train/alpha_obs_mean", alpha_obs.mean().item())
            self.logger.record("train/alpha_next_mean", alpha_next.mean().item())
