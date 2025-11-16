"""MPRL agent integrating policy, critics, and metaplasticity."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from .config import FeatureConfig, TrainingConfig
from .networks import Critic, DirichletPolicy
from .risk_module import RiskModule

Tensor = torch.Tensor


class MPRLAgent:
    def __init__(self, feature_cfg: FeatureConfig, train_cfg: TrainingConfig):
        self.feature_cfg = feature_cfg
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)

        self.actor = DirichletPolicy(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.critic1 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.critic2 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.target_critic1 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.target_critic2 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.risk_module = RiskModule(feature_cfg.risk_feature_dim, feature_cfg.hebbian_dim, train_cfg).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=train_cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=train_cfg.lr_critic
        )
        self.meta_opt = torch.optim.Adam(self.risk_module.parameters(), lr=train_cfg.lr_meta)

        self.base_lr_actor = train_cfg.lr_actor
        self.base_lr_critic = train_cfg.lr_critic
        self.base_lr_meta = train_cfg.lr_meta

    def select_action(
        self,
        state: np.ndarray,
        risk_features: np.ndarray,
        signal: float,
        plasticity_signal: float = 0.0,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        risk_t = torch.tensor(risk_features, dtype=torch.float32, device=self.device)
        signal_t = torch.tensor([signal], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            beta_val, _, hebb, next_hebb = self.risk_module.observe(risk_t, signal_t)
            if deterministic:
                action = self.actor.deterministic(state_t, beta_val.unsqueeze(0))
            else:
                action, _ = self.actor.sample(state_t, beta_val.unsqueeze(0))
                epsilon = self.cfg.epsilon_base * (1.0 + self.cfg.epsilon_beta * plasticity_signal)
                epsilon = float(np.clip(epsilon, 0.0, 0.5))
                if epsilon > 1e-6:
                    uniform = torch.ones_like(action) / action.size(-1)
                    action = (1.0 - epsilon) * action + epsilon * uniform
        return action.squeeze(0).cpu().numpy(), float(beta_val.item()), hebb, next_hebb

    def encode_beta_batch(self, risk_feat: Tensor, hebb: Tensor) -> Tuple[Tensor, Tensor]:
        return self.risk_module.encode_batch(risk_feat, hebb)

    def soft_update(self, net: nn.Module, target: nn.Module) -> None:
        tau = 1.0 - self.cfg.polyak
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(self.cfg.polyak)
            tp.data.add_(tau * p.data)

    def _alpha_with_plasticity(self, plasticity: Tensor) -> Tensor:
        return self.cfg.alpha * (1.0 + self.cfg.plasticity_lambda_entropy * plasticity)

    def _set_optimizer_lr(self, optimizer: torch.optim.Optimizer, base_lr: float, plasticity_value: float) -> None:
        scale = max(0.0, 1.0 + self.cfg.plasticity_lr_beta * plasticity_value)
        lr = base_lr * scale
        for group in optimizer.param_groups:
            group["lr"] = lr

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        risk_value = batch["risk"]
        done = batch["done"]
        next_state = batch["next_state"]
        risk_feat = batch["risk_feat"]
        next_risk_feat = batch["next_risk_feat"]
        hebb = batch["hebb"]
        next_hebb = batch["next_hebb"]
        plasticity = batch["plasticity"]
        next_plasticity = batch["next_plasticity"]

        mean_plasticity = float(plasticity.mean().item())
        self._set_optimizer_lr(self.actor_opt, self.base_lr_actor, mean_plasticity)
        self._set_optimizer_lr(self.critic_opt, self.base_lr_critic, mean_plasticity)
        self._set_optimizer_lr(self.meta_opt, self.base_lr_meta, mean_plasticity)

        beta_t, _ = self.encode_beta_batch(risk_feat, hebb)
        beta_tp1, _ = self.encode_beta_batch(next_risk_feat, next_hebb)

        tilde_reward = reward - self.cfg.risk_lambda * beta_t * risk_value

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, beta_tp1)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            alpha_tp1 = self._alpha_with_plasticity(next_plasticity)
            entropy_weight_tp1 = alpha_tp1 * (1.0 + self.cfg.k_alpha * beta_tp1)
            target_q = torch.min(target_q1, target_q2)
            target = tilde_reward + (1.0 - done) * self.cfg.gamma * (target_q - entropy_weight_tp1 * next_log_prob)

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_q1, target) + torch.nn.functional.mse_loss(current_q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 5.0)
        self.critic_opt.step()

        actions_pi, log_prob_pi = self.actor.sample(state, beta_t)
        q1_pi = self.critic1(state, actions_pi)
        q2_pi = self.critic2(state, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        value_weight = 1.0 + self.cfg.k_q * beta_t
        alpha_t = self._alpha_with_plasticity(plasticity)
        entropy_weight = alpha_t * (1.0 + self.cfg.k_alpha * beta_t)
        actor_loss = (entropy_weight * log_prob_pi - value_weight * q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        beta_t_meta, _ = self.encode_beta_batch(risk_feat, hebb)
        homeostat = self.risk_module.beta_center.detach()
        meta_loss = actor_loss.detach() + critic_loss.detach()
        meta_penalty = torch.mean((beta_t_meta - homeostat) ** 2)
        total_meta = meta_loss + meta_penalty
        self.meta_opt.zero_grad()
        total_meta.backward()
        torch.nn.utils.clip_grad_norm_(self.risk_module.parameters(), 5.0)
        self.meta_opt.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "meta_penalty": float(meta_penalty.item()),
            "beta_mean": float(beta_t_meta.mean().item()),
            "plasticity_mean": mean_plasticity,
        }
