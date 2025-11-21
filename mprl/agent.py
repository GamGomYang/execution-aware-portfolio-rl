"""MPRL agent integrating policy, critics, and metaplasticity."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import math
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

        self.micro_dim = feature_cfg.micro_dim
        self.sector_dim = feature_cfg.sector_dim
        self.macro_dim = feature_cfg.macro_dim
        self.portfolio_dim = feature_cfg.portfolio_dim
        self.sector_summary_dim = feature_cfg.sector_summary_dim

        sector_input_dim = self.macro_dim + self.portfolio_dim + self.sector_summary_dim

        self.asset_actor = DirichletPolicy(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.sector_actor = DirichletPolicy(sector_input_dim, len(feature_cfg.sector_names)).to(self.device)
        self.critic1 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.critic2 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.target_critic1 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.target_critic2 = Critic(feature_cfg.total_state_dim, feature_cfg.num_assets).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.risk_module = RiskModule(feature_cfg.risk_feature_dim, feature_cfg.hebbian_dim, train_cfg).to(self.device)

        actor_params = list(self.asset_actor.parameters()) + list(self.sector_actor.parameters())
        self.actor_opt = torch.optim.Adam(actor_params, lr=train_cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=train_cfg.lr_critic
        )
        self.meta_opt = torch.optim.Adam(self.risk_module.parameters(), lr=train_cfg.lr_meta)

        self.base_lr_actor = train_cfg.lr_actor
        self.base_lr_critic = train_cfg.lr_critic
        self.base_lr_meta = train_cfg.lr_meta

        self.sector_ids = torch.tensor(
            [feature_cfg.sector_names.index(feature_cfg.dow_sector_map.get(sym, feature_cfg.sector_names[0])) for sym in feature_cfg.dow_symbols],
            dtype=torch.long,
            device=self.device,
        )
        self.sector_ids_np = self.sector_ids.cpu().numpy()
        self.num_sectors = len(feature_cfg.sector_names)
        self.step_count = 0

    def policy_state_dict(self) -> Dict[str, Dict]:
        return {
            "actor": self.asset_actor.state_dict(),
            "sector_actor": self.sector_actor.state_dict(),
            "risk_module": self.risk_module.state_dict(),
            "feature_cfg": asdict(self.feature_cfg),
            "train_cfg": asdict(self.cfg),
        }

    def save_policy(self, path: str) -> None:
        torch.save(self.policy_state_dict(), path)

    def load_policy(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.asset_actor.load_state_dict(state["actor"], strict=strict)
        if "sector_actor" in state:
            self.sector_actor.load_state_dict(state["sector_actor"], strict=strict)
        self.risk_module.load_state_dict(state["risk_module"], strict=strict)

    def _split_state_components(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        idx = 0
        micro = state[..., idx : idx + self.micro_dim]
        idx += self.micro_dim
        sector = state[..., idx : idx + self.sector_dim]
        idx += self.sector_dim
        macro = state[..., idx : idx + self.macro_dim]
        idx += self.macro_dim
        portfolio = state[..., idx : idx + self.portfolio_dim]
        idx += self.portfolio_dim
        sector_summary = state[..., idx : idx + self.sector_summary_dim]
        return micro, sector, macro, portfolio, sector_summary

    def _combine_actions_tensor(self, sector_weights: Tensor, asset_weights: Tensor) -> Tensor:
        batch_size = sector_weights.size(0)
        ids = self.sector_ids.unsqueeze(0).expand(batch_size, -1)
        sector_alloc = sector_weights.gather(1, ids)
        sector_sums = torch.zeros(batch_size, self.num_sectors, device=asset_weights.device)
        sector_sums.scatter_add_(1, ids, asset_weights)
        denom = sector_sums.gather(1, ids).clamp_min(1e-6)
        intra = asset_weights / denom
        return intra * sector_alloc

    def _combine_actions_numpy(self, sector_weights: np.ndarray, asset_weights: np.ndarray) -> np.ndarray:
        sector_alloc = sector_weights[self.sector_ids_np]
        sector_sums = np.zeros(self.num_sectors, dtype=np.float32)
        np.add.at(sector_sums, self.sector_ids_np, asset_weights)
        denom = sector_sums[self.sector_ids_np]
        denom = np.where(denom <= 1e-6, 1.0, denom)
        intra = asset_weights / denom
        combined = intra * sector_alloc
        combined /= combined.sum() + 1e-6
        return combined

    def _sector_actor_input(self, state: Tensor) -> Tensor:
        _, _, macro, portfolio, sector_summary = self._split_state_components(state)
        return torch.cat([macro, portfolio, sector_summary], dim=-1)

    def _current_epsilon(self, plasticity_signal: float) -> float:
        base = self.cfg.epsilon_base * math.exp(-self.step_count / max(self.cfg.epsilon_decay, 1e-6))
        epsilon = max(self.cfg.epsilon_min, base)
        epsilon = epsilon * (1.0 + self.cfg.epsilon_beta * plasticity_signal)
        return float(np.clip(epsilon, 0.0, 0.5))

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
            beta_in = beta_val.unsqueeze(0)
            sector_state = self._sector_actor_input(state_t)
            epsilon = self._current_epsilon(plasticity_signal)
            if deterministic:
                sector_act = self.sector_actor.deterministic(sector_state, beta_in)
                asset_act = self.asset_actor.deterministic(state_t, beta_in)
            else:
                sector_act, _ = self.sector_actor.sample(sector_state, beta_in)
                asset_act, _ = self.asset_actor.sample(state_t, beta_in)
                if epsilon > 1e-6:
                    uniform_assets = torch.ones_like(asset_act) / asset_act.size(-1)
                    asset_act = (1.0 - epsilon) * asset_act + epsilon * uniform_assets
                    uniform_sectors = torch.ones_like(sector_act) / sector_act.size(-1)
                    sector_act = (1.0 - epsilon) * sector_act + epsilon * uniform_sectors
            combined = self._combine_actions_tensor(sector_act, asset_act)
        final_np = self._combine_actions_numpy(
            sector_act.squeeze(0).cpu().numpy(), asset_act.squeeze(0).cpu().numpy()
        )
        self.step_count += 1
        return final_np, float(beta_val.item()), hebb, next_hebb

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
        self.meta_opt.zero_grad()

        beta_t, _ = self.encode_beta_batch(risk_feat, hebb)
        beta_tp1, _ = self.encode_beta_batch(next_risk_feat, next_hebb)

        tilde_reward = reward - self.cfg.risk_lambda * beta_t * risk_value

        with torch.no_grad():
            next_sector_state = self._sector_actor_input(next_state)
            next_sector, next_sector_log = self.sector_actor.sample(next_sector_state, beta_tp1)
            next_asset, next_asset_log = self.asset_actor.sample(next_state, beta_tp1)
            next_action = self._combine_actions_tensor(next_sector, next_asset)
            next_log_prob = next_sector_log + next_asset_log
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

        sector_state = self._sector_actor_input(state)
        sector_pi, log_sector = self.sector_actor.sample(sector_state, beta_t)
        asset_pi, log_asset = self.asset_actor.sample(state, beta_t)
        actions_pi = self._combine_actions_tensor(sector_pi, asset_pi)
        log_prob_pi = log_sector + log_asset
        q1_pi = self.critic1(state, actions_pi)
        q2_pi = self.critic2(state, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        value_weight = 1.0 + self.cfg.k_q * beta_t
        alpha_t = self._alpha_with_plasticity(plasticity)
        entropy_weight = alpha_t * (1.0 + self.cfg.k_alpha * beta_t)
        actor_loss = (entropy_weight * log_prob_pi - value_weight * q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.asset_actor.parameters()) + list(self.sector_actor.parameters()), 5.0)
        self.actor_opt.step()

        beta_t_meta, _ = self.encode_beta_batch(risk_feat, hebb)
        homeostat = self.risk_module.beta_center.detach()
        meta_penalty = torch.mean((beta_t_meta - homeostat) ** 2)
        meta_penalty.backward()
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
