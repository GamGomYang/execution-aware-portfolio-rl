"""Replay buffer that stores state, risk features, and Hebbian traces."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

Tensor = torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        risk_dim: int,
        hebb_dim: int,
        log_interval: int = 1000,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.risk_dim = risk_dim
        self.hebb_dim = hebb_dim
        self.ptr = 0
        self.size = 0
        self.log_interval = log_interval
        self.use_stress_priority = False
        self.priority_exponent = 1.0
        self.priority_eps = 1e-3

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.risks = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.plasticity = np.zeros((capacity, 1), dtype=np.float32)
        self.next_plasticity = np.zeros((capacity, 1), dtype=np.float32)
        self.risk_feats = np.zeros((capacity, risk_dim), dtype=np.float32)
        self.next_risk_feats = np.zeros((capacity, risk_dim), dtype=np.float32)
        self.hebb = np.zeros((capacity, hebb_dim), dtype=np.float32)
        self.next_hebb = np.zeros((capacity, hebb_dim), dtype=np.float32)
        self.stress = np.zeros((capacity, 1), dtype=np.float32)
        self.next_stress = np.zeros((capacity, 1), dtype=np.float32)
        self.beta_values = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        risk_value: float,
        done: bool,
        next_state: np.ndarray,
        risk_feat: np.ndarray,
        next_risk_feat: np.ndarray,
        hebb_state: np.ndarray,
        next_hebb_state: np.ndarray,
        plasticity_value: float,
        next_plasticity_value: float,
        stress_value: float | None = None,
        next_stress_value: float | None = None,
        beta_value: float | None = None,
    ) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.risks[idx] = risk_value
        self.dones[idx] = float(done)
        self.next_states[idx] = next_state
        self.risk_feats[idx] = risk_feat
        self.next_risk_feats[idx] = next_risk_feat
        self.hebb[idx] = hebb_state
        self.next_hebb[idx] = next_hebb_state
        self.plasticity[idx] = plasticity_value
        self.next_plasticity[idx] = next_plasticity_value
        if stress_value is not None:
            self.stress[idx] = stress_value
        if next_stress_value is not None:
            self.next_stress[idx] = next_stress_value
        if beta_value is not None:
            self.beta_values[idx] = beta_value

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
        if self.size == self.capacity or (self.size > 0 and self.ptr % self.log_interval == 0):
            self._log_distributions()

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        if self.use_stress_priority and self.size > 0:
            diffs = np.abs(self.stress[: self.size] - self.next_stress[: self.size])
            probs = (diffs + self.priority_eps) ** self.priority_exponent
            probs = probs / probs.sum()
            idx = np.random.choice(self.size, size=batch_size, replace=True, p=probs.squeeze(-1))
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.tensor(self.states[idx], device=device),
            "action": torch.tensor(self.actions[idx], device=device),
            "reward": torch.tensor(self.rewards[idx], device=device),
            "risk": torch.tensor(self.risks[idx], device=device),
            "done": torch.tensor(self.dones[idx], device=device),
            "next_state": torch.tensor(self.next_states[idx], device=device),
            "risk_feat": torch.tensor(self.risk_feats[idx], device=device),
            "next_risk_feat": torch.tensor(self.next_risk_feats[idx], device=device),
            "hebb": torch.tensor(self.hebb[idx], device=device),
            "next_hebb": torch.tensor(self.next_hebb[idx], device=device),
            "plasticity": torch.tensor(self.plasticity[idx], device=device),
            "next_plasticity": torch.tensor(self.next_plasticity[idx], device=device),
            "stress": torch.tensor(self.stress[idx], device=device),
            "next_stress": torch.tensor(self.next_stress[idx], device=device),
            "beta": torch.tensor(self.beta_values[idx], device=device),
        }

    def set_stress_priority(self, enabled: bool, exponent: float = 1.0, eps: float = 1e-3) -> None:
        self.use_stress_priority = enabled
        self.priority_exponent = exponent
        self.priority_eps = eps

    def _log_distributions(self) -> None:
        if self.size == 0:
            return
        rewards = self.rewards[: self.size]
        stress_vals = self.stress[: self.size]
        beta_vals = self.beta_values[: self.size]
        print(
            "[replay] stats "
            f"reward_mean={rewards.mean():.4f} reward_min={rewards.min():.4f} reward_max={rewards.max():.4f} "
            f"stress_mean={stress_vals.mean():.4f} stress_std={stress_vals.std():.4f} "
            f"beta_mean={beta_vals.mean():.4f} beta_std={beta_vals.std():.4f}",
            flush=True,
        )
