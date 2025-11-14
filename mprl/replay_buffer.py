"""Replay buffer that stores state, risk features, and Hebbian traces."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

Tensor = torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, risk_dim: int, hebb_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.risk_dim = risk_dim
        self.hebb_dim = hebb_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.risks = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.risk_feats = np.zeros((capacity, risk_dim), dtype=np.float32)
        self.next_risk_feats = np.zeros((capacity, risk_dim), dtype=np.float32)
        self.hebb = np.zeros((capacity, hebb_dim), dtype=np.float32)
        self.next_hebb = np.zeros((capacity, hebb_dim), dtype=np.float32)

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

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
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
        }
