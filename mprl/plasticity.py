"""Plasticity signal controller computing adaptive modulation terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlasticityCoefficients:
    alpha_state: float
    alpha_reward: float
    alpha_uncertainty: float


class PlasticityController:
    """Tracks state/reward deltas and emits the plasticity scalar P_t."""

    def __init__(self, coeffs: PlasticityCoefficients) -> None:
        self.coeffs = coeffs
        self.prev_state: Optional[np.ndarray] = None
        self.prev_reward: Optional[float] = None
        self.current_signal: float = 0.5

    def reset(self) -> None:
        self.prev_state = None
        self.prev_reward = None
        self.current_signal = 0.5

    @property
    def value(self) -> float:
        return self.current_signal

    def observe(self, state: np.ndarray, reward: float, uncertainty: float) -> float:
        delta_state = 0.0
        if self.prev_state is not None:
            denom = np.linalg.norm(self.prev_state) + 1e-6
            delta = np.linalg.norm(state - self.prev_state) / denom
            delta_state = float(np.tanh(delta))
        delta_reward = 0.0
        if self.prev_reward is not None:
            delta_reward = float(np.tanh(abs(float(reward) - self.prev_reward)))
        raw = (
            self.coeffs.alpha_state * delta_state
            + self.coeffs.alpha_reward * delta_reward
            + self.coeffs.alpha_uncertainty * float(uncertainty)
        )
        self.current_signal = float(1.0 / (1.0 + np.exp(-raw)))
        self.prev_state = state.copy()
        self.prev_reward = float(reward)
        return self.current_signal
