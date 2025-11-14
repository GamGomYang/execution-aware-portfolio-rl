"""Utility helpers for technical indicators and risk statistics."""

from __future__ import annotations

import numpy as np


def rolling_rsi(series: np.ndarray, period: int = 14) -> float:
    gains = np.clip(series[1:] - series[:-1], a_min=0.0, a_max=None)
    losses = np.clip(series[:-1] - series[1:], a_min=0.0, a_max=None)
    avg_gain = gains[-period:].mean() + 1e-8
    avg_loss = losses[-period:].mean() + 1e-8
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    cutoff = int((1 - alpha) * len(values))
    cutoff = max(cutoff, 1)
    tail = np.sort(values)[:cutoff]
    return float(tail.mean())
