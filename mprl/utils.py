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


def safe_corrcoef(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute a correlation matrix that is numerically stable for near-constant columns."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < 2:
        size = arr.shape[1]
        return np.eye(size, dtype=np.float64)

    cov = np.cov(arr, rowvar=False)
    if cov.ndim == 0:
        return np.eye(1, dtype=np.float64)

    diag = np.diag(cov)
    scale = np.sqrt(np.maximum(diag, eps))
    denom = scale[:, None] * scale[None, :]
    corr = cov / denom
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr += eps * np.eye(corr.shape[0], dtype=np.float64)
    return corr
