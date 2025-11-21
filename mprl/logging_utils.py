"""Logging helpers for debugging/XAI during training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import FeatureConfig


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mprl.log"
    logger = logging.getLogger("mprl")
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def explain_risk_vector(vec: np.ndarray, names: List[str], top_k: int = 5) -> List[Dict[str, float]]:
    arr = np.asarray(vec, dtype=np.float32)
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(np.abs(safe))[::-1][:top_k]
    return [{"feature": names[i], "value": float(safe[i])} for i in idx]


class DecisionLogger:
    def __init__(self, log_dir: str | Path, feature_cfg: FeatureConfig):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "decisions.jsonl"
        self.raw_features_path = self.log_dir / "raw_features.jsonl"
        self.feature_cfg = feature_cfg
        self._fp = self.path.open("w", encoding="utf-8")
        self._raw_fp = self.raw_features_path.open("w", encoding="utf-8")

    def log_step(
        self,
        *,
        step: int,
        date: str,
        beta: float,
        action: np.ndarray,
        reward: float,
        risk_metric: float,
        stress_signal: float,
        risk_features: np.ndarray,
        plasticity: float,
        portfolio_return: float,
    ) -> None:
        risk_array = np.asarray(risk_features, dtype=np.float32).copy()
        safe_risk = np.nan_to_num(risk_array, nan=0.0, posinf=0.0, neginf=0.0)
        record = {
            "step": step,
            "date": date,
            "beta": beta,
            "action": action.tolist(),
            "reward": reward,
            "risk_metric": risk_metric,
            "stress_signal": stress_signal,
            "plasticity": plasticity,
            "explanations": explain_risk_vector(safe_risk, self.feature_cfg.risk_feature_names),
            "portfolio_return": portfolio_return,
        }
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()
        reason = None
        if not np.isfinite(risk_array).all():
            reason = "nan_in_features"
        elif np.allclose(safe_risk, 0.0, atol=1e-8):
            reason = "zero_vector"
        if reason is not None:
            self._log_raw_features(
                step=step,
                date=date,
                beta=beta,
                stress_signal=stress_signal,
                risk_vector=risk_array,
                reason=reason,
            )

    def _log_raw_features(
        self,
        *,
        step: int,
        date: str,
        beta: float,
        stress_signal: float,
        risk_vector: np.ndarray,
        reason: str,
    ) -> None:
        payload = {
            "step": step,
            "date": date,
            "beta": float(beta),
            "stress_signal": float(stress_signal),
            "reason": reason,
            "risk_features": np.asarray(risk_vector, dtype=np.float32).tolist(),
        }
        self._raw_fp.write(json.dumps(payload) + "\n")
        self._raw_fp.flush()

    def close(self) -> None:
        self._fp.close()
        self._raw_fp.close()
