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
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def explain_risk_vector(vec: np.ndarray, names: List[str], top_k: int = 5) -> List[Dict[str, float]]:
    vec = np.asarray(vec)
    idx = np.argsort(np.abs(vec))[::-1][:top_k]
    return [{"feature": names[i], "value": float(vec[i])} for i in idx]


class DecisionLogger:
    def __init__(self, log_dir: str | Path, feature_cfg: FeatureConfig):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "decisions.jsonl"
        self.feature_cfg = feature_cfg
        self._fp = self.path.open("a", encoding="utf-8")

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
    ) -> None:
        record = {
            "step": step,
            "date": date,
            "beta": beta,
            "action": action.tolist(),
            "reward": reward,
            "risk_metric": risk_metric,
            "stress_signal": stress_signal,
            "explanations": explain_risk_vector(risk_features, self.feature_cfg.risk_feature_names),
        }
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()
