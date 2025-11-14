"""MPRL package, exposing key configs and trainer entry points."""

from .config import FeatureConfig, TrainingConfig
from .trainer import run_mock_training, run_training

__all__ = ["FeatureConfig", "TrainingConfig", "run_training", "run_mock_training"]
