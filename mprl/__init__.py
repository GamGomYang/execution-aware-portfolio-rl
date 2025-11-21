"""MPRL package, exposing key configs and lazy entry points."""

from __future__ import annotations

from .config import FeatureConfig, TrainingConfig


def run_training(*args, **kwargs):
    from .trainer import run_training as _run_training

    return _run_training(*args, **kwargs)


def run_mock_training(*args, **kwargs):
    from .trainer import run_mock_training as _run_mock_training

    return _run_mock_training(*args, **kwargs)


def run_backtest(*args, **kwargs):
    from .backtest import run_backtest as _run_backtest

    return _run_backtest(*args, **kwargs)


def run_evaluation(*args, **kwargs):
    from .eval import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


__all__ = [
    "FeatureConfig",
    "TrainingConfig",
    "run_training",
    "run_mock_training",
    "run_backtest",
    "run_evaluation",
]
