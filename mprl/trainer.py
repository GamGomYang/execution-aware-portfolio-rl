"""Training harness tying environment, agent, and buffer together."""

from __future__ import annotations

from typing import Dict, List

import torch

from .agent import MPRLAgent
from .config import FeatureConfig, TrainingConfig
from .data_pipeline import MarketDataLoader
from .env_real import YFinancePortfolioEnv
from .env_stub import PortfolioEnvStub
from .logging_utils import DecisionLogger, setup_logging
from .plasticity import PlasticityCoefficients, PlasticityController
from .replay_buffer import ReplayBuffer


def run_training(
    *,
    steps: int = 512,
    env_type: str = "stub",
    start: str = "2005-01-01",
    end: str = "2024-01-01",
    data_dir: str = "data",
    log_dir: str = "logs",
    cache: bool = True,
) -> None:
    feature_cfg = FeatureConfig()
    train_cfg = TrainingConfig(total_steps=steps)
    logger = setup_logging(log_dir)
    decision_logger = DecisionLogger(log_dir, feature_cfg)

    if env_type == "real":
        loader = MarketDataLoader(feature_cfg, start=start, end=end, data_dir=data_dir)
        dataset = loader.load(cache=cache)
        env = YFinancePortfolioEnv(feature_cfg, train_cfg, dataset)
        logger.info(
            "Loaded real market data from %s to %s (%d samples)",
            start,
            end,
            dataset.length,
        )
    else:
        env = PortfolioEnvStub(feature_cfg, train_cfg)
        logger.info("Using stochastic stub environment")

    agent = MPRLAgent(feature_cfg, train_cfg)
    buffer = ReplayBuffer(
        train_cfg.replay_capacity,
        feature_cfg.total_state_dim,
        feature_cfg.num_assets,
        feature_cfg.risk_feature_dim,
        feature_cfg.hebbian_dim,
    )

    state, risk_feat, signal = env.reset()
    plasticity_ctrl = PlasticityController(
        PlasticityCoefficients(
            train_cfg.plasticity_alpha_state,
            train_cfg.plasticity_alpha_reward,
            train_cfg.plasticity_alpha_uncertainty,
        )
    )
    plasticity_ctrl.reset()
    current_plasticity = plasticity_ctrl.observe(state, reward=0.0, uncertainty=signal)
    stats_history: List[Dict[str, float]] = []

    for step in range(train_cfg.total_steps):
        action, beta_val, hebb_state, next_hebb_state = agent.select_action(
            state,
            risk_feat,
            signal,
            plasticity_signal=current_plasticity,
        )
        next_state, reward, risk_metric, next_signal, info = env.step(action)
        done = info.get("done", False)
        next_plasticity = plasticity_ctrl.observe(next_state, reward, next_signal)
        buffer.push(
            state,
            action,
            reward,
            risk_metric,
            done,
            next_state,
            risk_feat,
            info["risk_features"],
            hebb_state,
            next_hebb_state,
            current_plasticity,
            next_plasticity,
        )

        decision_logger.log_step(
            step=step,
            date=info.get("date", f"step_{step}"),
            beta=beta_val,
            action=action,
            reward=reward,
            risk_metric=risk_metric,
            stress_signal=info.get("signal", float(next_signal)),
            risk_features=info["risk_features"],
            plasticity=current_plasticity,
        )

        if done:
            plasticity_ctrl.reset()
            state, risk_feat, signal = env.reset()
            current_plasticity = plasticity_ctrl.observe(state, reward=0.0, uncertainty=signal)
            logger.info("Episode reset at step %d", step)
            continue

        state = next_state
        risk_feat = info["risk_features"]
        signal = next_signal
        current_plasticity = next_plasticity

        if buffer.size > train_cfg.warmup_steps and step % train_cfg.update_every == 0:
            batch = buffer.sample(train_cfg.batch_size, torch.device(train_cfg.device))
            stats = agent.update(batch)
            stats_history.append(stats)
            if step % 50 == 0:
                logger.info(
                    "step=%d critic=%.4f actor=%.4f meta=%.4f beta=%.3f",
                    step,
                    stats["critic_loss"],
                    stats["actor_loss"],
                    stats["meta_penalty"],
                    stats["beta_mean"],
                )

    decision_logger.close()
    if stats_history:
        last = stats_history[-1]
        logger.info(
            "Final stats critic=%.4f actor=%.4f meta=%.4f beta=%.3f",
            last["critic_loss"],
            last["actor_loss"],
            last["meta_penalty"],
            last["beta_mean"],
        )
    else:
        logger.warning("Buffer did not reach warmup size; no training stats to report.")


def run_mock_training(steps: int = 512) -> None:
    run_training(steps=steps, env_type="stub")
