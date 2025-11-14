"""Training harness tying environment, agent, and buffer together."""

from __future__ import annotations

from typing import Dict, List

import torch

from .agent import MPRLAgent
from .config import FeatureConfig, TrainingConfig
from .env_stub import PortfolioEnvStub
from .replay_buffer import ReplayBuffer


def run_mock_training(steps: int = 512) -> None:
    feature_cfg = FeatureConfig()
    train_cfg = TrainingConfig(total_steps=steps)
    env = PortfolioEnvStub(feature_cfg, train_cfg)
    agent = MPRLAgent(feature_cfg, train_cfg)
    buffer = ReplayBuffer(
        train_cfg.replay_capacity,
        feature_cfg.total_state_dim,
        feature_cfg.num_assets,
        feature_cfg.risk_feature_dim,
        feature_cfg.hebbian_dim,
    )

    state, risk_feat, signal = env.reset()
    stats_history: List[Dict[str, float]] = []

    for step in range(train_cfg.total_steps):
        action, beta_val, hebb_state, next_hebb_state = agent.select_action(state, risk_feat, signal)
        next_state, reward, risk_metric, next_signal, info = env.step(action)
        buffer.push(
            state,
            action,
            reward,
            risk_metric,
            False,
            next_state,
            risk_feat,
            info["risk_features"],
            hebb_state,
            next_hebb_state,
        )

        state = next_state
        risk_feat = info["risk_features"]
        signal = next_signal

        if buffer.size > train_cfg.warmup_steps and step % train_cfg.update_every == 0:
            batch = buffer.sample(train_cfg.batch_size, torch.device(train_cfg.device))
            stats = agent.update(batch)
            stats_history.append(stats)
            if step % 50 == 0:
                print(
                    f"step={step} critic={stats['critic_loss']:.4f} actor={stats['actor_loss']:.4f} "
                    f"meta={stats['meta_penalty']:.4f} beta={stats['beta_mean']:.3f}"
                )

    if stats_history:
        last = stats_history[-1]
        print(
            "Final stats: "
            f"critic={last['critic_loss']:.4f}, actor={last['actor_loss']:.4f}, "
            f"meta={last['meta_penalty']:.4f}, beta={last['beta_mean']:.3f}"
        )
    else:
        print("Buffer did not reach warmup size; no training stats to report.")
