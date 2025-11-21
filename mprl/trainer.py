"""Training harness tying environment, agent, and buffer together."""

from __future__ import annotations

from collections import deque
import csv
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
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
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    data_dir: str = "data",
    log_dir: str = "logs",
    cache: bool = True,
    checkpoint_path: str | None = None,
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
    reward_scale = 100.0
    log_dir_path = Path(log_dir)
    metrics_path = log_dir_path / "training_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fp = metrics_path.open("w", newline="")
    metrics_writer = csv.writer(metrics_fp)
    metrics_writer.writerow(
        [
            "step",
            "actor_loss",
            "critic_loss",
            "meta_penalty",
            "beta_mean",
            "beta_instant",
            "stress",
            "reward_raw",
            "plasticity",
        ]
    )
    metrics_fp.flush()

    def _attach_grad_clip(optimizer: torch.optim.Optimizer, params, tag: str) -> None:
        if getattr(optimizer, "_clip_wrapped", False):
            return
        orig_step = optimizer.step

        def step_with_clip(*args, **kwargs):
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return orig_step(*args, **kwargs)

        optimizer.step = step_with_clip  # type: ignore
        optimizer._clip_wrapped = True  # type: ignore

    _attach_grad_clip(
        agent.actor_opt,
        list(agent.asset_actor.parameters()) + list(agent.sector_actor.parameters()),
        "actor",
    )
    _attach_grad_clip(
        agent.critic_opt,
        list(agent.critic1.parameters()) + list(agent.critic2.parameters()),
        "critic",
    )
    _attach_grad_clip(agent.meta_opt, agent.risk_module.parameters(), "meta")
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
    diag_window: Deque[Tuple[float, float, float, float, float]] = deque(maxlen=1000)
    prev_executed: np.ndarray | None = None

    for step in range(train_cfg.total_steps):
        action, beta_val, hebb_state, next_hebb_state = agent.select_action(
            state,
            risk_feat,
            signal,
            plasticity_signal=current_plasticity,
        )
        next_state, reward, risk_metric, next_signal, info = env.step(action)
        done = info.get("done", False)
        executed_action = info.get("executed_action", action)
        next_plasticity = plasticity_ctrl.observe(next_state, reward, next_signal)
        turnover = 0.0
        if prev_executed is not None:
            turnover = float(np.abs(executed_action - prev_executed).sum())
        prev_executed = executed_action
        stress_now = float(info.get("stress_now", info.get("signal", float(next_signal))))
        homeostat_val = float(agent.risk_module.beta_center.item())
        diag_window.append((float(beta_val), stress_now, turnover, float(reward), homeostat_val))
        buffer.push(
            state,
            executed_action,
            reward * reward_scale,
            risk_metric,
            done,
            next_state,
            risk_feat,
            info["risk_features"],
            hebb_state,
            next_hebb_state,
            current_plasticity,
            next_plasticity,
            stress_value=stress_now,
            next_stress_value=float(next_signal),
            beta_value=float(beta_val),
        )

        decision_logger.log_step(
            step=step,
            date=info.get("date", f"step_{step}"),
            beta=beta_val,
            action=executed_action,
            reward=reward,
            risk_metric=risk_metric,
            stress_signal=stress_now,
            risk_features=info["risk_features"],
            plasticity=current_plasticity,
            portfolio_return=float(info.get("portfolio_return", reward)),
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
        if step % 500 == 0:
            print(
                f"[trainer] step={step} beta={float(beta_val):.6f} stress={stress_now:.6f} homeostat={homeostat_val:.6f}",
                flush=True,
            )
        if step % 1000 == 0 and step > 0 and diag_window:
            diag_arr = np.asarray(diag_window)
            beta_vals = diag_arr[:, 0]
            stresses = diag_arr[:, 1]
            turnover_vals = diag_arr[:, 2]
            rewards = diag_arr[:, 3]
            homeostats = diag_arr[:, 4]
            if beta_vals.std() > 1e-6 and stresses.std() > 1e-6:
                corr = float(np.corrcoef(beta_vals, stresses)[0, 1])
            else:
                corr = 0.0
            msg = (
                f"[trainer] step {step}/{train_cfg.total_steps} "
                f"beta_mean={beta_vals.mean():.3f} corr(beta,stress)={corr:.3f} "
                f"reward_mean={rewards.mean():.4f} turnover_mean={turnover_vals.mean():.3f} "
                f"homeostat_mean={homeostats.mean():.3f}"
            )
            print(msg, flush=True)

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

    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        agent.save_policy(str(checkpoint_file))
        logger.info("Saved policy checkpoint to %s", checkpoint_file)

    metrics_fp.close()


def run_mock_training(steps: int = 512) -> None:
    run_training(steps=steps, env_type="stub")


def parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(description="Train MPRL agent.")
    parser.add_argument("--steps", type=int, default=512, help="Training steps")
    parser.add_argument("--env-type", type=str, choices=("real", "stub"), default="stub", help="Environment type")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Dataset start date (real env)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="Dataset end date (real env)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data/cache directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Training log directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint output path")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset cache reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        steps=args.steps,
        env_type=args.env_type,
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        cache=not args.no_cache,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
            metrics_writer.writerow(
                [
                    step,
                    stats["actor_loss"],
                    stats["critic_loss"],
                    stats["meta_penalty"],
                    stats["beta_mean"],
                    float(beta_val),
                    stress_now,
                    float(reward),
                    float(current_plasticity),
                ]
            )
            metrics_fp.flush()
