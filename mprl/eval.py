"""Evaluation (inference) runner for out-of-sample backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import MPRLAgent
from .backtest import run_backtest
from .config import FeatureConfig, TrainingConfig
from .data_pipeline import MarketDataLoader
from .env_real import YFinancePortfolioEnv
from .logging_utils import DecisionLogger, setup_logging
from .plasticity import PlasticityCoefficients, PlasticityController


def run_evaluation(
    *,
    start: str,
    end: str,
    data_dir: str | Path,
    log_dir: str | Path,
    checkpoint_path: str | Path,
    steps: int | None = None,
    deterministic: bool = False,
    cache: bool = True,
) -> Path:
    feature_cfg = FeatureConfig()
    train_cfg = TrainingConfig(total_steps=steps or 0)
    logger = setup_logging(log_dir)

    loader = MarketDataLoader(feature_cfg, start=start, end=end, data_dir=data_dir)
    dataset = loader.load(cache=cache)
    env = YFinancePortfolioEnv(feature_cfg, train_cfg, dataset)

    agent = MPRLAgent(feature_cfg, train_cfg)
    agent.load_policy(str(checkpoint_path))
    logger.info("Loaded checkpoint %s", checkpoint_path)

    decision_logger = DecisionLogger(log_dir, feature_cfg)
    plasticity_ctrl = PlasticityController(
        PlasticityCoefficients(
            train_cfg.plasticity_alpha_state,
            train_cfg.plasticity_alpha_reward,
            train_cfg.plasticity_alpha_uncertainty,
        )
    )
    plasticity_ctrl.reset()

    state, risk_feat, signal = env.reset()
    current_plasticity = plasticity_ctrl.observe(state, reward=0.0, uncertainty=signal)

    max_steps = steps if steps is not None else dataset.length
    completed_steps = 0

    for step in range(max_steps):
        action, beta_val, hebb_state, next_hebb_state = agent.select_action(
            state,
            risk_feat,
            signal,
            plasticity_signal=current_plasticity,
            deterministic=deterministic,
        )
        next_state, reward, risk_metric, next_signal, info = env.step(action)
        done = info.get("done", False)
        executed_action = info.get("executed_action", action)
        next_plasticity = plasticity_ctrl.observe(next_state, reward, next_signal)

        decision_logger.log_step(
            step=step,
            date=info.get("date", f"eval_{step}"),
            beta=beta_val,
            action=executed_action,
            reward=reward,
            risk_metric=risk_metric,
            stress_signal=info.get("stress_now", info.get("signal", float(next_signal))),
            risk_features=info["risk_features"],
            plasticity=current_plasticity,
            portfolio_return=float(info.get("portfolio_return", reward)),
        )

        completed_steps += 1
        if done:
            break
        state = next_state
        risk_feat = info["risk_features"]
        signal = next_signal
        current_plasticity = next_plasticity

    decision_logger.close()
    logger.info("Evaluation finished after %d steps (%s to %s).", completed_steps, start, end)
    return decision_logger.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation-only rollout with a saved policy.")
    parser.add_argument("--start", type=str, required=True, help="Evaluation start date")
    parser.add_argument("--end", type=str, required=True, help="Evaluation end date")
    parser.add_argument("--steps", type=int, default=None, help="Maximum rollout steps (defaults to dataset length)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for cached data")
    parser.add_argument("--log-dir", type=str, default="logs/eval", help="Directory for evaluation logs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    parser.add_argument("--no-cache", action="store_true", help="Disable cached dataset reload")
    parser.add_argument("--report", action="store_true", help="Run backtest metrics after evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = run_evaluation(
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        deterministic=args.deterministic,
        cache=not args.no_cache,
    )
    if args.report:
        metrics = run_backtest(
            start=args.start,
            end=args.end,
            data_dir=args.data_dir,
            log_path=log_path,
            cache=not args.no_cache,
        )
        print("Out-of-sample metrics:")
        for key, value in metrics.items():
            if "ratio" in key or "frequency" in key or key == "win_rate":
                print(f"- {key}: {value:.4f}")
            elif "vol" in key or "deviation" in key or "turnover" in key or key == "stability":
                print(f"- {key}: {value:.4f}")
            else:
                print(f"- {key}: {value:.4%}")


if __name__ == "__main__":
    main()
