import argparse
import csv
import logging
import json
import hashlib
import shutil
from pathlib import Path

import yaml

from prl.eval import assert_env_compatible, load_model, run_backtest_episode
from prl.train import build_env_for_range, build_signal_features, create_scheduler, prepare_market_and_features


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SAC/PRL models on 2022-2025 backtest.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model-type", choices=["baseline", "prl"], required=True, help="Model variant to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed identifier used during training.")
    parser.add_argument("--model-path", type=str, help="Optional explicit model path.")
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Base directory for reports/models/logs (default: outputs).",
    )
    return parser.parse_args()


def write_metrics(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "eval_id",
        "model_type",
        "seed",
        "period",
        "total_reward",
        "avg_reward",
        "cumulative_return",
        "cumulative_return_net_exp",
        "cumulative_return_net_lin",
        "avg_turnover",
        "total_turnover",
        "avg_turnover_exec",
        "total_turnover_exec",
        "avg_turnover_target",
        "total_turnover_target",
        "sharpe",
        "sharpe_net_exp",
        "sharpe_net_lin",
        "max_drawdown",
        "max_drawdown_net_exp",
        "max_drawdown_net_lin",
        "mean_daily_return_gross",
        "std_daily_return_gross",
        "mean_daily_net_return_exp",
        "std_daily_net_return_exp",
        "mean_daily_net_return_lin",
        "std_daily_net_return_lin",
        "mean_daily_return_gross_mid",
        "std_daily_return_gross_mid",
        "mean_daily_net_return_exp_mid",
        "std_daily_net_return_exp_mid",
        "mean_daily_net_return_lin_mid",
        "std_daily_net_return_lin_mid",
        "steps",
    ]
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _archive_config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if k != "config_path"}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _archive_exp_id(config_path: str, cfg: dict) -> str:
    return f"{Path(config_path).stem}__{_archive_config_hash(cfg)}"


def _next_archive_path(archive_dir: Path, prefix: str, exp_id: str) -> Path:
    candidate = archive_dir / f"{prefix}_{exp_id}.csv"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = archive_dir / f"{prefix}_{exp_id}__{idx}.csv"
        if not candidate.exists():
            return candidate
        idx += 1


def _archive_reports(reports_dir: Path, *, config_path: str, cfg: dict) -> dict[str, str]:
    archive_dir = reports_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    exp_id = _archive_exp_id(config_path, cfg)
    written: dict[str, str] = {}
    targets = {
        "metrics": reports_dir / "metrics.csv",
        "summary": reports_dir / "summary.csv",
        "regime_metrics": reports_dir / "regime_metrics.csv",
    }
    for prefix, src in targets.items():
        if not src.exists():
            continue
        dst = _next_archive_path(archive_dir, prefix, exp_id)
        shutil.copy2(src, dst)
        written[prefix] = str(dst)
    return written


def _load_latest_run_metadata(model_type: str, seed: int, reports_dir: Path) -> dict | None:
    if not reports_dir.exists():
        return None
    candidates = []
    for path in reports_dir.glob("run_metadata_*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("model_type") != model_type or int(data.get("seed", -1)) != seed:
            continue
        created_at = data.get("created_at")
        try:
            created_ts = created_at or ""
        except Exception:
            created_ts = ""
        candidates.append((created_ts, data))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _load_run_metadata_for_model(model_path: Path, reports_dir: Path) -> dict | None:
    run_id = model_path.stem
    if run_id.endswith("_final"):
        run_id = run_id[: -len("_final")]
    meta_path = reports_dir / f"run_metadata_{run_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["config_path"] = args.config
    dates = cfg["dates"]
    env_cfg = cfg["env"]
    prl_cfg = cfg.get("prl", {})
    data_cfg = cfg.get("data", {})
    if data_cfg.get("paper_mode", False) and not data_cfg.get("require_cache", False):
        raise ValueError("paper_mode=true requires require_cache=true.")
    paper_mode = data_cfg.get("paper_mode", False)
    require_cache_cfg = data_cfg.get("require_cache", False)
    offline_cfg = data_cfg.get("offline", False)
    offline = args.offline or offline_cfg or paper_mode or require_cache_cfg
    require_cache = require_cache_cfg or paper_mode or offline
    cache_only = paper_mode or require_cache_cfg or offline_cfg or args.offline
    session_opts = data_cfg.get("session_opts", None)

    market, features = prepare_market_and_features(
        config=cfg,
        lv=env_cfg["Lv"],
        force_refresh=data_cfg.get("force_refresh", True),
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        cache_only=cache_only,
        session_opts=session_opts,
    )
    signal_features, _ = build_signal_features(market, config=cfg)

    if "logit_scale" not in env_cfg or env_cfg["logit_scale"] is None:
        raise ValueError("env.logit_scale is required for evaluation.")
    env = build_env_for_range(
        market=market,
        features=features,
        start=dates["test_start"],
        end=dates["test_end"],
        window_size=env_cfg["L"],
        c_tc=env_cfg["c_tc"],
        seed=args.seed,
        logit_scale=env_cfg["logit_scale"],
        risk_lambda=env_cfg.get("risk_lambda", 0.0),
        risk_penalty_type=env_cfg.get("risk_penalty_type", "r2"),
        rebalance_eta=env_cfg.get("rebalance_eta"),
        signal_features=signal_features,
    )

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    models_dir = output_root / "models"

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None:
        meta = _load_latest_run_metadata(args.model_type, args.seed, reports_dir)
        if meta:
            artifact_paths = meta.get("artifact_paths") or meta.get("artifacts") or {}
            model_path_value = artifact_paths.get("model_path")
            if model_path_value:
                model_path = Path(model_path_value)
        if not meta or model_path is None:
            model_path = models_dir / f"{args.model_type}_seed{args.seed}_final.zip"
    meta = _load_run_metadata_for_model(model_path, reports_dir)
    if meta is None:
        raise ValueError(f"RUN_METADATA_NOT_FOUND for model_path={model_path}")

    scheduler = None
    if args.model_type == "prl":
        num_assets = market.returns.shape[1]
        scheduler = create_scheduler(prl_cfg, env_cfg["L"], num_assets, features.stats_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first or provide --model-path to an existing *_final.zip."
        )
    assert_env_compatible(env, meta, Lv=env_cfg.get("Lv"))
    model = load_model(model_path, args.model_type, env, scheduler=scheduler)
    metrics = run_backtest_episode(model, env)
    run_id = meta.get("run_id") or model_path.stem
    if run_id.endswith("_final"):
        run_id = run_id[: -len("_final")]
    row = {
        "run_id": run_id,
        "eval_id": run_id,
        "model_type": args.model_type,
        "seed": args.seed,
        "period": "test",
        **metrics.to_dict(),
    }
    metrics_path = reports_dir / "metrics.csv"
    write_metrics(metrics_path, row)
    archived = _archive_reports(reports_dir, config_path=args.config, cfg=cfg)
    if archived:
        logging.info("Archived reports for exp_id=%s: %s", _archive_exp_id(args.config, cfg), archived)
    print(f"Backtest complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
