from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.step6_sanity import build_eval_context, run_eval_case


ALLOWED_ENV_KEYS = {
    "transaction_cost",
    "risk_lambda",
    "eta_mode",
    "rebalance_eta",
    "rule_vol",
    "logit_scale",
    "L",
    "Lv",
}
ALLOWED_RULE_VOL_KEYS = {"window", "a", "eta_clip"}
ALLOWED_ETA_MODES = {"legacy", "none", "fixed", "rule_vol"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6 production matrix runner.")
    parser.add_argument("--config", type=str, default="configs/step6_main.yaml", help="Config YAML path.")
    parser.add_argument("--kappas", nargs="+", type=float, default=[0.0, 0.0005, 0.001], help="Kappa grid.")
    parser.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02],
        help="Eta grid for fixed execution smoothing.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Paired seed list.")
    parser.add_argument("--out", type=str, default="outputs/step6", help="Output root.")
    parser.add_argument("--model-type", choices=["baseline", "prl"], default="prl", help="Model type.")
    parser.add_argument("--model-root", type=str, default="outputs", help="Root containing reports/models.")
    parser.add_argument("--model-path", type=str, help="Optional explicit model path (single-seed only).")
    parser.add_argument("--offline", action="store_true", help="Use cached data without downloading.")
    parser.add_argument("--max-steps", type=int, default=252, help="Max eval steps per run.")
    return parser.parse_args()


def _format_kappa(value: float) -> str:
    return f"{value:g}"


def _format_eta(value: float) -> str:
    return f"{value:g}"


def _compute_sharpe(returns: pd.Series) -> float:
    arr = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = float(arr.std(ddof=0))
    if std <= 1e-8:
        return 0.0
    return float((arr.mean() / std) * np.sqrt(252.0))


def _compute_max_drawdown(equity: pd.Series) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    run_max = np.maximum.accumulate(arr)
    dd = arr / run_max - 1.0
    return float(np.min(dd))


def _compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    final = float(arr[-1])
    if final <= 0.0:
        return float("nan")
    years = float(arr.size) / float(periods_per_year)
    if years <= 0.0:
        return 0.0
    return float(final ** (1.0 / years) - 1.0)


def _normalize_env_keys(env_cfg: dict[str, Any]) -> set[str]:
    normalized = set()
    for key in env_cfg.keys():
        if key == "c_tc":
            normalized.add("transaction_cost")
        else:
            normalized.add(key)
    return normalized


def _freeze_guard(cfg: dict[str, Any]) -> None:
    env_cfg = cfg.get("env", {}) or {}
    normalized_keys = _normalize_env_keys(env_cfg)

    suspicious = []
    for key in normalized_keys:
        if key in ALLOWED_ENV_KEYS:
            continue
        if key.startswith("eta") or key.startswith("rule_vol") or key in {
            "rebalance_eta",
            "risk_lambda",
            "transaction_cost",
            "logit_scale",
            "L",
            "Lv",
        }:
            suspicious.append(key)
    if suspicious:
        raise ValueError(f"Config freeze guard failed. Unexpected Step6-related env keys: {sorted(set(suspicious))}")

    missing_required = [k for k in ("L", "Lv", "logit_scale") if k not in normalized_keys]
    if missing_required:
        raise ValueError(f"Config freeze guard failed. Missing required env keys: {missing_required}")
    if "transaction_cost" not in normalized_keys:
        raise ValueError("Config freeze guard failed. env.transaction_cost (or env.c_tc) is required.")

    eta_mode = str(env_cfg.get("eta_mode", "legacy"))
    if eta_mode not in ALLOWED_ETA_MODES:
        raise ValueError(f"Config freeze guard failed. Unsupported eta_mode: {eta_mode}")

    rule_vol = env_cfg.get("rule_vol")
    if rule_vol is not None:
        if not isinstance(rule_vol, dict):
            raise ValueError("Config freeze guard failed. env.rule_vol must be a mapping.")
        unexpected_rule_keys = sorted(set(rule_vol.keys()) - ALLOWED_RULE_VOL_KEYS)
        if unexpected_rule_keys:
            raise ValueError(f"Config freeze guard failed. Unexpected env.rule_vol keys: {unexpected_rule_keys}")
        eta_clip = rule_vol.get("eta_clip")
        if eta_clip is not None:
            if not isinstance(eta_clip, (list, tuple)) or len(eta_clip) != 2:
                raise ValueError("Config freeze guard failed. env.rule_vol.eta_clip must be [min, max].")


def _ensure_paired_seeds(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return list(seeds)


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.exists():
        return path

    # Convenience fallback for Step6 production command examples.
    if path.name == "step6_main.yaml":
        fallback = ROOT / "configs" / "prl_100k.yaml"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Config file not found: {config_path}")


def _run_sanity(args: argparse.Namespace, first_seed: int, config_path: Path) -> None:
    sanity_script = ROOT / "scripts" / "step6_sanity.py"
    cmd = [
        sys.executable,
        str(sanity_script),
        "--config",
        str(config_path),
        "--model-type",
        args.model_type,
        "--seed",
        str(first_seed),
        "--model-root",
        args.model_root,
        "--max-steps",
        str(args.max_steps),
    ]
    if args.offline:
        cmd.append("--offline")
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    seeds = _ensure_paired_seeds([int(s) for s in args.seeds])
    kappas = [float(k) for k in args.kappas]
    etas = [float(e) for e in args.etas]
    if not etas:
        raise ValueError("etas must not be empty")
    if any((e <= 0.0 or e > 1.0) for e in etas):
        raise ValueError(f"etas must be in (0, 1], got: {etas}")

    config_path = _resolve_config_path(args.config)
    base_cfg = yaml.safe_load(config_path.read_text())
    _freeze_guard(base_cfg)

    # A) Run sanity first (stop entire run on failure).
    _run_sanity(args, seeds[0], config_path)

    # Eval-only frontier: keep one trained model fixed and vary execution layer / env seed.
    model_probe_ctx = build_eval_context(
        config_path=str(config_path),
        model_type=args.model_type,
        seed=seeds[0],
        model_root=args.model_root,
        offline=bool(args.offline),
        max_steps=int(args.max_steps),
        model_path_arg=args.model_path,
        prefer_metadata_config=True,
    )
    shared_model_path = str(model_probe_ctx.model_path)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    seeds_by_arm: dict[tuple[float, float], set[int]] = {}
    for kappa in kappas:
        kappa_dir = out_root / f"kappa_{_format_kappa(kappa)}"
        kappa_dir.mkdir(parents=True, exist_ok=True)
        for eta in etas:
            eta_dir = kappa_dir / f"eta_{_format_eta(eta)}"
            eta_dir.mkdir(parents=True, exist_ok=True)
            done: set[int] = set()

            for seed in seeds:
                run_cfg = copy.deepcopy(base_cfg)
                run_cfg.setdefault("env", {})
                run_cfg["env"]["transaction_cost"] = float(kappa)
                run_cfg["env"]["c_tc"] = float(kappa)
                run_cfg["env"]["eta_mode"] = "fixed"
                run_cfg["env"]["rebalance_eta"] = float(eta)

                seed_dir = eta_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                cfg_out_path = seed_dir / "config.yaml"
                cfg_out_path.write_text(yaml.safe_dump(run_cfg, sort_keys=True, allow_unicode=False))

                cmd_text = (
                    f"{sys.executable} scripts/step6_run_matrix.py --config {config_path} "
                    f"--kappas {' '.join(str(x) for x in kappas)} --etas {' '.join(str(x) for x in etas)} "
                    f"--seeds {' '.join(str(x) for x in seeds)} --out {args.out} "
                    f"--model-type {args.model_type} --model-root {args.model_root} "
                    f"--max-steps {args.max_steps} --offline {args.offline} "
                    f"[per-run kappa={kappa}, eta={eta}, seed={seed}]"
                )
                (seed_dir / "cmd.txt").write_text(cmd_text + "\n")

                ctx = build_eval_context(
                    config_path=str(cfg_out_path),
                    model_type=args.model_type,
                    seed=seed,
                    model_root=args.model_root,
                    offline=bool(args.offline),
                    max_steps=int(args.max_steps),
                    model_path_arg=shared_model_path,
                    prefer_metadata_config=False,
                )

                metrics, trace, df, current_sig = run_eval_case(
                    ctx,
                    eta_mode="fixed",
                    rebalance_eta=float(eta),
                    transaction_cost=float(kappa),
                    eval_tag=f"kappa_{_format_kappa(kappa)}__eta_{_format_eta(eta)}__seed_{seed}",
                )

                trace_path = seed_dir / "trace.parquet"
                df.to_parquet(trace_path, index=False)

                env_signature_payload = {
                    "expected_env_signature_hash": (ctx.run_meta or {}).get("env_signature_hash"),
                    "expected_env_signature_version": (ctx.run_meta or {}).get("env_signature_version"),
                    "current": current_sig,
                }
                (seed_dir / "env_signature.json").write_text(json.dumps(env_signature_payload, indent=2))

                sharpe_net_lin = (
                    float(metrics.sharpe_net_lin) if metrics.sharpe_net_lin is not None else _compute_sharpe(df["net_return_lin"])
                )
                cagr = _compute_cagr(df["equity_net_lin"])
                maxdd = (
                    float(metrics.max_drawdown_net_lin)
                    if metrics.max_drawdown_net_lin is not None
                    else _compute_max_drawdown(df["equity_net_lin"])
                )

                turnover_exec_mean = float(pd.to_numeric(df["turnover_exec"], errors="coerce").mean())
                turnover_target_mean = float(pd.to_numeric(df["turnover_target"], errors="coerce").mean())
                tracking_error_mean = float(pd.to_numeric(df["tracking_error_l2"], errors="coerce").mean())
                misalignment_gap_mean = float(
                    (
                        pd.to_numeric(df["net_return_lin"], errors="coerce")
                        - pd.to_numeric(df["net_return_lin_target"], errors="coerce")
                    ).mean()
                )
                collapse_any = bool(df.get("collapse_flag", pd.Series(False, index=df.index)).astype(bool).any())
                collapse_count = int(df.get("collapse_flag", pd.Series(False, index=df.index)).astype(bool).sum())
                eta_value = float(pd.to_numeric(df.get("eta_t", pd.Series([np.nan])), errors="coerce").mean())

                metrics_row = pd.DataFrame(
                    [
                        {
                            "kappa": float(kappa),
                            "seed": int(seed),
                            "run_id": ctx.run_id,
                            "model_type": ctx.model_type,
                            "eta_mode": "fixed",
                            "eta_requested": float(eta),
                            "eta": eta_value,
                            "n_steps": int(len(df)),
                            "sharpe_net_lin": sharpe_net_lin,
                            "cagr": cagr,
                            "maxdd": maxdd,
                            "avg_turnover_exec": turnover_exec_mean,
                            "avg_turnover_target": turnover_target_mean,
                            "tracking_error_l2_mean": tracking_error_mean,
                            "misalignment_gap_mean": misalignment_gap_mean,
                            "collapse_flag_any": collapse_any,
                            "collapse_count": collapse_count,
                            "trace_path": str(trace_path),
                        }
                    ]
                )
                metrics_row.to_csv(seed_dir / "metrics.csv", index=False)

                done.add(int(seed))

            seeds_by_arm[(float(kappa), float(eta))] = done

    target = set(seeds)
    for (kappa, eta), got in seeds_by_arm.items():
        if got != target:
            raise RuntimeError(
                f"Paired seed enforcement failed for kappa={kappa}, eta={eta}: "
                f"expected={sorted(target)}, got={sorted(got)}"
            )

    build_report_cmd = [sys.executable, str(ROOT / "scripts" / "step6_build_reports.py"), "--root", str(out_root)]
    subprocess.run(build_report_cmd, check=True)


if __name__ == "__main__":
    main()
