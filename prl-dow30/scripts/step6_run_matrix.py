from __future__ import annotations

import argparse
import copy
import hashlib
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
ALLOWED_RULE_VOL_KEYS = {"window", "a", "a_values", "eta_clip"}
ALLOWED_ETA_MODES = {"legacy", "none", "fixed", "rule_vol"}
ALLOWED_SEED_MODEL_MODES = {"independent", "shared"}


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
    parser.add_argument(
        "--seed-model-mode",
        choices=sorted(ALLOWED_SEED_MODEL_MODES),
        default="independent",
        help="Model resolution mode across seeds.",
    )
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
        a_values = rule_vol.get("a_values")
        if a_values is not None:
            if not isinstance(a_values, (list, tuple)) or len(a_values) == 0:
                raise ValueError("Config freeze guard failed. env.rule_vol.a_values must be a non-empty list.")

    baseline_env = cfg.get("baseline_env")
    if baseline_env is not None and not isinstance(baseline_env, dict):
        raise ValueError("Config freeze guard failed. baseline_env must be a mapping.")
    if isinstance(baseline_env, dict):
        for key in baseline_env.keys():
            if key not in {"eta_mode", "rebalance_eta", "rule_vol"}:
                raise ValueError(f"Config freeze guard failed. baseline_env key not allowed: {key}")
        eta_mode_base = str(baseline_env.get("eta_mode", "legacy"))
        if eta_mode_base not in ALLOWED_ETA_MODES:
            raise ValueError(f"Config freeze guard failed. Unsupported baseline_env.eta_mode: {eta_mode_base}")


def _ensure_paired_seeds(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return list(seeds)


def _flag_provided(flag: str) -> bool:
    return f"--{flag}" in sys.argv


def _resolve_scalar_from_config(args_value: Any, *, cfg_value: Any, flag: str) -> Any:
    if _flag_provided(flag):
        return args_value
    if cfg_value is not None:
        return cfg_value
    return args_value


def _resolve_list_from_config(args_values: list[Any], *, cfg_values: Any, flag: str) -> list[Any]:
    if _flag_provided(flag):
        return list(args_values)
    if cfg_values is None:
        return list(args_values)
    if not isinstance(cfg_values, (list, tuple)):
        raise ValueError(f"Config value for {flag} must be a list.")
    return list(cfg_values)


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


def _run_sanity(
    args: argparse.Namespace,
    first_seed: int,
    config_path: Path,
    *,
    model_type: str,
) -> None:
    sanity_script = ROOT / "scripts" / "step6_sanity.py"
    cmd = [
        sys.executable,
        str(sanity_script),
        "--config",
        str(config_path),
        "--model-type",
        model_type,
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


def _deep_merge_env(base_env: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base_env)
    for key, value in patch.items():
        if key == "rule_vol" and isinstance(value, dict):
            existing = out.get("rule_vol", {}) or {}
            if not isinstance(existing, dict):
                existing = {}
            merged = copy.deepcopy(existing)
            merged.update(copy.deepcopy(value))
            out["rule_vol"] = merged
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_arm_specs(cfg: dict[str, Any], *, etas: list[float]) -> list[dict[str, Any]]:
    env_cfg = cfg.get("env", {}) or {}
    baseline_env = cfg.get("baseline_env")
    rule_vol_cfg = (env_cfg.get("rule_vol", {}) or {}) if isinstance(env_cfg.get("rule_vol", {}) or {}, dict) else {}
    eta_mode = str(env_cfg.get("eta_mode", "legacy"))

    specs: list[dict[str, Any]] = []

    # EXP-1: main vs baseline paired protocol.
    if isinstance(baseline_env, dict):
        specs.append(
            {
                "dir_name": "main",
                "arm_name": "main",
                "env_patch": {},
                "eta_requested": env_cfg.get("rebalance_eta"),
            }
        )
        specs.append(
            {
                "dir_name": "baseline",
                "arm_name": "baseline",
                "env_patch": baseline_env,
                "eta_requested": baseline_env.get("rebalance_eta"),
            }
        )
        return specs

    # EXP-4: adaptive rule-vol protocol (+ fixed comparison arm).
    if eta_mode == "rule_vol" or "a_values" in rule_vol_cfg:
        a_values = rule_vol_cfg.get("a_values")
        if a_values is None:
            a_values = [rule_vol_cfg.get("a", 1.0)]
        for raw_a in list(a_values):
            a = float(raw_a)
            specs.append(
                {
                    "dir_name": f"rule_vol_a_{_format_eta(a)}",
                    "arm_name": "rule_vol",
                    "env_patch": {
                        "eta_mode": "rule_vol",
                        "rebalance_eta": None,
                        "rule_vol": {"a": a},
                    },
                    "eta_requested": np.nan,
                    "rule_vol_a": a,
                }
            )
        comparison = cfg.get("comparison", {}) or {}
        fixed_eta = comparison.get("fixed_eta")
        if fixed_eta is not None:
            fixed_eta = float(fixed_eta)
            specs.append(
                {
                    "dir_name": f"fixed_eta_{_format_eta(fixed_eta)}",
                    "arm_name": "fixed_comparison",
                    "env_patch": {
                        "eta_mode": "fixed",
                        "rebalance_eta": fixed_eta,
                    },
                    "eta_requested": fixed_eta,
                    "rule_vol_a": np.nan,
                }
            )
        return specs

    # EXP-3 and backward-compatible default: fixed eta sweep.
    for eta in etas:
        specs.append(
            {
                "dir_name": f"eta_{_format_eta(eta)}",
                "arm_name": "eta_sweep",
                "env_patch": {
                    "eta_mode": "fixed",
                    "rebalance_eta": float(eta),
                },
                "eta_requested": float(eta),
                "rule_vol_a": np.nan,
            }
        )
    return specs


def _resolve_model_path_by_seed(
    *,
    args: argparse.Namespace,
    config_path: Path,
    seeds: list[int],
    model_type: str,
    seed_model_mode: str,
) -> dict[int, str]:
    mode = str(seed_model_mode).strip().lower()
    if mode not in ALLOWED_SEED_MODEL_MODES:
        raise ValueError(
            f"Unsupported seed_model_mode: {seed_model_mode}. "
            f"Allowed={sorted(ALLOWED_SEED_MODEL_MODES)}"
        )

    if mode == "independent":
        if args.model_path and len(seeds) > 1:
            raise ValueError(
                "--model-path with multiple seeds is incompatible with --seed-model-mode independent. "
                "Either provide per-seed trained models discoverable by seed, or use --seed-model-mode shared."
            )

        model_path_by_seed: dict[int, str] = {}
        for seed in seeds:
            try:
                ctx = build_eval_context(
                    config_path=str(config_path),
                    model_type=model_type,
                    seed=int(seed),
                    model_root=args.model_root,
                    offline=bool(args.offline),
                    max_steps=int(args.max_steps),
                    model_path_arg=None,
                    prefer_metadata_config=True,
                )
            except FileNotFoundError as exc:
                expected_path = Path(args.model_root) / "models" / f"{model_type}_seed{int(seed)}_final.zip"
                raise FileNotFoundError(
                    "Independent seed model resolution failed. "
                    f"missing_seed={int(seed)}, expected_path={expected_path}, "
                    "resolution=Train/export this seed model (or run with --seed-model-mode shared)."
                ) from exc
            model_path_by_seed[int(seed)] = str(ctx.model_path)
        return model_path_by_seed

    # Shared eval-only behavior: probe once, reuse across all seeds.
    model_probe_ctx = build_eval_context(
        config_path=str(config_path),
        model_type=model_type,
        seed=int(seeds[0]),
        model_root=args.model_root,
        offline=bool(args.offline),
        max_steps=int(args.max_steps),
        model_path_arg=args.model_path,
        prefer_metadata_config=True,
    )
    shared_model_path = str(model_probe_ctx.model_path)
    return {int(seed): shared_model_path for seed in seeds}


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    base_cfg = yaml.safe_load(config_path.read_text())
    _freeze_guard(base_cfg)

    model_type = str(
        _resolve_scalar_from_config(
            args.model_type,
            cfg_value=base_cfg.get("model_type"),
            flag="model-type",
        )
    )
    kappas = [float(k) for k in _resolve_list_from_config(args.kappas, cfg_values=base_cfg.get("kappas"), flag="kappas")]
    seeds = _ensure_paired_seeds(
        [int(s) for s in _resolve_list_from_config(args.seeds, cfg_values=base_cfg.get("seeds"), flag="seeds")]
    )
    etas = [float(e) for e in _resolve_list_from_config(args.etas, cfg_values=base_cfg.get("etas"), flag="etas")]
    if not etas:
        raise ValueError("etas must not be empty")
    if any((e <= 0.0 or e > 1.0) for e in etas):
        raise ValueError(f"etas must be in (0, 1], got: {etas}")

    output_cfg = base_cfg.get("output", {}) or {}
    out_root_arg = _resolve_scalar_from_config(args.out, cfg_value=output_cfg.get("root"), flag="out")
    out_root = Path(str(out_root_arg))
    save_trace = bool(output_cfg.get("save_trace", True))
    save_env_signature = bool(output_cfg.get("save_env_signature", True))
    save_cmd = bool(output_cfg.get("save_cmd", True))
    execution_cfg = base_cfg.get("execution", {}) or {}
    enforce_paired = bool(execution_cfg.get("enforce_paired_seeds", True))
    seed_model_mode = str(
        _resolve_scalar_from_config(
            args.seed_model_mode,
            cfg_value=execution_cfg.get("seed_model_mode"),
            flag="seed-model-mode",
        )
    ).strip().lower()
    if seed_model_mode not in ALLOWED_SEED_MODEL_MODES:
        raise ValueError(
            f"execution.seed_model_mode must be one of {sorted(ALLOWED_SEED_MODEL_MODES)}, "
            f"got: {seed_model_mode}"
        )
    experiment_name = str(base_cfg.get("experiment_name", config_path.stem))
    arm_specs = _build_arm_specs(base_cfg, etas=etas)
    if not arm_specs:
        raise ValueError("No experiment arms resolved from config.")

    # A) Run sanity first (stop entire run on failure).
    _run_sanity(args, seeds[0], config_path, model_type=model_type)

    model_path_by_seed = _resolve_model_path_by_seed(
        args=args,
        config_path=config_path,
        seeds=seeds,
        model_type=model_type,
        seed_model_mode=seed_model_mode,
    )

    out_root.mkdir(parents=True, exist_ok=True)

    seeds_by_arm: dict[tuple[float, str], set[int]] = {}
    for kappa in kappas:
        kappa_dir = out_root / f"kappa_{_format_kappa(kappa)}"
        kappa_dir.mkdir(parents=True, exist_ok=True)
        for arm_spec in arm_specs:
            arm_dir = kappa_dir / str(arm_spec["dir_name"])
            arm_dir.mkdir(parents=True, exist_ok=True)
            done: set[int] = set()

            for seed in seeds:
                run_cfg = copy.deepcopy(base_cfg)
                run_cfg.setdefault("env", {})
                run_cfg["env"]["transaction_cost"] = float(kappa)
                run_cfg["env"]["c_tc"] = float(kappa)
                run_cfg["env"] = _deep_merge_env(run_cfg["env"], arm_spec.get("env_patch", {}))

                seed_dir = arm_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                cfg_out_path = seed_dir / "config.yaml"
                cfg_out_path.write_text(yaml.safe_dump(run_cfg, sort_keys=True, allow_unicode=False))

                cmd_text = (
                    f"{sys.executable} scripts/step6_run_matrix.py --config {config_path} "
                    f"--kappas {' '.join(str(x) for x in kappas)} --etas {' '.join(str(x) for x in etas)} "
                    f"--seeds {' '.join(str(x) for x in seeds)} --out {out_root} "
                    f"--model-type {model_type} --model-root {args.model_root} "
                    f"--seed-model-mode {seed_model_mode} "
                    f"--max-steps {args.max_steps} --offline {args.offline} "
                    f"[per-run kappa={kappa}, arm={arm_spec['dir_name']}, seed={seed}]"
                )
                if save_cmd:
                    (seed_dir / "cmd.txt").write_text(cmd_text + "\n")

                ctx = build_eval_context(
                    config_path=str(cfg_out_path),
                    model_type=model_type,
                    seed=seed,
                    model_root=args.model_root,
                    offline=bool(args.offline),
                    max_steps=int(args.max_steps),
                    model_path_arg=model_path_by_seed[int(seed)],
                    prefer_metadata_config=False,
                )
                (seed_dir / "model_resolved_path.txt").write_text(str(ctx.model_path) + "\n")

                eta_mode_run = str(run_cfg["env"].get("eta_mode", "legacy"))
                rebalance_eta_run = run_cfg["env"].get("rebalance_eta")
                rebalance_eta_run = float(rebalance_eta_run) if rebalance_eta_run is not None else None
                metrics, trace, df, current_sig = run_eval_case(
                    ctx,
                    eta_mode=eta_mode_run,
                    rebalance_eta=rebalance_eta_run,
                    transaction_cost=float(kappa),
                    eval_tag=f"kappa_{_format_kappa(kappa)}__{arm_spec['dir_name']}__seed_{seed}",
                )

                trace_path = seed_dir / "trace.parquet"
                if save_trace:
                    df.to_parquet(trace_path, index=False)
                else:
                    trace_path = Path("")

                env_signature_payload = {
                    "expected_env_signature_hash": (ctx.run_meta or {}).get("env_signature_hash"),
                    "expected_env_signature_version": (ctx.run_meta or {}).get("env_signature_version"),
                    "current": current_sig,
                    "experiment_name": experiment_name,
                    "arm": arm_spec["arm_name"],
                    "step6_params": {
                        "eta_mode": str(run_cfg["env"].get("eta_mode", "legacy")),
                        "rule_vol_window": int((run_cfg["env"].get("rule_vol", {}) or {}).get("window", 20)),
                        "rule_vol_a": (
                            float((run_cfg["env"].get("rule_vol", {}) or {}).get("a"))
                            if (run_cfg["env"].get("rule_vol", {}) or {}).get("a") is not None
                            else None
                        ),
                        "eta_clip_min": (
                            float((run_cfg["env"].get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5])[0])
                            if isinstance((run_cfg["env"].get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5]), (list, tuple))
                            else 0.02
                        ),
                        "eta_clip_max": (
                            float((run_cfg["env"].get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5])[1])
                            if isinstance((run_cfg["env"].get("rule_vol", {}) or {}).get("eta_clip", [0.02, 0.5]), (list, tuple))
                            else 0.5
                        ),
                    },
                }
                step6_sig_seed = {
                    "env_signature_hash": current_sig.get("env_signature_hash"),
                    "step6_params": env_signature_payload["step6_params"],
                }
                env_signature_payload["step6_signature_hash"] = hashlib.sha256(
                    json.dumps(step6_sig_seed, sort_keys=True).encode("utf-8")
                ).hexdigest()
                if save_env_signature:
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
                rule_vol_info = run_cfg["env"].get("rule_vol", {}) or {}
                rule_vol_a = rule_vol_info.get("a")
                rule_vol_a = float(rule_vol_a) if rule_vol_a is not None else arm_spec.get("rule_vol_a", np.nan)

                metrics_row = pd.DataFrame(
                    [
                        {
                            "experiment_name": experiment_name,
                            "kappa": float(kappa),
                            "seed": int(seed),
                            "arm": str(arm_spec["arm_name"]),
                            "arm_dir": str(arm_spec["dir_name"]),
                            "run_id": ctx.run_id,
                            "model_type": ctx.model_type,
                            "eta_mode": eta_mode_run,
                            "eta_requested": (
                                float(arm_spec["eta_requested"])
                                if arm_spec.get("eta_requested") is not None
                                and np.isfinite(float(arm_spec["eta_requested"]))
                                else np.nan
                            ),
                            "eta": eta_value,
                            "rule_vol_a": rule_vol_a,
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
                            "trace_path": str(trace_path) if save_trace else "",
                        }
                    ]
                )
                metrics_row.to_csv(seed_dir / "metrics.csv", index=False)

                done.add(int(seed))

            seeds_by_arm[(float(kappa), str(arm_spec["dir_name"]))] = done

    if enforce_paired:
        target = set(seeds)
        for (kappa, arm_key), got in seeds_by_arm.items():
            if got != target:
                raise RuntimeError(
                    f"Paired seed enforcement failed for kappa={kappa}, arm={arm_key}: "
                    f"expected={sorted(target)}, got={sorted(got)}"
                )

    build_report_cmd = [sys.executable, str(ROOT / "scripts" / "step6_build_reports.py"), "--root", str(out_root)]
    subprocess.run(build_report_cmd, check=True)


if __name__ == "__main__":
    main()
