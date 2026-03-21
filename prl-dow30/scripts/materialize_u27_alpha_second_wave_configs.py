#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES: list[dict[str, Any]] = [
    {
        "key": "ctrl",
        "label": "control_mean_reversion",
        "manifest": "alpha2_ctrl.json",
        "signals": ["reversal_5d", "short_term_reversal"],
        "notes": "Current incumbent control basket used as the second-wave baseline.",
    },
    {
        "key": "cs312",
        "label": "control_plus_cs_mom_3_12",
        "manifest": "alpha2_cs312.json",
        "signals": ["reversal_5d", "short_term_reversal", "cs_mom_3_12"],
        "notes": "Add the one remaining momentum family not directly screened in first-wave.",
    },
    {
        "key": "cs61vsmom",
        "label": "control_plus_cs_mom_6_1_plus_vol_scaled_mom",
        "manifest": "alpha2_cs61_vsmom.json",
        "signals": ["reversal_5d", "short_term_reversal", "cs_mom_6_1", "vol_scaled_mom"],
        "notes": "Pair medium-horizon momentum with volatility-scaled momentum to test two-addon complementarity.",
    },
    {
        "key": "cs61resid",
        "label": "control_plus_cs_mom_6_1_plus_residual_mom",
        "manifest": "alpha2_cs61_resid.json",
        "signals": ["reversal_5d", "short_term_reversal", "cs_mom_6_1", "residual_mom_beta_neutral"],
        "notes": "Pair medium-horizon momentum with residual momentum to test orthogonal addon stacking.",
    },
    {
        "key": "cs312resid",
        "label": "control_plus_cs_mom_3_12_plus_residual_mom",
        "manifest": "alpha2_cs312_resid.json",
        "signals": ["reversal_5d", "short_term_reversal", "cs_mom_3_12", "residual_mom_beta_neutral"],
        "notes": "Pair the untouched 3_12 momentum family with residual momentum for a second-wave recovery shot.",
    },
]

CONTROL_SIGNALS = ["reversal_5d", "short_term_reversal"]
CANDIDATE_BY_KEY: dict[str, dict[str, Any]] = {item["key"]: item for item in CANDIDATES}


def candidate_tag(key: str, tag_suffix: str) -> str:
    return f"u27_eta082_alpha2_{key}_{tag_suffix}"


def candidate_key_from_tag(tag_or_key: str) -> str:
    raw = str(tag_or_key).strip()
    if not raw:
        raise ValueError("Empty candidate identifier is not allowed.")
    if raw in CANDIDATE_BY_KEY:
        return raw
    prefix = "u27_eta082_alpha2_"
    if raw.startswith(prefix):
        remainder = raw[len(prefix) :]
        key = remainder.split("_", 1)[0]
        if key in CANDIDATE_BY_KEY:
            return key
    raise ValueError(
        f"Unknown candidate identifier: {raw}. "
        f"Expected one of keys={list(CANDIDATE_BY_KEY)} or tags starting with {prefix}."
    )


def normalize_candidate_keys(values: list[str] | None) -> list[str]:
    if not values:
        return [item["key"] for item in CANDIDATES]
    out: list[str] = []
    for raw in values:
        token = str(raw).strip()
        if not token:
            continue
        key = candidate_key_from_tag(token)
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one valid candidate must be selected.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize alpha second-wave signal manifests and screen configs.")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/prl_100k_signals_u27_eta082_current.yaml",
        help="Current incumbent config to clone fixed execution settings from.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="configs/signal_sets/alpha_second_wave",
        help="Directory to write signal selection manifests into.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/exp",
        help="Directory to write alpha second-wave configs into.",
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        default="outputs/reports/u27_alpha_second_wave_materialization.json",
        help="Metadata JSON output path.",
    )
    parser.add_argument(
        "--rationale-csv-out",
        type=str,
        default="outputs/reports/u27_alpha_second_wave_signal_rationale_20260321.csv",
        help="CSV rationale output path.",
    )
    parser.add_argument(
        "--rationale-md-out",
        type=str,
        default="outputs/reports/u27_alpha_second_wave_signal_rationale_20260321.md",
        help="Markdown rationale output path.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Optional candidate subset. Accepts candidate keys or full alpha2 tags.",
    )
    parser.add_argument(
        "--tag-suffix",
        type=str,
        default="20k_r1",
        help="Suffix appended to generated config tags, e.g. 20k_r1 or 100k_r1.",
    )
    parser.add_argument(
        "--skip-rationale",
        action="store_true",
        help="Skip rationale recomputation and CSV/Markdown writes.",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Do not overwrite signal manifest files. Existing manifests must already exist.",
    )
    parser.add_argument("--train-start", type=str, default="2010-01-01")
    parser.add_argument("--train-end", type=str, default="2021-12-31")
    parser.add_argument("--test-start", type=str, default="2022-01-01")
    parser.add_argument("--test-end", type=str, default="2023-12-31")
    parser.add_argument("--timesteps", type=int, default=20000)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _read_yaml(path_str: str) -> dict[str, Any]:
    return yaml.safe_load(_resolve(path_str).read_text())


def _write_yaml(path_str: str, payload: dict[str, Any]) -> None:
    out = _resolve(path_str)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def _write_json(path_str: str, payload: dict[str, Any]) -> None:
    out = _resolve(path_str)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


def _compute_rationale(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import sys

    sys.path.insert(0, str(ROOT))
    from prl.signals import AVAILABLE_SIGNALS, compute_signal_frames

    prices = pd.read_parquet((_resolve(processed_dir) / "prices.parquet")).loc["2010-01-01":"2021-12-31"]
    returns = pd.read_parquet((_resolve(processed_dir) / "returns.parquet")).loc["2010-01-01":"2021-12-31"]
    signal_frames = compute_signal_frames(prices, returns, signals=list(AVAILABLE_SIGNALS))

    flat: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for name, frame in signal_frames.items():
        stacked = frame.stack().dropna()
        flat[name] = stacked
        rows.append(
            {
                "signal": name,
                "n_obs": int(stacked.shape[0]),
                "coverage_dates": int(frame.notna().any(axis=1).sum()),
                "mean_abs_z": float(stacked.abs().mean()),
                "std_z": float(stacked.std(ddof=0)),
            }
        )

    control_corr_rows: list[dict[str, Any]] = []
    for name in [sig for sig in AVAILABLE_SIGNALS if sig not in CONTROL_SIGNALS]:
        joined_rev5 = pd.concat([flat["reversal_5d"].rename("a"), flat[name].rename("b")], axis=1, join="inner").dropna()
        joined_str = pd.concat([flat["short_term_reversal"].rename("a"), flat[name].rename("b")], axis=1, join="inner").dropna()
        control_corr_rows.append(
            {
                "signal": name,
                "corr_reversal_5d": float(joined_rev5["a"].corr(joined_rev5["b"])),
                "corr_short_term_reversal": float(joined_str["a"].corr(joined_str["b"])),
                "n_overlap": int(min(len(joined_rev5), len(joined_str))),
            }
        )

    stats_df = pd.DataFrame(rows).sort_values("signal").reset_index(drop=True)
    corr_df = pd.DataFrame(control_corr_rows).sort_values(
        ["corr_short_term_reversal", "corr_reversal_5d"]
    ).reset_index(drop=True)
    return stats_df, corr_df


def main() -> None:
    args = parse_args()
    base_cfg = _read_yaml(args.base_config)
    selected_keys = normalize_candidate_keys(args.candidates)

    processed_dir = str((base_cfg.get("data", {}) or {}).get("processed_dir", "data/processed_u27"))
    stats_df: pd.DataFrame | None = None
    corr_df: pd.DataFrame | None = None
    if not args.skip_rationale:
        stats_df, corr_df = _compute_rationale(processed_dir)

    manifest_dir = _resolve(args.manifest_dir)
    config_dir = _resolve(args.config_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for candidate_key in selected_keys:
        candidate = CANDIDATE_BY_KEY[candidate_key]
        tag = candidate_tag(candidate_key, args.tag_suffix)
        manifest_rel = f"configs/signal_sets/alpha_second_wave/{candidate['manifest']}"
        manifest_payload = {
            "tag": tag,
            "candidate_key": candidate_key,
            "tag_prefix": f"u27_eta082_alpha2_{candidate_key}",
            "label": candidate["label"],
            "selected_signals": candidate["signals"],
            "notes": candidate["notes"],
        }
        manifest_path = _resolve(manifest_rel)
        if args.skip_manifests:
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest write skipped but file does not exist: {manifest_path}")
        else:
            _write_json(manifest_rel, manifest_payload)

        cfg = json.loads(json.dumps(base_cfg))
        cfg["mode"] = "screen_u27_eta082_alpha_second_wave"
        cfg.setdefault("dates", {})
        cfg["dates"]["train_start"] = args.train_start
        cfg["dates"]["train_end"] = args.train_end
        cfg["dates"]["test_start"] = args.test_start
        cfg["dates"]["test_end"] = args.test_end

        data_cfg = cfg.setdefault("data", {})
        data_cfg["force_refresh"] = False
        data_cfg["offline"] = True
        data_cfg["require_cache"] = True
        data_cfg["paper_mode"] = True

        signals_cfg = cfg.setdefault("signals", {})
        signals_cfg["enabled"] = True
        signals_cfg["signal_state"] = True
        signals_cfg["selection_policy"] = "alpha_research"
        signals_cfg["allow_nonfixed_selection"] = True
        signals_cfg["signal_names"] = list(candidate["signals"])
        signals_cfg["selected_signals_path"] = f"../signal_sets/alpha_second_wave/{candidate['manifest']}"

        cfg.setdefault("output", {})
        cfg["output"]["root"] = f"outputs/{tag}"

        sac_cfg = cfg.setdefault("sac", {})
        sac_cfg["total_timesteps"] = int(args.timesteps)

        cfg["eval"] = {
            "write_trace": False,
            "trace_stride": 10,
            "run_baselines": False,
            "write_step4": False,
        }

        config_rel = f"configs/exp/{tag}.yaml"
        _write_yaml(config_rel, cfg)
        generated.append(
            {
                "candidate_key": candidate_key,
                "tag": tag,
                "label": candidate["label"],
                "signals": candidate["signals"],
                "manifest": manifest_rel,
                "config": config_rel,
                "output_root": cfg["output"]["root"],
                "notes": candidate["notes"],
            }
        )

    if stats_df is not None and corr_df is not None:
        stats_merge = stats_df.merge(corr_df, on="signal", how="left")
        stats_merge.to_csv(_resolve(args.rationale_csv_out), index=False)

        md_lines: list[str] = []
        md_lines.append("# U27 Alpha Second Wave Signal Rationale")
        md_lines.append("")
        md_lines.append(f"- base_config: {args.base_config}")
        md_lines.append(f"- selection_window: {args.test_start}~{args.test_end}")
        md_lines.append(f"- processed_dir: {processed_dir}")
        md_lines.append(f"- timesteps: {args.timesteps}")
        md_lines.append(f"- tag_suffix: {args.tag_suffix}")
        md_lines.append("")
        md_lines.append("## Candidate Set")
        md_lines.append("")
        for item in generated:
            md_lines.append(f"- `{item['tag']}`: {item['signals']} :: {item['notes']}")
        md_lines.append("")
        md_lines.append("## Signal Stats")
        md_lines.append("")
        md_lines.append("| signal | coverage_dates | mean_abs_z | corr_reversal_5d | corr_short_term_reversal |")
        md_lines.append("| --- | --- | --- | --- | --- |")
        for row in stats_merge.to_dict(orient="records"):
            md_lines.append(
                f"| {row['signal']} | {row['coverage_dates']} | {row['mean_abs_z']:.4f} | "
                f"{row['corr_reversal_5d'] if pd.notna(row['corr_reversal_5d']) else ''} | "
                f"{row['corr_short_term_reversal'] if pd.notna(row['corr_short_term_reversal']) else ''} |"
            )
        md_lines.append("")
        md_lines.append("## Selection Rationale")
        md_lines.append("")
        md_lines.append("- `cs_mom_3_12` is the only momentum family not directly tested in first-wave, so it is the first recovery candidate.")
        md_lines.append("- Two-addon combos are explicitly allowed in second-wave because first-wave single-addon tests were directionally useful but not promotion-ready.")
        md_lines.append("- `vol_scaled_mom` and `residual_mom_beta_neutral` remain in scope only as complements to momentum, not as solo promotion bets.")
        _resolve(args.rationale_md_out).write_text("\n".join(md_lines) + "\n")

    meta = {
        "base_config": args.base_config,
        "manifest_dir": args.manifest_dir,
        "config_dir": args.config_dir,
        "selected_candidate_keys": selected_keys,
        "tag_suffix": args.tag_suffix,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "timesteps": args.timesteps,
        "processed_dir": processed_dir,
        "generated": generated,
        "skip_rationale": bool(args.skip_rationale),
        "skip_manifests": bool(args.skip_manifests),
        "rationale_csv_out": None if args.skip_rationale else args.rationale_csv_out,
        "rationale_md_out": None if args.skip_rationale else args.rationale_md_out,
    }
    _write_json(args.meta_out, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
