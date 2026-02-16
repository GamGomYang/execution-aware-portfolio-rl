from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


STEP5_EXPERIMENT_STEMS: dict[str, str] = {
    "A": "exp_S5_final_baseline_eta010",
    "B": "exp_S5_final_prl_eta010",
    "C": "exp_S5_ablate_baseline_etaNone",
    "D": "exp_S5_ablate_prl_etaNone",
}

STEP5_EXPERIMENT_LABELS: dict[str, str] = {
    "A": "Baseline (PRL-off, eta=0.10)",
    "B": "PRL Mainline (eta=0.10)",
    "C": "Baseline (PRL-off, eta=None)",
    "D": "PRL Mainline (eta=None)",
}

STEP5_MAIN_METRICS: list[str] = [
    "sharpe_net_exp",
    "cumulative_return_net_exp",
    "max_drawdown_net_exp",
    "avg_turnover_exec",
    "std_daily_net_return_exp",
    "avg_turnover_target",
]


def strip_retry_suffix(exp_id: str) -> str:
    return re.sub(r"__\d+$", "", exp_id)


def parse_exp_id_from_archive(path: Path, *, prefix: str) -> str | None:
    stem = path.stem
    token = f"{prefix}_"
    if not stem.startswith(token):
        return None
    return stem[len(token) :]


def classify_exp_id(exp_id: str, *, stems: dict[str, str] | None = None) -> str | None:
    source = strip_retry_suffix(exp_id)
    mapping = stems or STEP5_EXPERIMENT_STEMS
    for key, expected in mapping.items():
        if expected in source:
            return key
    return None


def safe_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def load_latest_archive_frames(
    input_root: Path,
    *,
    prefix: str,
    stems: dict[str, str] | None = None,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    archive_dir = input_root / "reports" / "archive"
    if not archive_dir.exists():
        return {}
    selected: dict[str, tuple[Path, str]] = {}
    for path in archive_dir.glob(f"{prefix}_*.csv"):
        exp_id = parse_exp_id_from_archive(path, prefix=prefix)
        if not exp_id:
            continue
        key = classify_exp_id(exp_id, stems=stems)
        if key is None:
            continue
        prev = selected.get(key)
        if prev is None or path.stat().st_mtime > prev[0].stat().st_mtime:
            selected[key] = (path, exp_id)

    loaded: dict[str, tuple[Path, pd.DataFrame]] = {}
    for key, (path, exp_id) in selected.items():
        df = pd.read_csv(path)
        df["step5_exp_key"] = key
        df["step5_exp_id"] = strip_retry_suffix(exp_id)
        df["step5_source_path"] = str(path)
        loaded[key] = (path, df)
    return loaded


def describe(values) -> dict[str, float]:
    arr = safe_values(values)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "iqr": float("nan"),
        }
    p25, p50, p75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "iqr": float(p75 - p25),
    }


def pair_seed_values(base_df: pd.DataFrame, comp_df: pd.DataFrame, metric: str) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    if metric not in base_df.columns or metric not in comp_df.columns:
        raise KeyError(metric)
    base = base_df.drop_duplicates(subset=["seed"]).set_index("seed")
    comp = comp_df.drop_duplicates(subset=["seed"]).set_index("seed")
    base_seeds = set(int(v) for v in base.index.tolist())
    comp_seeds = set(int(v) for v in comp.index.tolist())
    missing_in_comp = sorted(base_seeds - comp_seeds)
    missing_in_base = sorted(comp_seeds - base_seeds)
    if missing_in_comp or missing_in_base:
        raise ValueError(
            f"SEED_PAIRING_MISMATCH metric={metric} missing_in_comp={missing_in_comp} missing_in_base={missing_in_base}"
        )
    seeds = sorted(base_seeds)
    base_vals = pd.to_numeric(base.loc[seeds, metric], errors="coerce").to_numpy(dtype=np.float64)
    comp_vals = pd.to_numeric(comp.loc[seeds, metric], errors="coerce").to_numpy(dtype=np.float64)
    delta = comp_vals - base_vals
    return seeds, base_vals, comp_vals, delta


def load_run_metadata_by_exp(input_root: Path, *, stems: dict[str, str] | None = None) -> dict[str, list[dict]]:
    reports_dir = input_root / "reports"
    grouped: dict[str, list[dict]] = {}
    if not reports_dir.exists():
        return grouped
    mapping = stems or STEP5_EXPERIMENT_STEMS
    for path in reports_dir.glob("run_metadata_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        cfg_stem = Path(str(data.get("config_path") or "")).stem
        exp_key = classify_exp_id(cfg_stem, stems=mapping)
        if exp_key is None:
            continue
        payload = {**data, "_meta_path": str(path), "_config_stem": cfg_stem}
        grouped.setdefault(exp_key, []).append(payload)
    return grouped
