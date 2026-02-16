"""
Pack multiple per-seed run outputs into a single PACK (metrics/regime/run_index).

Usage example (Gate3):
  python3 scripts/build_run_pack.py \
    --input-run-indexes "outputs/exp_runs/gate3/reference_baseline_sac_seed*/**/reports/run_index.json" \
    --output-root outputs/exp_runs/gate3/reference_baseline_sac_PACK/20260126_120000
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _collect_paths(patterns: List[str]) -> List[Path]:
    paths: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            path = Path(p)
            if path.exists():
                paths.append(path)
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _load_index(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["run_index_path"] = str(path)
    return data


def _load_filtered(csv_path: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    run_ids = list(run_ids)
    if run_ids:
        df = df[df["run_id"].isin(run_ids)].copy()
    return df


def _dedup_preserve(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pack multiple run_index roots into a single PACK output_root.")
    parser.add_argument("--input-run-indexes", nargs="+", required=True, help="Paths/globs to run_index.json files.")
    parser.add_argument("--output-root", required=True, help="PACK output root (reports/metrics/regime/run_index written here).")
    parser.add_argument("--exp-name", default=None, help="Override exp_name in packed run_index (default: basename of output_root).")
    args = parser.parse_args()

    index_paths = _collect_paths(args.input_run_indexes)
    if not index_paths:
        raise SystemExit("No run_index paths found for given patterns.")

    run_ids_all: list[str] = []
    seeds_all: list[int] = []
    metrics_parts: list[pd.DataFrame] = []
    regime_parts: list[pd.DataFrame] = []
    eval_windows = None
    model_types = None
    config_path = None

    for p in index_paths:
        idx = _load_index(p)
        run_ids = idx.get("run_ids", [])
        run_ids_all.extend(run_ids)
        seeds_all.extend(idx.get("seeds", []))
        eval_windows = eval_windows or idx.get("eval_windows")
        model_types = model_types or idx.get("model_types")
        config_path = config_path or idx.get("config_path")
        metrics_parts.append(_load_filtered(Path(idx["metrics_path"]), run_ids))
        regime_parts.append(_load_filtered(Path(idx["regime_metrics_path"]), run_ids))

    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    regime_df = pd.concat(regime_parts, ignore_index=True)

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.csv"
    regime_path = reports_dir / "regime_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    now = datetime.now(timezone.utc).isoformat()
    run_index = {
        "exp_name": args.exp_name or Path(args.output_root).name,
        "timestamp": now,
        "config_path": config_path or "packed",
        "model_types": model_types or [],
        "seeds": sorted(set(seeds_all)),
        "eval_windows": eval_windows or {},
        "run_ids": _dedup_preserve(run_ids_all),
        "metrics_path": str(metrics_path),
        "regime_metrics_path": str(regime_path),
        "reports_dir": str(reports_dir),
        "traces_dir": str(output_root / "traces"),
        "models_dir": str(output_root / "models"),
        "logs_dir": str(output_root / "logs"),
        "output_root": str(output_root),
        "source_run_indexes": [str(p) for p in index_paths],
    }
    (reports_dir / "run_index.json").write_text(json.dumps(run_index, indent=2))
    print(f"Packed {len(index_paths)} run_index files -> {reports_dir}")


if __name__ == "__main__":
    main()
