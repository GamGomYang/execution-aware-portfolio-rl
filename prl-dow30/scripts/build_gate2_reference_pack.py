"""
Pack split reference baseline runs (per-seed output_roots) into a single combined reference.

Usage:
  python -m scripts.build_gate2_reference_pack \\
    --ref-run-indexes "outputs/exp_runs/gate2/reference_baseline_sac_seed*/**/reports/run_index.json" \\
    --output-dir outputs/exp_runs/gate2/reference_baseline_sac_PACK

Outputs (under <output-dir>/reports):
  - metrics.csv (concat, filtered by run_ids)
  - regime_metrics.csv (concat, filtered by run_ids)
  - run_index.json (combined run_ids/seeds/paths)
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _load_run_index(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["run_index_path"] = str(path)
    return data


def _load_with_filter(csv_path: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    run_ids = set(run_ids)
    if run_ids:
        df = df[df["run_id"].isin(run_ids)].copy()
    return df


def _collect_paths(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            path = Path(p)
            if path.exists():
                paths.append(path)
    uniq = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="Pack split reference baseline run indexes into a combined reference.")
    parser.add_argument("--ref-run-indexes", nargs="+", required=True, help="Paths/globs to reference run_index.json files.")
    parser.add_argument("--output-dir", default="outputs/exp_runs/gate2/reference_baseline_sac_PACK", help="Output root for packed reference.")
    args = parser.parse_args()

    ref_index_paths = _collect_paths(args.ref_run_indexes)
    if not ref_index_paths:
        raise SystemExit("No reference run_index paths found.")

    run_ids: list[str] = []
    seeds: list[int] = []
    metrics_rows: list[pd.DataFrame] = []
    regime_rows: list[pd.DataFrame] = []
    eval_windows = None
    model_types = None
    config_path = None

    for path in ref_index_paths:
        idx = _load_run_index(path)
        eval_windows = eval_windows or idx.get("eval_windows")
        model_types = model_types or idx.get("model_types")
        config_path = config_path or idx.get("config_path")
        run_ids_this = idx.get("run_ids", [])
        run_ids.extend(run_ids_this)
        seeds.extend(idx.get("seeds", []))
        metrics_rows.append(_load_with_filter(Path(idx["metrics_path"]), run_ids_this))
        regime_rows.append(_load_with_filter(Path(idx["regime_metrics_path"]), run_ids_this))

    metrics_df = pd.concat(metrics_rows, ignore_index=True)
    regime_df = pd.concat(regime_rows, ignore_index=True)

    output_root = Path(args.output_dir)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.csv"
    regime_path = reports_dir / "regime_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    now = datetime.now(timezone.utc).isoformat()
    run_index = {
        "exp_name": "reference_baseline_sac_PACK",
        "timestamp": now,
        "config_path": config_path or "packed",
        "model_types": model_types or ["baseline"],
        "seeds": sorted(set(seeds)),
        "eval_windows": eval_windows or {},
        "run_ids": run_ids,
        "metrics_path": str(metrics_path),
        "regime_metrics_path": str(regime_path),
        "reports_dir": str(reports_dir),
        "traces_dir": str(output_root / "traces"),
        "models_dir": str(output_root / "models"),
        "logs_dir": str(output_root / "logs"),
        "output_root": str(output_root),
        "source_run_indexes": [str(p) for p in ref_index_paths],
    }
    (reports_dir / "run_index.json").write_text(json.dumps(run_index, indent=2))
    print(f"Packed reference written to {reports_dir}")


if __name__ == "__main__":
    main()
