from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PRL_GATE_THRESHOLDS = {
    "emergency_rate_max": 0.05,
    "prl_prob_p95_min": 0.8,
    "prl_prob_p05_max": 0.2,
    "prl_prob_std_min": 0.25,
}


@dataclass
class PRLGateResult:
    passed: bool
    reason: str
    emergency_rate: float | None
    prl_prob_p05: float | None
    prl_prob_p95: float | None
    prl_prob_std: float | None
    prl_prob_min: float | None
    prl_prob_max: float | None
    source: str | None


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return None


def _resolve_logs_dir(run_index: dict, metrics_path: Path) -> Path | None:
    logs_dir = run_index.get("logs_dir")
    if logs_dir:
        return Path(logs_dir)
    reports_dir = run_index.get("reports_dir")
    if reports_dir:
        return Path(reports_dir).parent / "logs"
    try:
        return metrics_path.parent.parent / "logs"
    except Exception:
        return None


def _resolve_train_log_path(run_id: str, run_index: dict, metrics_path: Path) -> Path | None:
    logs_dir = _resolve_logs_dir(run_index, metrics_path)
    if logs_dir is not None:
        candidate = logs_dir / f"train_{run_id}.csv"
        if candidate.exists():
            return candidate
    reports_dir = run_index.get("reports_dir")
    if reports_dir:
        meta_path = Path(reports_dir) / f"run_metadata_{run_id}.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
            except Exception:
                data = {}
            artifacts = data.get("artifact_paths") or data.get("artifacts") or {}
            train_log = artifacts.get("train_log_path")
            if train_log:
                candidate = Path(train_log)
                if candidate.exists():
                    return candidate
    return None


def _gate_from_row(row: pd.Series, *, source: str) -> PRLGateResult:
    emergency_rate = _safe_float(row.get("emergency_rate"))
    prl_prob_p05 = _safe_float(row.get("prl_prob_p05"))
    prl_prob_p95 = _safe_float(row.get("prl_prob_p95"))
    prl_prob_std = _safe_float(row.get("prl_prob_std"))
    prl_prob_min = _safe_float(row.get("prl_prob_min"))
    prl_prob_max = _safe_float(row.get("prl_prob_max"))

    missing = [
        name
        for name, val in {
            "emergency_rate": emergency_rate,
            "prl_prob_p05": prl_prob_p05,
            "prl_prob_p95": prl_prob_p95,
            "prl_prob_std": prl_prob_std,
        }.items()
        if val is None
    ]
    if missing:
        return PRLGateResult(
            passed=False,
            reason=f"missing_prl_metrics:{','.join(missing)}",
            emergency_rate=emergency_rate,
            prl_prob_p05=prl_prob_p05,
            prl_prob_p95=prl_prob_p95,
            prl_prob_std=prl_prob_std,
            prl_prob_min=prl_prob_min,
            prl_prob_max=prl_prob_max,
            source=source,
        )

    fails = []
    if emergency_rate > PRL_GATE_THRESHOLDS["emergency_rate_max"]:
        fails.append("emergency_rate")
    if prl_prob_p95 < PRL_GATE_THRESHOLDS["prl_prob_p95_min"]:
        fails.append("prl_prob_p95")
    if prl_prob_p05 > PRL_GATE_THRESHOLDS["prl_prob_p05_max"]:
        fails.append("prl_prob_p05")
    if prl_prob_std < PRL_GATE_THRESHOLDS["prl_prob_std_min"]:
        fails.append("prl_prob_std")

    if fails:
        reason = "prl_gate_fail:" + ",".join(fails)
        passed = False
    else:
        reason = "prl_gate_pass"
        passed = True

    return PRLGateResult(
        passed=passed,
        reason=reason,
        emergency_rate=emergency_rate,
        prl_prob_p05=prl_prob_p05,
        prl_prob_p95=prl_prob_p95,
        prl_prob_std=prl_prob_std,
        prl_prob_min=prl_prob_min,
        prl_prob_max=prl_prob_max,
        source=source,
    )


def load_prl_gate_for_run_id(run_id: str, run_index: dict, metrics_path: Path) -> PRLGateResult:
    log_path = _resolve_train_log_path(run_id, run_index, metrics_path)
    if log_path is None:
        return PRLGateResult(
            passed=False,
            reason="missing_prl_log",
            emergency_rate=None,
            prl_prob_p05=None,
            prl_prob_p95=None,
            prl_prob_std=None,
            prl_prob_min=None,
            prl_prob_max=None,
            source=None,
        )
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return PRLGateResult(
            passed=False,
            reason="unreadable_prl_log",
            emergency_rate=None,
            prl_prob_p05=None,
            prl_prob_p95=None,
            prl_prob_std=None,
            prl_prob_min=None,
            prl_prob_max=None,
            source=str(log_path),
        )
    if df.empty:
        return PRLGateResult(
            passed=False,
            reason="empty_prl_log",
            emergency_rate=None,
            prl_prob_p05=None,
            prl_prob_p95=None,
            prl_prob_std=None,
            prl_prob_min=None,
            prl_prob_max=None,
            source=str(log_path),
        )
    row = df.iloc[-1]
    return _gate_from_row(row, source=str(log_path))


def aggregate_prl_gate(run_ids: Iterable[str], run_index: dict, metrics_path: Path) -> PRLGateResult:
    results = [load_prl_gate_for_run_id(run_id, run_index, metrics_path) for run_id in run_ids]
    if not results:
        return PRLGateResult(
            passed=False,
            reason="missing_prl_log",
            emergency_rate=None,
            prl_prob_p05=None,
            prl_prob_p95=None,
            prl_prob_std=None,
            prl_prob_min=None,
            prl_prob_max=None,
            source=None,
        )

    if any(not r.passed for r in results):
        # propagate most informative failure reason
        reason = ";".join(sorted({r.reason for r in results if not r.passed}))
        return PRLGateResult(
            passed=False,
            reason=reason,
            emergency_rate=max([r.emergency_rate for r in results if r.emergency_rate is not None], default=None),
            prl_prob_p05=max([r.prl_prob_p05 for r in results if r.prl_prob_p05 is not None], default=None),
            prl_prob_p95=min([r.prl_prob_p95 for r in results if r.prl_prob_p95 is not None], default=None),
            prl_prob_std=min([r.prl_prob_std for r in results if r.prl_prob_std is not None], default=None),
            prl_prob_min=min([r.prl_prob_min for r in results if r.prl_prob_min is not None], default=None),
            prl_prob_max=max([r.prl_prob_max for r in results if r.prl_prob_max is not None], default=None),
            source=None,
        )

    return PRLGateResult(
        passed=True,
        reason="prl_gate_pass",
        emergency_rate=max([r.emergency_rate for r in results if r.emergency_rate is not None], default=None),
        prl_prob_p05=max([r.prl_prob_p05 for r in results if r.prl_prob_p05 is not None], default=None),
        prl_prob_p95=min([r.prl_prob_p95 for r in results if r.prl_prob_p95 is not None], default=None),
        prl_prob_std=min([r.prl_prob_std for r in results if r.prl_prob_std is not None], default=None),
        prl_prob_min=min([r.prl_prob_min for r in results if r.prl_prob_min is not None], default=None),
        prl_prob_max=max([r.prl_prob_max for r in results if r.prl_prob_max is not None], default=None),
        source=None,
    )
