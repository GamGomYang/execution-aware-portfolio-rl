#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass
class ArchiveExperiment:
    exp_tag: str
    summary_path: Path
    metrics_path: Path
    n_seeds: int
    sharpe_net_lin_mean: float | None
    sharpe_net_lin_std: float | None
    cumulative_return_net_lin_mean: float | None
    cumulative_return_net_lin_std: float | None
    avg_turnover_exec_mean: float | None
    avg_turnover_exec_std: float | None
    max_drawdown_net_lin_mean: float | None
    max_drawdown_net_lin_std: float | None
    duration_min_mean: float | None
    duration_min_min: float | None
    duration_min_max: float | None
    per_seed_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append Step6 progress/result analysis snapshot.")
    parser.add_argument(
        "--output-file",
        type=str,
        default="/workspace/execution-aware-portfolio-rl/결과 분석",
        help="Output analysis log file path.",
    )
    parser.add_argument(
        "--mode",
        choices=["append", "overwrite"],
        default="append",
        help="Write mode.",
    )
    parser.add_argument(
        "--job-ts",
        type=str,
        default="",
        help="Current detached job timestamp (optional).",
    )
    parser.add_argument(
        "--run-log",
        type=str,
        default="",
        help="Path to current detached master.log (optional).",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _parse_run_start_utc(run_id: str) -> datetime | None:
    m = re.match(r"^(\d{8}T\d{6}Z)_", run_id)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect_archive_experiments(prl_root: Path) -> list[ArchiveExperiment]:
    archive_dir = prl_root / "tmp_fast_real_out" / "20260216_060623" / "reports" / "archive"
    run_meta_dir = prl_root / "tmp_fast_real_out" / "20260216_060623" / "reports"
    if not archive_dir.exists():
        return []

    entries: list[ArchiveExperiment] = []
    for summary_path in sorted(archive_dir.glob("summary_*.csv")):
        exp_tag = summary_path.stem.replace("summary_", "")
        metrics_path = archive_dir / summary_path.name.replace("summary_", "metrics_")
        if not metrics_path.exists():
            continue

        summary_rows = _read_csv_rows(summary_path)
        if not summary_rows:
            continue
        summary = summary_rows[0]

        metrics_rows = _read_csv_rows(metrics_path)
        duration_list: list[float] = []
        per_seed_rows: list[dict[str, Any]] = []
        for row in sorted(metrics_rows, key=lambda x: int(x.get("seed", "0"))):
            run_id = str(row.get("run_id", ""))
            meta_path = run_meta_dir / f"run_metadata_{run_id}.json"
            duration_min: float | None = None
            if meta_path.exists():
                try:
                    payload = json.loads(meta_path.read_text())
                    start_utc = _parse_run_start_utc(run_id)
                    end_utc = datetime.fromisoformat(str(payload.get("created_at")))
                    if start_utc is not None:
                        duration_min = (end_utc - start_utc).total_seconds() / 60.0
                except Exception:
                    duration_min = None
            if duration_min is not None:
                duration_list.append(duration_min)

            per_seed_rows.append(
                {
                    "seed": int(row.get("seed", "0")),
                    "run_id": run_id,
                    "sharpe_net_lin": _safe_float(row.get("sharpe_net_lin")),
                    "cumulative_return_net_lin": _safe_float(row.get("cumulative_return_net_lin")),
                    "avg_turnover_exec": _safe_float(row.get("avg_turnover_exec")),
                    "max_drawdown_net_lin": _safe_float(row.get("max_drawdown_net_lin")),
                    "duration_min": duration_min,
                }
            )

        entries.append(
            ArchiveExperiment(
                exp_tag=exp_tag,
                summary_path=summary_path,
                metrics_path=metrics_path,
                n_seeds=len(metrics_rows),
                sharpe_net_lin_mean=_safe_float(summary.get("sharpe_net_lin_mean")),
                sharpe_net_lin_std=_safe_float(summary.get("sharpe_net_lin_std")),
                cumulative_return_net_lin_mean=_safe_float(summary.get("cumulative_return_net_lin_mean")),
                cumulative_return_net_lin_std=_safe_float(summary.get("cumulative_return_net_lin_std")),
                avg_turnover_exec_mean=_safe_float(summary.get("avg_turnover_exec_mean")),
                avg_turnover_exec_std=_safe_float(summary.get("avg_turnover_exec_std")),
                max_drawdown_net_lin_mean=_safe_float(summary.get("max_drawdown_net_lin_mean")),
                max_drawdown_net_lin_std=_safe_float(summary.get("max_drawdown_net_lin_std")),
                duration_min_mean=mean(duration_list) if duration_list else None,
                duration_min_min=min(duration_list) if duration_list else None,
                duration_min_max=max(duration_list) if duration_list else None,
                per_seed_rows=per_seed_rows,
            )
        )
    return entries


def collect_spec_stage6_record(spec_path: Path) -> dict[str, Any]:
    if not spec_path.exists():
        return {}
    text = spec_path.read_text()
    out: dict[str, Any] = {}
    patterns = {
        "run_path": r"- 런 경로\(당시\):\s*(.+)",
        "total_minutes": r"- 총 소요:\s*([0-9.]+)분",
        "final_status": r"- 종료 상태:\s*(.+)",
        "a1": r"- A1:\s*(.+)",
        "a2": r"- A2:\s*(.+)",
        "kappa_0005": r"- kappa=0\.0005:\s*positive seeds\s*([0-9]+)/([0-9]+),\s*median delta sharpe\s*([+-]?[0-9.]+)",
        "kappa_0010": r"- kappa=0\.0010:\s*positive seeds\s*([0-9]+)/([0-9]+),\s*median delta sharpe\s*([+-]?[0-9.]+)",
        "collapse_rate": r"- collapse rate:\s*([0-9.]+)\s*\(([0-9]+)/([0-9]+)\)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue
        if key in {"kappa_0005", "kappa_0010"}:
            out[key] = {
                "positive": int(m.group(1)),
                "total": int(m.group(2)),
                "median_delta_sharpe": float(m.group(3)),
            }
        elif key == "collapse_rate":
            out[key] = {
                "rate": float(m.group(1)),
                "count": int(m.group(2)),
                "total": int(m.group(3)),
            }
        elif key == "total_minutes":
            out[key] = float(m.group(1))
        else:
            out[key] = m.group(1).strip()

    check_block = re.search(
        r"StageB validation.*?\n\s*-\s*check1\s*(PASS|FAIL)\n\s*-\s*check2\s*(PASS|FAIL)\n\s*-\s*check3\s*(PASS|FAIL)\n\s*-\s*overall\s*(PASS|FAIL)",
        text,
        re.DOTALL,
    )
    if check_block:
        out["stageb_checks"] = {
            "check1": check_block.group(1),
            "check2": check_block.group(2),
            "check3": check_block.group(3),
            "overall": check_block.group(4),
        }
    return out


def _latest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def collect_current_resume(prl_root: Path, run_log_arg: str) -> dict[str, Any]:
    log_path: Path | None = None
    if run_log_arg:
        candidate = Path(run_log_arg)
        if candidate.exists():
            log_path = candidate
    if log_path is None:
        log_candidates = list((prl_root / "outputs" / "logs").glob("step6_resume_*/master.log"))
        log_path = _latest_file(log_candidates)

    payload: dict[str, Any] = {
        "log_path": str(log_path) if log_path else None,
        "initial_missing_seeds": [],
        "train_started": [],
        "train_completed": [],
        "current_step": None,
    }
    if log_path and log_path.exists():
        lines = log_path.read_text().splitlines()
        started: dict[str, str] = {}
        completed: set[str] = set()
        for line in lines:
            m_missing = re.search(r"initial_missing_seeds=([0-9,<>none ]+)", line)
            if m_missing:
                token = m_missing.group(1).strip()
                if token and token != "<none>":
                    payload["initial_missing_seeds"] = [int(x) for x in token.split(",") if x.strip().isdigit()]

            m_start = re.search(r"\[STEP-START\]\s+([^\s]+)\s+::\s+(.+)$", line)
            if m_start:
                started[m_start.group(1)] = m_start.group(2).strip()
            m_end = re.search(r"\[STEP-END\]\s+([^\s]+)\s+::\s+(.+)$", line)
            if m_end:
                completed.add(m_end.group(1))

        train_started = sorted(
            [key for key in started if key.startswith("train_seed_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        train_completed = sorted(
            [key for key in completed if key.startswith("train_seed_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        payload["train_started"] = [int(name.split("_")[-1]) for name in train_started]
        payload["train_completed"] = [int(name.split("_")[-1]) for name in train_completed]

        current_step: str | None = None
        for step_name in started:
            if step_name not in completed:
                current_step = step_name
        payload["current_step"] = current_step

    report_dir = prl_root / "outputs" / "step3_u27" / "reports"
    seed_records: dict[int, dict[str, Any]] = {}
    if report_dir.exists():
        for meta_path in sorted(report_dir.glob("run_metadata_*.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            if meta.get("model_type") != "prl":
                continue
            if meta.get("config_path") != "configs/prl_100k_signals_u27.yaml":
                continue
            seed = meta.get("seed")
            if not isinstance(seed, int):
                continue
            artifact_paths = meta.get("artifact_paths") or meta.get("artifacts") or {}
            model_path_raw = artifact_paths.get("model_path")
            model_path = Path(model_path_raw) if model_path_raw else None
            if model_path and not model_path.is_absolute():
                model_path = (prl_root / model_path).resolve()
            created_at = str(meta.get("created_at", ""))
            prev = seed_records.get(seed)
            if prev is None or created_at > str(prev.get("created_at")):
                seed_records[seed] = {
                    "seed": seed,
                    "run_id": meta.get("run_id"),
                    "created_at": created_at,
                    "obs_dim_expected": meta.get("obs_dim_expected"),
                    "env_signature_hash": meta.get("env_signature_hash"),
                    "model_path": str(model_path) if model_path else None,
                    "model_exists": bool(model_path and model_path.exists()),
                    "metadata_path": str(meta_path),
                }
    payload["seed_records"] = [seed_records[seed] for seed in sorted(seed_records.keys())]
    payload["remaining_seeds"] = [seed for seed in range(10) if seed not in seed_records]

    eta_results: list[dict[str, Any]] = []
    for eta in ["079", "080", "082", "078", "081"]:
        run_root = prl_root / "outputs" / f"step6_fixedeta_tune_2022_2023_eta{eta}"
        acceptance = run_root / "acceptance_report.json"
        aggregate = run_root / "aggregate.csv"
        row: dict[str, Any] = {
            "eta": f"0.{eta}",
            "run_root": str(run_root),
            "acceptance_exists": acceptance.exists(),
            "aggregate_exists": aggregate.exists(),
            "overall_pass": None,
            "check2_min_positive_seeds": None,
            "check2_kappa_rows": [],
            "main_pos_kappa_rows": [],
        }
        if acceptance.exists():
            try:
                report = json.loads(acceptance.read_text())
                row["overall_pass"] = bool(report.get("overall_pass"))
                for check in report.get("checks", []):
                    if str(check.get("name", "")).startswith("check2_mode="):
                        per_kappa = check.get("details", {}).get("per_kappa", [])
                        row["check2_kappa_rows"] = per_kappa
                        positives = [
                            int(item.get("n_positive_delta_sharpe"))
                            for item in per_kappa
                            if str(item.get("n_positive_delta_sharpe", "")).isdigit()
                        ]
                        if positives:
                            row["check2_min_positive_seeds"] = min(positives)
                        break
            except Exception:
                pass

        if aggregate.exists():
            try:
                agg_rows = _read_csv_rows(aggregate)
                filtered = []
                for agg in agg_rows:
                    if agg.get("arm") != "main":
                        continue
                    kappa = _safe_float(agg.get("kappa"))
                    if kappa is None or kappa <= 0.0:
                        continue
                    filtered.append(
                        {
                            "kappa": kappa,
                            "median_sharpe": _safe_float(agg.get("median_sharpe")),
                            "median_turnover_exec": _safe_float(agg.get("median_turnover_exec")),
                            "collapse_rate": _safe_float(agg.get("collapse_rate")),
                        }
                    )
                row["main_pos_kappa_rows"] = filtered
            except Exception:
                pass
        eta_results.append(row)

    payload["eta_results"] = eta_results
    return payload


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_snapshot_text(
    *,
    job_ts: str,
    archive_experiments: list[ArchiveExperiment],
    spec_record: dict[str, Any],
    current: dict[str, Any],
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append(f"## Snapshot UTC {now}")
    lines.append("")
    lines.append("### 0) 스냅샷 메타")
    lines.append(f"- job_ts: {job_ts or 'N/A'}")
    lines.append(f"- run_log: {current.get('log_path')}")
    lines.append("")

    lines.append("### 1) 과거 동형 실험 4회 상세 (archive 기반)")
    if not archive_experiments:
        lines.append("- archive 실험을 찾지 못함")
    else:
        lines.append(
            "| exp_tag | n_seeds | sharpe_net_lin(mean±std) | cumret_net_lin(mean±std) | "
            "avg_turnover_exec(mean±std) | maxdd_net_lin(mean±std) | seed_duration_min(mean/min/max) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for exp in archive_experiments[:4]:
            lines.append(
                f"| {exp.exp_tag} | {exp.n_seeds} | "
                f"{_fmt_float(exp.sharpe_net_lin_mean)} ± {_fmt_float(exp.sharpe_net_lin_std)} | "
                f"{_fmt_float(exp.cumulative_return_net_lin_mean)} ± {_fmt_float(exp.cumulative_return_net_lin_std)} | "
                f"{_fmt_float(exp.avg_turnover_exec_mean)} ± {_fmt_float(exp.avg_turnover_exec_std)} | "
                f"{_fmt_float(exp.max_drawdown_net_lin_mean)} ± {_fmt_float(exp.max_drawdown_net_lin_std)} | "
                f"{_fmt_float(exp.duration_min_mean, 2)}/{_fmt_float(exp.duration_min_min, 2)}/{_fmt_float(exp.duration_min_max, 2)} |"
            )
        lines.append("")
        lines.append("#### 1-1) seed 단위 수치")
        for exp in archive_experiments[:4]:
            lines.append(f"- `{exp.exp_tag}`")
            lines.append("  - source_summary: " + str(exp.summary_path))
            lines.append("  - source_metrics: " + str(exp.metrics_path))
            for row in exp.per_seed_rows:
                lines.append(
                    "  - "
                    f"seed={row['seed']}, run_id={row['run_id']}, "
                    f"sharpe_net_lin={_fmt_float(row['sharpe_net_lin'])}, "
                    f"cumret_net_lin={_fmt_float(row['cumulative_return_net_lin'])}, "
                    f"avg_turnover_exec={_fmt_float(row['avg_turnover_exec'])}, "
                    f"maxdd_net_lin={_fmt_float(row['max_drawdown_net_lin'])}, "
                    f"duration_min={_fmt_float(row['duration_min'], 2)}"
                )
        lines.append("")

    lines.append("### 2) 2026-02-26 Stage6 기록 (명세 원문 복원)")
    if not spec_record:
        lines.append("- 명세에서 Stage6 기록 추출 실패")
    else:
        lines.append(f"- run_path: {spec_record.get('run_path')}")
        lines.append(f"- total_minutes: {spec_record.get('total_minutes')}")
        lines.append(f"- final_status: {spec_record.get('final_status')}")
        lines.append(f"- A1: {spec_record.get('a1')}")
        lines.append(f"- A2: {spec_record.get('a2')}")
        checks = spec_record.get("stageb_checks", {})
        if checks:
            lines.append(
                f"- StageB checks: check1={checks.get('check1')} check2={checks.get('check2')} "
                f"check3={checks.get('check3')} overall={checks.get('overall')}"
            )
        k5 = spec_record.get("kappa_0005")
        if k5:
            lines.append(
                f"- kappa=0.0005: positive={k5['positive']}/{k5['total']}, "
                f"median_delta_sharpe={k5['median_delta_sharpe']}"
            )
        k10 = spec_record.get("kappa_0010")
        if k10:
            lines.append(
                f"- kappa=0.0010: positive={k10['positive']}/{k10['total']}, "
                f"median_delta_sharpe={k10['median_delta_sharpe']}"
            )
        collapse = spec_record.get("collapse_rate")
        if collapse:
            lines.append(
                f"- collapse_rate={collapse['rate']} ({collapse['count']}/{collapse['total']})"
            )
    lines.append("")

    lines.append("### 3) 현재 재시작 실험 진행 상태")
    lines.append(f"- initial_missing_seeds: {current.get('initial_missing_seeds')}")
    lines.append(f"- train_started: {current.get('train_started')}")
    lines.append(f"- train_completed: {current.get('train_completed')}")
    lines.append(f"- current_step: {current.get('current_step')}")
    lines.append(f"- remaining_seeds: {current.get('remaining_seeds')}")
    lines.append("")
    lines.append("#### 3-1) seed 메타데이터 상세 (outputs/step3_u27/reports)")
    seed_records = current.get("seed_records", [])
    if not seed_records:
        lines.append("- 아직 생성된 seed metadata 없음")
    else:
        for row in seed_records:
            lines.append(
                "- "
                f"seed={row['seed']}, run_id={row['run_id']}, created_at={row['created_at']}, "
                f"obs_dim_expected={row['obs_dim_expected']}, env_signature_hash={row['env_signature_hash']}, "
                f"model_exists={row['model_exists']}, model_path={row['model_path']}"
            )
    lines.append("")

    lines.append("### 4) Validation eta 결과 추적 (0.079/0.080/0.082 중심)")
    eta_results = current.get("eta_results", [])
    lines.append("| eta | acceptance_exists | overall_pass | check2_min_positive_seeds | aggregate_exists |")
    lines.append("| --- | --- | --- | --- | --- |")
    for eta in eta_results:
        lines.append(
            f"| {eta['eta']} | {eta['acceptance_exists']} | {eta['overall_pass']} | "
            f"{eta['check2_min_positive_seeds']} | {eta['aggregate_exists']} |"
        )
    lines.append("")
    lines.append("#### 4-1) kappa 상세")
    for eta in eta_results:
        lines.append(f"- eta={eta['eta']}")
        if eta["check2_kappa_rows"]:
            for item in eta["check2_kappa_rows"]:
                lines.append(
                    "  - check2: "
                    f"kappa={item.get('kappa')}, n={item.get('n')}, "
                    f"n_positive={item.get('n_positive_delta_sharpe')}, "
                    f"median_delta_sharpe={item.get('median_delta_sharpe')}, pass={item.get('pass')}"
                )
        else:
            lines.append("  - check2: pending")
        if eta["main_pos_kappa_rows"]:
            for item in eta["main_pos_kappa_rows"]:
                lines.append(
                    "  - aggregate(main,kappa>0): "
                    f"kappa={item.get('kappa')}, median_sharpe={item.get('median_sharpe')}, "
                    f"median_turnover_exec={item.get('median_turnover_exec')}, collapse_rate={item.get('collapse_rate')}"
                )
        else:
            lines.append("  - aggregate(main,kappa>0): pending")
    lines.append("")

    lines.append("### 5) 루프 방지용 의사결정 규칙 (고정)")
    lines.append("- Rule-1: `eta 0.079/0.080/0.082` acceptance 완료 전에는 새 eta 추가 금지.")
    lines.append("- Rule-2: check2 최소값이 6 미만이면 Spec-D(0.078/0.081)로만 확장.")
    lines.append("- Rule-3: validation PASS 후보가 없으면 final 구간 실행 금지.")
    lines.append("- Rule-4: 매 실행 종료 시 `결과 분석` 파일에 snapshot 1회 추가 후 다음 액션 결정.")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    this_file = Path(__file__).resolve()
    prl_root = this_file.parents[1]
    repo_root = prl_root.parent
    spec_path = repo_root / "명세"
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    archive_experiments = collect_archive_experiments(prl_root)
    spec_record = collect_spec_stage6_record(spec_path)
    current = collect_current_resume(prl_root, args.run_log)
    snapshot = build_snapshot_text(
        job_ts=args.job_ts,
        archive_experiments=archive_experiments,
        spec_record=spec_record,
        current=current,
    )

    if args.mode == "overwrite":
        output_file.write_text(snapshot)
    else:
        if output_file.exists() and output_file.read_text().strip():
            with output_file.open("a") as handle:
                handle.write("\n")
                handle.write(snapshot)
        else:
            output_file.write_text(snapshot)

    print(f"[RESULT_ANALYSIS] wrote snapshot to: {output_file}")


if __name__ == "__main__":
    main()
