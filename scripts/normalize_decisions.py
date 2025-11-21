"""Utilities to sort decision logs and diagnose timeline issues."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_GAP_THRESHOLD_DAYS = 7


@dataclass
class TimelineIssue:
    previous_date: str
    current_date: str
    gap_days: int


def _parse_line(line: str) -> Dict:
    record = json.loads(line)
    missing = [k for k in ("date", "action", "beta", "stress_signal", "reward", "portfolio_return", "explanations") if k not in record]
    if missing:
        raise ValueError(f"Record is missing required keys: {missing}")
    return record


def _load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(_parse_line(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _to_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, DATE_FORMAT)


def _detect_backward_jumps(records: Sequence[Dict]) -> List[Tuple[int, str, str]]:
    backwards: List[Tuple[int, str, str]] = []
    prev_dt: datetime | None = None
    prev_date_str: str | None = None
    for idx, rec in enumerate(records):
        curr_date = rec["date"]
        curr_dt = _to_datetime(curr_date)
        if prev_dt is not None and curr_dt < prev_dt:
            backwards.append((idx, prev_date_str or "", curr_date))
        prev_dt = curr_dt
        prev_date_str = curr_date
    return backwards


def _detect_duplicates(sorted_records: Sequence[Dict]) -> List[Tuple[str, int]]:
    counts = Counter(rec["date"] for rec in sorted_records)
    duplicates = [(date, count) for date, count in counts.items() if count > 1]
    duplicates.sort(key=lambda item: item[0])
    return duplicates


def _detect_missing(sorted_records: Sequence[Dict], threshold_days: int) -> List[TimelineIssue]:
    missing: List[TimelineIssue] = []
    prev_dt: datetime | None = None
    prev_date_str: str | None = None
    for rec in sorted_records:
        curr_dt = _to_datetime(rec["date"])
        curr_date_str = rec["date"]
        if prev_dt is not None:
            delta = (curr_dt - prev_dt).days
            if delta > threshold_days:
                missing.append(
                    TimelineIssue(
                        previous_date=prev_date_str or "",
                        current_date=curr_date_str,
                        gap_days=delta - 1,
                    )
                )
        prev_dt = curr_dt
        prev_date_str = curr_date_str
    return missing


def _deduplicate_records(sorted_records: Sequence[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    seen_dates = set()
    for record in sorted_records:
        date = record["date"]
        if date in seen_dates:
            continue
        deduped.append(record)
        seen_dates.add(date)
    return deduped


def _write_sorted(records: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _write_report(
    report_path: Path,
    source: Path,
    sorted_path: Path,
    missing: Sequence[TimelineIssue],
    duplicates: Sequence[Tuple[str, int]],
    backwards: Sequence[Tuple[int, str, str]],
) -> None:
    lines: List[str] = []
    lines.append(f"Timeline analysis for {source}")
    lines.append(f"Sorted output: {sorted_path}")
    lines.append("")

    def _append_section(title: str, values: Sequence, formatter, limit: int = 50) -> None:
        lines.append(title)
        if not values:
            lines.append("- None detected")
            lines.append("")
            return
        total = len(values)
        display_count = min(total, limit)
        for idx in range(display_count):
            lines.append(formatter(values[idx]))
        if total > limit:
            lines.append(f"- ... ({total - limit} additional entries truncated)")
        lines.append("")

    _append_section(
        "Missing date ranges (gap greater than threshold):",
        missing,
        lambda issue: f"- {issue.previous_date} -> {issue.current_date} (missing {issue.gap_days} day(s))",
        limit=50,
    )
    _append_section(
        "Duplicated dates:",
        duplicates,
        lambda item: f"- {item[0]} (occurs {item[1]} times)",
        limit=50,
    )
    _append_section(
        "Backward time jumps (original order):",
        backwards,
        lambda item: f"- index {item[0]}: {item[1]} -> {item[2]}",
        limit=50,
    )
    lines.append("Threshold for gap detection: %d days" % DEFAULT_GAP_THRESHOLD_DAYS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def normalize_log(
    input_path: Path,
    sorted_path: Path | None,
    report_path: Path | None,
) -> None:
    records = _load_records(input_path)
    sorted_records = sorted(records, key=lambda rec: _to_datetime(rec["date"]))
    deduped_records = _deduplicate_records(sorted_records)
    if sorted_path is None:
        sorted_path = input_path.with_name(f"{input_path.stem}_sorted{input_path.suffix}")
    _write_sorted(deduped_records, sorted_path)
    missing = _detect_missing(deduped_records, DEFAULT_GAP_THRESHOLD_DAYS)
    duplicates = _detect_duplicates(sorted_records)
    backwards = _detect_backward_jumps(records)
    if report_path is None:
        report_path = input_path.with_name("timeline_report.txt")
    _write_report(report_path, input_path, sorted_path, missing, duplicates, backwards)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort decisions.jsonl and report timeline issues.")
    parser.add_argument("input", type=Path, help="Path to the original decisions.jsonl file")
    parser.add_argument(
        "--sorted-output",
        type=Path,
        default=None,
        help="Path to write the sorted JSONL file (defaults to <input>_sorted.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Timeline report output path (defaults to <input dir>/timeline_report.txt)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    normalize_log(args.input, args.sorted_output, args.report)


if __name__ == "__main__":
    main()
