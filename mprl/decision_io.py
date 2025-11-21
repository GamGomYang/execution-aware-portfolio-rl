"""Helpers for reading decision logs and preferring sorted snapshots when available."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def resolve_decision_log_path(path: str | Path) -> Path:
    """Return the sorted variant of a decision log if it exists."""
    orig_path = Path(path)
    sorted_candidate = orig_path.with_name(f"{orig_path.stem}_sorted{orig_path.suffix}")
    if sorted_candidate.exists():
        return sorted_candidate
    return orig_path


def iter_decision_records(path: str | Path) -> Iterator[Dict]:
    """Yield JSON records from a decision log, preferring the sorted version."""
    resolved = resolve_decision_log_path(path)
    with resolved.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_decision_records(path: str | Path) -> Tuple[List[Dict], Path]:
    records = list(iter_decision_records(path))
    if not records:
        raise ValueError(f"No decision records found in {path}")
    resolved = resolve_decision_log_path(path)
    return records, resolved
