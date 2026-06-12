"""Compute share intervals for the actionability report-card tables.

The reporting convention is intentionally simple:
- Wilson score intervals for rows with n >= 30.
- Clopper-Pearson exact intervals for rows with n < 30.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import comb, sqrt
from pathlib import Path


Z_975 = 1.959963984540054


@dataclass(frozen=True)
class ShareRow:
    table: str
    domain: str
    interface: str
    friction: str
    suboptimal: int
    seeds: int

    @property
    def agreement(self) -> float:
        return 1.0 - self.suboptimal / self.seeds

    @property
    def share(self) -> float:
        return self.suboptimal / self.seeds

    @property
    def ci_type(self) -> str:
        return "Wilson" if self.seeds >= 30 else "Clopper--Pearson"


Q2_ROWS = [
    ShareRow("q2", "Event-micro", r"threshold $\tau=0.55$", "0.50", 69, 100),
    ShareRow("q2", "Event-micro", r"threshold $\tau=0.55$", "1.00", 99, 100),
    ShareRow("q2", r"Traffic Top-\(k\)", r"budget $k=249$", "0.50", 100, 100),
    ShareRow("q2", r"Traffic Top-\(k\)", r"budget $k=249$", "1.00", 100, 100),
    ShareRow("q2", "Inventory", "replenishment", "0.50", 74, 100),
    ShareRow("q2", "Inventory", "replenishment", "1.00", 99, 100),
    ShareRow("q2", "Load-following", "secondary", "0.25", 7, 10),
    ShareRow("q2", "Load-following", "secondary", "0.50", 7, 10),
    ShareRow("q2", "Load-following", "secondary", "1.00", 8, 10),
]

EVENT_INTERFACE_ROWS = [
    ShareRow("event_interface", "Fixed threshold", "", "0.50", 69, 100),
    ShareRow("event_interface", "Fixed threshold", "", "1.00", 99, 100),
    ShareRow("event_interface", "Hysteresis threshold", "", "0.50", 55, 100),
    ShareRow("event_interface", "Hysteresis threshold", "", "1.00", 82, 100),
]


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    phat = k / n
    denom = 1.0 + Z_975 * Z_975 / n
    center = (phat + Z_975 * Z_975 / (2.0 * n)) / denom
    half = Z_975 * sqrt((phat * (1.0 - phat) + Z_975 * Z_975 / (4.0 * n)) / n) / denom
    return center - half, center + half


def beta_cdf_integer(x: float, a: int, b: int) -> float:
    # For integer a,b: I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1,j) x^j (1-x)^(a+b-1-j).
    total = 0.0
    m = a + b - 1
    for j in range(a, m + 1):
        total += comb(m, j) * (x**j) * ((1.0 - x) ** (m - j))
    return total


def beta_ppf_integer(p: float, a: int, b: int) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if beta_cdf_integer(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta_ppf_integer(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf_integer(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi


def share_interval(row: ShareRow) -> tuple[float, float]:
    if row.seeds >= 30:
        return wilson_interval(row.suboptimal, row.seeds)
    return clopper_pearson_interval(row.suboptimal, row.seeds)


def fmt_float(x: float) -> str:
    return f"{x:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_float(lo)}, {fmt_float(hi)}]"


def plain_label(value: str) -> str:
    return (
        value.replace(r"Traffic Top-\(k\)", "Traffic Top-k")
        .replace(r"\tau", "tau")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
    )


def audit_rows(rows: list[ShareRow]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        lo, hi = share_interval(row)
        out.append(
            {
                "table": row.table,
                "domain": plain_label(row.domain),
                "interface": plain_label(row.interface),
                "friction": row.friction,
                "seeds": str(row.seeds),
                "agreement": f"{row.agreement:.2f}",
                "suboptimal_share_count": f"{row.share:.2f} ({row.suboptimal}/{row.seeds})",
                "ci_type": row.ci_type.replace("--", "-"),
                "ci_lower": fmt_float(lo),
                "ci_upper": fmt_float(hi),
                "ci": fmt_ci(lo, hi),
            }
        )
    return out


def write_audit_csv(path: Path, rows: list[ShareRow]) -> None:
    records = audit_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    audit_path = repo / "results" / "tables" / "share_interval_audit.csv"
    write_audit_csv(audit_path, Q2_ROWS + EVENT_INTERFACE_ROWS)
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
