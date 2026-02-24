from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step6 acceptance criteria from aggregate/paired CSV files.")
    parser.add_argument("--paired", type=str, required=True, help="Path to paired_delta.csv")
    parser.add_argument("--aggregate", type=str, required=True, help="Path to aggregate.csv")
    parser.add_argument(
        "--check2-mode",
        choices=["delta_sharpe_gt0"],
        default="delta_sharpe_gt0",
        help="Definition lock for checklist-2.",
    )
    parser.add_argument(
        "--check2-min-positive-seeds",
        type=int,
        default=6,
        help="Minimum #seeds with positive metric for checklist-2.",
    )
    parser.add_argument(
        "--check3-min-median-delta-sharpe",
        type=float,
        default=-0.005,
        help="Lower bound for checklist-3 median delta_sharpe at kappa=0.",
    )
    parser.add_argument("--out-md", type=str, default=None, help="Output markdown report path.")
    parser.add_argument("--out-json", type=str, default=None, help="Output JSON report path.")
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Always exit with code 0 (default exits non-zero when acceptance fails).",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _to_float(value: str) -> float:
    return float(value.strip())


def _key_kappa(value: float) -> str:
    return f"{value:.12f}"


def _compute_check1(aggregate_rows: list[dict[str, str]]) -> CheckResult:
    main_rows = [row for row in aggregate_rows if str(row.get("arm")) == "main"]
    pos_rows = [row for row in main_rows if _to_float(row["kappa"]) > 0.0]
    per_kappa = []
    for row in sorted(pos_rows, key=lambda r: _to_float(r["kappa"])):
        kappa = _to_float(row["kappa"])
        med = _to_float(row["median_sharpe"])
        per_kappa.append({"kappa": kappa, "median_sharpe": med, "pass": med > 0.0})
    passed = bool(per_kappa) and all(item["pass"] for item in per_kappa)
    return CheckResult(
        name="kappa>0_main_median_sharpe_gt_0",
        passed=passed,
        details={"per_kappa": per_kappa},
    )


def _compute_check2(
    paired_rows: list[dict[str, str]],
    *,
    min_positive_seeds: int,
) -> CheckResult:
    buckets: dict[str, list[float]] = {}
    for row in paired_rows:
        kappa = _to_float(row["kappa"])
        if kappa <= 0.0:
            continue
        key = _key_kappa(kappa)
        buckets.setdefault(key, []).append(_to_float(row["delta_sharpe"]))

    per_kappa = []
    for key in sorted(buckets.keys(), key=float):
        values = buckets[key]
        n = len(values)
        n_pos = sum(1 for v in values if v > 0.0)
        item = {
            "kappa": float(key),
            "n": n,
            "n_positive_delta_sharpe": n_pos,
            "median_delta_sharpe": median(values),
            "mean_delta_sharpe": mean(values),
            "min_delta_sharpe": min(values),
            "max_delta_sharpe": max(values),
            "pass": n_pos >= min_positive_seeds,
        }
        per_kappa.append(item)

    passed = bool(per_kappa) and all(item["pass"] for item in per_kappa)
    return CheckResult(
        name=f"check2_mode=delta_sharpe_gt0_min_positive_seeds={min_positive_seeds}",
        passed=passed,
        details={"per_kappa": per_kappa},
    )


def _compute_check3(
    paired_rows: list[dict[str, str]],
    *,
    min_median_delta_sharpe: float,
) -> CheckResult:
    values = [_to_float(row["delta_sharpe"]) for row in paired_rows if abs(_to_float(row["kappa"])) < 1e-12]
    if not values:
        return CheckResult(
            name="kappa0_median_delta_sharpe_floor",
            passed=False,
            details={
                "n": 0,
                "reason": "No rows found for kappa=0.",
                "threshold": min_median_delta_sharpe,
            },
        )
    med = median(values)
    return CheckResult(
        name="kappa0_median_delta_sharpe_floor",
        passed=med >= min_median_delta_sharpe,
        details={
            "n": len(values),
            "median_delta_sharpe": med,
            "mean_delta_sharpe": mean(values),
            "min_delta_sharpe": min(values),
            "max_delta_sharpe": max(values),
            "threshold": min_median_delta_sharpe,
        },
    )


def _render_md(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Step6 Acceptance Report")
    lines.append(f"- overall_pass: {payload['overall_pass']}")
    lines.append(f"- check2_mode: {payload['check2_mode']}")
    lines.append(f"- paired_csv: {payload['paired_csv']}")
    lines.append(f"- aggregate_csv: {payload['aggregate_csv']}")
    lines.append("")
    lines.append("## Checklist")

    for result in payload["checks"]:
        lines.append(f"### {result['name']}")
        lines.append(f"- pass: {result['pass']}")
        details = result["details"]
        per_kappa = details.get("per_kappa")
        if isinstance(per_kappa, list):
            for item in per_kappa:
                item_text = ", ".join(f"{k}={v}" for k, v in item.items())
                lines.append(f"- {item_text}")
        else:
            for key, value in details.items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    paired_path = Path(args.paired)
    aggregate_path = Path(args.aggregate)
    if not paired_path.exists():
        raise FileNotFoundError(f"paired_delta.csv not found: {paired_path}")
    if not aggregate_path.exists():
        raise FileNotFoundError(f"aggregate.csv not found: {aggregate_path}")

    out_md = Path(args.out_md) if args.out_md else paired_path.parent / "acceptance_report.md"
    out_json = Path(args.out_json) if args.out_json else paired_path.parent / "acceptance_report.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    paired_rows = _read_csv(paired_path)
    aggregate_rows = _read_csv(aggregate_path)

    check1 = _compute_check1(aggregate_rows)
    check2 = _compute_check2(
        paired_rows,
        min_positive_seeds=int(args.check2_min_positive_seeds),
    )
    check3 = _compute_check3(
        paired_rows,
        min_median_delta_sharpe=float(args.check3_min_median_delta_sharpe),
    )
    checks = [check1, check2, check3]
    overall_pass = all(check.passed for check in checks)

    payload = {
        "overall_pass": overall_pass,
        "check2_mode": args.check2_mode,
        "paired_csv": str(paired_path),
        "aggregate_csv": str(aggregate_path),
        "checks": [
            {
                "name": check.name,
                "pass": check.passed,
                "details": check.details,
            }
            for check in checks
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False))
    out_md.write_text(_render_md(payload))

    status = "PASS" if overall_pass else "FAIL"
    print(f"ACCEPTANCE {status}")
    print(f"- markdown: {out_md}")
    print(f"- json: {out_json}")

    if not overall_pass and not args.no_fail_exit:
        sys.exit(2)


if __name__ == "__main__":
    main()
