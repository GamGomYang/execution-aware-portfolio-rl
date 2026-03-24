#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a validation eta operating point from step6 eta-frontier runs.")
    parser.add_argument("--root", type=str, required=True, help="Validation step6 root directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write selection artifacts into.")
    parser.add_argument(
        "--baseline-eta",
        type=float,
        default=1.0,
        help="Reference eta used for paired delta diagnostics.",
    )
    parser.add_argument(
        "--positive-kappas",
        type=str,
        default="0.0005,0.001",
        help="Comma-separated positive-cost kappas used in the selection score.",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=0.95,
        help="Qualify etas whose validation score is within this fraction of the best score.",
    )
    return parser.parse_args()


def _parse_float_set(raw: str) -> list[float]:
    out: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def _iter_metrics_paths(root: Path) -> list[Path]:
    return sorted(set(root.glob("kappa_*/*/seed_*/metrics.csv")) | set(root.glob("kappa_*/seed_*/metrics.csv")))


def _parse_dir_value(name: str, prefix: str) -> float | int:
    if not name.startswith(prefix):
        raise ValueError(f"Expected prefix {prefix!r} in {name!r}")
    raw = name.split(prefix, 1)[1]
    return float(raw) if "." in raw else int(raw)


def _load_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in _iter_metrics_paths(root):
        seed_dir = metrics_path.parent
        seed = int(_parse_dir_value(seed_dir.name, "seed_"))

        parent_dir = seed_dir.parent
        if parent_dir.name.startswith("kappa_"):
            eta = None
            kappa_dir = parent_dir
        else:
            eta_dir = parent_dir
            kappa_dir = eta_dir.parent
            if eta_dir.name.startswith("eta_"):
                eta = float(_parse_dir_value(eta_dir.name, "eta_"))
            else:
                eta = None

        kappa = float(_parse_dir_value(kappa_dir.name, "kappa_"))
        with metrics_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
        if first is None:
            continue
        eta_value = first.get("eta_requested") or first.get("eta")
        if eta_value in (None, "", "nan") and eta is not None:
            eta_value = eta
        if eta_value in (None, "", "nan"):
            continue
        rows.append(
            {
                "kappa": kappa,
                "seed": seed,
                "eta": float(eta_value),
                "sharpe_net_lin": float(first["sharpe_net_lin"]),
                "avg_turnover_exec": float(first["avg_turnover_exec"]),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _qualifying_score(best_score: float, relative_threshold: float) -> float:
    if best_score > 0:
        return float(best_score * relative_threshold)
    if best_score < 0:
        return float(best_score / relative_threshold)
    return 0.0


def _build_rows(
    runs: list[dict[str, Any]],
    *,
    baseline_eta: float,
    positive_kappas: list[float],
    relative_threshold: float,
) -> tuple[list[dict[str, Any]], float | None]:
    by_kappa_eta: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    by_kappa_seed_eta: dict[tuple[float, int, float], dict[str, Any]] = {}
    for row in runs:
        key = (float(row["kappa"]), float(row["eta"]))
        by_kappa_eta[key].append(row)
        by_kappa_seed_eta[(float(row["kappa"]), int(row["seed"]), float(row["eta"]))] = row

    all_etas = sorted({float(row["eta"]) for row in runs}, reverse=True)
    summary_rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    for eta in all_etas:
        pos_medians: list[float] = []
        pos_delta_medians: list[float] = []
        pos_turnovers: list[float] = []
        n_pairs = 0
        for kappa in positive_kappas:
            group = by_kappa_eta.get((float(kappa), float(eta)), [])
            if not group:
                continue
            sharpe_values = [float(item["sharpe_net_lin"]) for item in group]
            turnover_values = [float(item["avg_turnover_exec"]) for item in group]
            pos_medians.append(_median(sharpe_values))
            pos_turnovers.append(_median(turnover_values))

            deltas: list[float] = []
            for item in group:
                base = by_kappa_seed_eta.get((float(kappa), int(item["seed"]), float(baseline_eta)))
                if base is None:
                    continue
                deltas.append(float(item["sharpe_net_lin"]) - float(base["sharpe_net_lin"]))
            if deltas:
                pos_delta_medians.append(_median(deltas))
                n_pairs += len(deltas)

        score = _mean(pos_medians)
        row = {
            "eta": float(eta),
            "n_positive_kappas": int(len(pos_medians)),
            "n_pairs_vs_eta1": int(n_pairs),
            "score_mean_median_sharpe_pos_kappa": score,
            "score_mean_median_delta_sharpe_vs_eta1_pos_kappa": _mean(pos_delta_medians),
            "median_turnover_exec_pos_kappa_mean": _mean(pos_turnovers),
            "qualifies": False,
            "selected": False,
        }
        summary_rows.append(row)
        if row["n_positive_kappas"] > 0 and score > best_score:
            best_score = score

    if best_score == float("-inf"):
        return summary_rows, None

    threshold_score = _qualifying_score(best_score, float(relative_threshold))
    qualified = [row for row in summary_rows if row["n_positive_kappas"] > 0 and row["score_mean_median_sharpe_pos_kappa"] >= threshold_score]
    selected_eta = None
    if qualified:
        selected_eta = max(float(row["eta"]) for row in qualified)
        for row in summary_rows:
            if row["n_positive_kappas"] > 0 and row["score_mean_median_sharpe_pos_kappa"] >= threshold_score:
                row["qualifies"] = True
            if float(row["eta"]) == float(selected_eta):
                row["selected"] = True
    return summary_rows, selected_eta


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    lines = [
        "# Validation Eta Selection Report",
        "",
        f"- root: {payload['root']}",
        f"- baseline_eta: {payload['baseline_eta']}",
        f"- positive_kappas: {', '.join(str(x) for x in payload['positive_kappas'])}",
        f"- relative_threshold: {payload['relative_threshold']}",
        f"- selected_eta: {payload['selected_eta'] if payload['selected_eta'] is not None else '<none>'}",
        "",
        "## Rule",
        "",
        "Select the largest eta whose validation score is within the configured fraction of the best score.",
        "The score is the mean of per-kappa median `sharpe_net_lin` over positive transaction-cost regimes.",
        "",
        "## Summary",
        "",
        "| eta | n_pos_kappa | n_pairs_vs_eta1 | score_mean_median_sharpe_pos_kappa | score_mean_median_delta_sharpe_vs_eta1_pos_kappa | median_turnover_exec_pos_kappa_mean | qualifies | selected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['eta']} | {row['n_positive_kappas']} | {row['n_pairs_vs_eta1']} | "
            f"{row['score_mean_median_sharpe_pos_kappa']:.6f} | "
            f"{row['score_mean_median_delta_sharpe_vs_eta1_pos_kappa']:.6f} | "
            f"{row['median_turnover_exec_pos_kappa_mean']:.6f} | "
            f"{row['qualifies']} | {row['selected']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(root)
    if not runs:
        raise FileNotFoundError(f"No metrics.csv runs found under: {root}")

    positive_kappas = _parse_float_set(args.positive_kappas)
    rows, selected_eta = _build_rows(
        runs,
        baseline_eta=float(args.baseline_eta),
        positive_kappas=positive_kappas,
        relative_threshold=float(args.relative_threshold),
    )

    payload = {
        "root": str(root),
        "baseline_eta": float(args.baseline_eta),
        "positive_kappas": positive_kappas,
        "relative_threshold": float(args.relative_threshold),
        "selected_eta": selected_eta,
        "rows": rows,
    }
    _write_csv(output_dir / "validation_eta_selection.csv", rows)
    (output_dir / "validation_eta_selection.json").write_text(json.dumps(payload, indent=2))
    _write_md(output_dir / "validation_eta_selection.md", rows, payload)

    if selected_eta is None:
        print("SELECTED_ETA=")
    else:
        print(f"SELECTED_ETA={selected_eta}")


if __name__ == "__main__":
    main()
