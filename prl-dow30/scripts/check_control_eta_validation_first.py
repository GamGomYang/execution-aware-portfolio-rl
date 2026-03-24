#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check validation-first control-eta run artifacts.")
    parser.add_argument("--run-root", type=str, required=True, help="Run root produced by the validation-first orchestration.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="checklists/control_eta_validation_first",
        help="Relative path under run-root for markdown/json checklist outputs.",
    )
    return parser.parse_args()


def _check(path: Path, description: str, *, required: bool = True) -> dict:
    exists = path.exists()
    status = "PASS" if exists else ("FAIL" if required else "SKIP")
    return {
        "path": str(path),
        "description": description,
        "required": required,
        "exists": exists,
        "status": status,
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_prefix = Path(args.output_prefix)
    out_dir = run_root / out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = run_root / f"{args.output_prefix}.md"
    json_path = run_root / f"{args.output_prefix}.json"

    train_root = run_root / "train_control"
    validation_root = run_root / "validation_eta"
    final_root = run_root / "final_eta"
    baselines_root = run_root / "external_baselines"
    pack_root = run_root / "paper_pack"
    config_root = run_root / "configs"

    implementation_checks = [
        _check(run_root, "Run root created"),
        _check(train_root, "train_control directory created"),
        _check(validation_root, "validation_eta directory created"),
        _check(final_root, "final_eta directory created", required=False),
        _check(baselines_root, "external_baselines directory created", required=False),
        _check(pack_root, "paper_pack directory created", required=False),
        _check(config_root / "snapshot_control.yaml", "Frozen training config emitted"),
        _check(config_root / "validation_eta.yaml", "Validation eval config emitted"),
        _check(config_root / "final_eta.yaml", "Final eval config emitted", required=False),
        _check(config_root / "materialization_meta.json", "Materialization metadata emitted"),
    ]

    execution_checks = [
        _check(validation_root / "aggregate.csv", "Validation aggregate report written"),
        _check(validation_root / "paired_delta.csv", "Validation paired report written"),
        _check(validation_root / "fig_frontier.png", "Validation frontier figure written", required=False),
        _check(validation_root / "selection" / "validation_eta_selection.json", "Validation eta selection JSON written"),
        _check(validation_root / "selection" / "validation_eta_selection.md", "Validation eta selection markdown written"),
        _check(final_root / "aggregate.csv", "Final/test aggregate report written", required=False),
        _check(final_root / "paired_delta.csv", "Final/test paired report written", required=False),
        _check(baselines_root / "aggregate.csv", "External baseline aggregate report written", required=False),
        _check(baselines_root / "protocol.json", "External baseline protocol JSON written", required=False),
        _check(pack_root / "README.md", "Paper pack README written", required=False),
        _check(pack_root / "protocol_lock.md", "Pack protocol lock markdown written", required=False),
        _check(pack_root / "tables" / "validation_selection.csv", "Validation selection table written", required=False),
        _check(pack_root / "tables" / "test_selected_vs_eta1.csv", "Test selected-vs-eta1 table written", required=False),
        _check(pack_root / "stats" / "selected_eta_vs_eta1_stats.csv", "Selected-vs-eta1 stats CSV written", required=False),
        _check(pack_root / "stats" / "selected_eta_seedwise_deltas.csv", "Selected-vs-eta1 seedwise deltas CSV written", required=False),
        _check(
            pack_root / "tables" / "test_selected_vs_external_baselines.csv",
            "Test selected-vs-external-baselines table written",
            required=False,
        ),
        _check(pack_root / "tables" / "diagnostic_selected_eta.csv", "Diagnostic selected-eta table written", required=False),
        _check(pack_root / "diagnostics" / "diagnostic_selected_eta_v2.csv", "Diagnostic selected-eta v2 table written", required=False),
        _check(
            pack_root / "diagnostics" / "representative_seed_metrics.json",
            "Representative selected-eta seed metadata written",
            required=False,
        ),
        _check(pack_root / "figures" / "fig_selected_trace.png", "Selected-trace figure written", required=False),
        _check(pack_root / "figures" / "fig_validation_frontier.png", "Validation frontier figure written", required=False),
        _check(pack_root / "figures" / "fig_seed_scatter.png", "Seed scatter figure written", required=False),
    ]

    payload = {
        "run_root": str(run_root),
        "implementation_checks": implementation_checks,
        "execution_checks": execution_checks,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Control Eta Validation-First Checklist",
        "",
        f"- run_root: {run_root}",
        "",
        "## Implementation Checklist",
        "",
        "| status | description | path |",
        "| --- | --- | --- |",
    ]
    for item in implementation_checks:
        lines.append(f"| {item['status']} | {item['description']} | {item['path']} |")
    lines.extend(
        [
            "",
            "## Execution Checklist",
            "",
            "| status | description | path |",
            "| --- | --- | --- |",
        ]
    )
    for item in execution_checks:
        lines.append(f"| {item['status']} | {item['description']} | {item['path']} |")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"WROTE_MD={md_path}")
    print(f"WROTE_JSON={json_path}")


if __name__ == "__main__":
    main()
