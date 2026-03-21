#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
CHECK2_MIN_POSITIVE_SEEDS="${CHECK2_MIN_POSITIVE_SEEDS:-6}"
FINAL_ETA_DEFAULT="${FINAL_ETA_DEFAULT:-082}"
PRIMARY_ETAS=(079 080 082)
EXPAND_ETAS=(078 081)

SPEC_FILE="/workspace/execution-aware-portfolio-rl/명세"
RESULT_ANALYSIS_FILE="/workspace/execution-aware-portfolio-rl/결과 분석"
RESULT_ANALYSIS_SCRIPT="scripts/update_result_analysis_log.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/step6_gate_final_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR" "outputs/reports"

log() {
  echo "$1" | tee -a "$MASTER_LOG"
}

append_docs() {
  local title="$1"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo ""
    echo "## [${ts}] ${title}"
    cat
  } | tee -a "$SPEC_FILE" "$RESULT_ANALYSIS_FILE" >> "$MASTER_LOG"
}

run_cmd() {
  local step_name="$1"
  shift
  log "[STEP-START] ${step_name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  log "[STEP-END] ${step_name} rc=${rc} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$rc"
}

latest_resume_log() {
  ls -1dt outputs/logs/step6_resume_*/master.log 2>/dev/null | head -n 1 || true
}

eta_acceptance_exists() {
  local eta="$1"
  local run_root="outputs/step6_fixedeta_tune_2022_2023_eta${eta}"
  [[ -f "${run_root}/aggregate.csv" && -f "${run_root}/paired_delta.csv" && -f "${run_root}/acceptance_report.json" ]]
}

run_eta_validation() {
  local eta="$1"
  local cfg="configs/step6_fixedeta_tune_2022_2023_eta${eta}_seed10.yaml"
  local run_root="outputs/step6_fixedeta_tune_2022_2023_eta${eta}"
  run_cmd "step6_run_eta_${eta}_manual" \
    "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$cfg" \
      --model-type prl \
      --model-root outputs/step3_u27 \
      --seed-model-mode independent \
      --offline \
      --max-steps 0

  run_cmd "acceptance_eta_${eta}_manual" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "${run_root}/paired_delta.csv" \
      --aggregate "${run_root}/aggregate.csv" \
      --check2-min-positive-seeds "$CHECK2_MIN_POSITIVE_SEEDS" \
      --out-md "${run_root}/acceptance_report.md" \
      --out-json "${run_root}/acceptance_report.json" \
      --no-fail-exit
}

build_gate_report() {
  local label="$1"
  shift
  local etas=("$@")
  ETAS_STR="${etas[*]}" \
  CHECK2_MIN_POSITIVE_SEEDS="$CHECK2_MIN_POSITIVE_SEEDS" \
  JOB_TS="$JOB_TS" \
  LABEL="$label" \
  "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

etas = [token.strip() for token in os.environ["ETAS_STR"].split() if token.strip()]
threshold = int(os.environ["CHECK2_MIN_POSITIVE_SEEDS"])
job_ts = os.environ["JOB_TS"]
label = os.environ["LABEL"]

out_csv = Path(f"outputs/reports/step6_gate_{label}_{job_ts}.csv")
out_md = Path(f"outputs/reports/step6_gate_{label}_{job_ts}.md")
out_json = Path(f"outputs/reports/step6_gate_{label}_{job_ts}.json")


def _extract_check2_min_positive(report: dict) -> int | None:
    for check in report.get("checks", []):
        name = str(check.get("name", ""))
        if not name.startswith("check2_mode="):
            continue
        vals: list[int] = []
        for item in check.get("details", {}).get("per_kappa", []):
            try:
                vals.append(int(item.get("n_positive_delta_sharpe")))
            except Exception:
                continue
        if vals:
            return min(vals)
    return None


rows: list[dict[str, object]] = []
for eta in etas:
    run_root = Path(f"outputs/step6_fixedeta_tune_2022_2023_eta{eta}")
    report_path = run_root / "acceptance_report.json"
    aggregate_path = run_root / "aggregate.csv"

    exists = report_path.exists() and aggregate_path.exists()
    overall_pass = None
    check2_min_positive = None
    median_delta_floor = None
    collapse_rate_max = None
    qualifies = False
    reason = "missing_acceptance"

    if exists:
        report = json.loads(report_path.read_text())
        overall_pass = bool(report.get("overall_pass"))
        check2_min_positive = _extract_check2_min_positive(report)

        with aggregate_path.open(newline="") as handle:
            agg_rows = list(csv.DictReader(handle))
        pos_main = []
        for item in agg_rows:
            if item.get("arm") != "main":
                continue
            try:
                kappa = float(item.get("kappa", "nan"))
            except Exception:
                continue
            if kappa <= 0.0:
                continue
            pos_main.append(item)
        if pos_main:
            try:
                median_delta_floor = min(float(item.get("median_delta_sharpe")) for item in pos_main)
            except Exception:
                median_delta_floor = None
            try:
                collapse_rate_max = max(float(item.get("collapse_rate")) for item in pos_main)
            except Exception:
                collapse_rate_max = None

        check2_ok = check2_min_positive is not None and check2_min_positive >= threshold
        collapse_ok = collapse_rate_max is None or collapse_rate_max <= 0.05
        qualifies = bool(overall_pass) and check2_ok and collapse_ok

        if qualifies:
            reason = "qualified"
        elif not overall_pass:
            reason = "overall_pass_false"
        elif not check2_ok:
            reason = "check2_below_threshold"
        elif not collapse_ok:
            reason = "collapse_rate_high"
        else:
            reason = "unknown"

    rows.append(
        {
            "eta": f"0.{eta}",
            "acceptance_exists": exists,
            "overall_pass": overall_pass,
            "check2_min_positive_seeds": check2_min_positive,
            "check2_threshold": threshold,
            "median_delta_sharpe_pos_kappa_floor": median_delta_floor,
            "collapse_rate_pos_kappa_max": collapse_rate_max,
            "qualifies": qualifies,
            "reason": reason,
            "run_root": str(run_root),
        }
    )

if not rows:
    rows.append(
        {
            "eta": "",
            "acceptance_exists": False,
            "overall_pass": None,
            "check2_min_positive_seeds": None,
            "check2_threshold": threshold,
            "median_delta_sharpe_pos_kappa_floor": None,
            "collapse_rate_pos_kappa_max": None,
            "qualifies": False,
            "reason": "empty_eta_list",
            "run_root": "",
        }
    )

fieldnames = list(rows[0].keys())
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

qualified = [row for row in rows if bool(row.get("qualifies"))]

def _score(row: dict[str, object]) -> tuple[float, float]:
    delta = row.get("median_delta_sharpe_pos_kappa_floor")
    collapse = row.get("collapse_rate_pos_kappa_max")
    delta_v = float(delta) if delta is not None else -1e9
    collapse_v = float(collapse) if collapse is not None else 1e9
    return (delta_v, -collapse_v)

best_eta = ""
if qualified:
    best = sorted(qualified, key=_score, reverse=True)[0]
    best_eta = str(best["eta"]).replace("0.", "")

payload = {
    "label": label,
    "job_ts": job_ts,
    "check2_threshold": threshold,
    "best_eta": best_eta,
    "qualified_count": len(qualified),
    "rows": rows,
}
out_json.write_text(json.dumps(payload, indent=2))

lines = [f"# Step6 Gate Report ({label})", ""]
lines.append(f"- job_ts: {job_ts}")
lines.append(f"- check2_threshold: {threshold}")
lines.append(f"- qualified_count: {len(qualified)}")
lines.append(f"- best_eta: {('0.' + best_eta) if best_eta else '<none>'}")
lines.append("")
lines.append("| eta | exists | overall_pass | check2 | median_delta_floor | collapse_rate_max | qualifies | reason |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
for row in rows:
    lines.append(
        f"| {row['eta']} | {row['acceptance_exists']} | {row['overall_pass']} | {row['check2_min_positive_seeds']} | "
        f"{row['median_delta_sharpe_pos_kappa_floor']} | {row['collapse_rate_pos_kappa_max']} | {row['qualifies']} | {row['reason']} |"
    )
lines.append("")
lines.append(f"- csv: {out_csv}")
lines.append(f"- json: {out_json}")
out_md.write_text("\n".join(lines) + "\n")

print(f"REPORT_CSV={out_csv}")
print(f"REPORT_MD={out_md}")
print(f"REPORT_JSON={out_json}")
print(f"BEST_ETA={best_eta}")
print(f"QUALIFIED_COUNT={len(qualified)}")
PYCODE
}

append_docs "Step6 Gate Pipeline START (${JOB_TS})" <<EOF
- pipeline_log: ${MASTER_LOG}
- primary_etas: 0.079, 0.080, 0.082
- expand_etas_if_needed: 0.078, 0.081
- final_eta_default: 0.${FINAL_ETA_DEFAULT}
- check2_threshold: ${CHECK2_MIN_POSITIVE_SEEDS}
- objective: resume step6 validation -> gate -> optional expansion -> final test
EOF

log "[START] step6 gate/final pipeline :: ${JOB_TS}"
log "[INFO] python=${PYTHON_BIN}"
log "[INFO] check2_threshold=${CHECK2_MIN_POSITIVE_SEEDS}"

resume_rc=0
run_cmd "step6_resume_detached_reentry" env PYTHON_BIN="$PYTHON_BIN" scripts/run_step6_resume_detached.sh || resume_rc=$?
resume_log="$(latest_resume_log)"
log "[INFO] latest_resume_log=${resume_log:-<none>}"
log "[INFO] step6_resume_reentry_rc=${resume_rc}"

if [[ -n "${resume_log}" && -f "${resume_log}" ]]; then
  run_cmd "result_analysis_snapshot_after_step6_resume" \
    "$PYTHON_BIN" "$RESULT_ANALYSIS_SCRIPT" \
      --output-file "$RESULT_ANALYSIS_FILE" \
      --mode append \
      --job-ts "$JOB_TS" \
      --run-log "$resume_log" || true
fi

append_docs "Step6 Resume Reentry Completed (${JOB_TS})" <<EOF
- step6_resume_rc: ${resume_rc}
- step6_resume_log: ${resume_log:-<none>}
- note: if primary eta outputs are missing, manual fallback validation will run.
EOF

missing_primary=()
for eta in "${PRIMARY_ETAS[@]}"; do
  if ! eta_acceptance_exists "$eta"; then
    missing_primary+=("$eta")
  fi
done

if [[ "${#missing_primary[@]}" -gt 0 ]]; then
  log "[WARN] missing primary eta outputs after reentry: ${missing_primary[*]}"
  for eta in "${missing_primary[@]}"; do
    run_eta_validation "$eta"
  done
else
  log "[INFO] primary eta outputs already present: ${PRIMARY_ETAS[*]}"
fi

primary_gate_raw="$(build_gate_report "primary" "${PRIMARY_ETAS[@]}")"
printf '%s\n' "$primary_gate_raw" | tee -a "$MASTER_LOG"
primary_best_eta="$(printf '%s\n' "$primary_gate_raw" | awk -F= '/^BEST_ETA=/{print $2}')"
primary_qualified_count="$(printf '%s\n' "$primary_gate_raw" | awk -F= '/^QUALIFIED_COUNT=/{print $2}')"
primary_report_md="$(printf '%s\n' "$primary_gate_raw" | awk -F= '/^REPORT_MD=/{print $2}')"
primary_report_csv="$(printf '%s\n' "$primary_gate_raw" | awk -F= '/^REPORT_CSV=/{print $2}')"

append_docs "Primary Gate Decision (${JOB_TS})" <<EOF
- qualified_count: ${primary_qualified_count:-0}
- best_eta_primary: ${primary_best_eta:-<none>}
- report_csv: ${primary_report_csv:-<none>}
- report_md: ${primary_report_md:-<none>}
EOF

selected_eta="$primary_best_eta"
all_gate_etas=("${PRIMARY_ETAS[@]}")
expanded=false
if [[ -z "${selected_eta}" ]]; then
  expanded=true
  log "[INFO] no qualified eta in primary gate. run expansion etas: ${EXPAND_ETAS[*]}"
  for eta in "${EXPAND_ETAS[@]}"; do
    run_eta_validation "$eta"
  done
  all_gate_etas=("${PRIMARY_ETAS[@]}" "${EXPAND_ETAS[@]}")
  expanded_gate_raw="$(build_gate_report "expanded" "${all_gate_etas[@]}")"
  printf '%s\n' "$expanded_gate_raw" | tee -a "$MASTER_LOG"
  selected_eta="$(printf '%s\n' "$expanded_gate_raw" | awk -F= '/^BEST_ETA=/{print $2}')"
  expanded_report_md="$(printf '%s\n' "$expanded_gate_raw" | awk -F= '/^REPORT_MD=/{print $2}')"
  expanded_report_csv="$(printf '%s\n' "$expanded_gate_raw" | awk -F= '/^REPORT_CSV=/{print $2}')"
  expanded_qualified_count="$(printf '%s\n' "$expanded_gate_raw" | awk -F= '/^QUALIFIED_COUNT=/{print $2}')"
  append_docs "Expanded Gate Decision (${JOB_TS})" <<EOF
- expanded: true
- qualified_count: ${expanded_qualified_count:-0}
- best_eta_expanded: ${selected_eta:-<none>}
- report_csv: ${expanded_report_csv:-<none>}
- report_md: ${expanded_report_md:-<none>}
EOF
fi

final_eta="$FINAL_ETA_DEFAULT"
if [[ -n "${selected_eta}" ]]; then
  final_eta="$selected_eta"
fi
log "[INFO] final_eta_selected=0.${final_eta} (expanded=${expanded})"

final_cfg="configs/step6_fixedeta_final_test_eta${final_eta}_seed10.yaml"
final_root="outputs/step6_fixedeta_final_test_eta${final_eta}"
if [[ ! -f "$final_cfg" ]]; then
  log "[ERROR] final config missing: ${final_cfg}"
  exit 2
fi

run_cmd "step6_final_test_eta_${final_eta}" \
  "$PYTHON_BIN" scripts/step6_run_matrix.py \
    --config "$final_cfg" \
    --model-type prl \
    --model-root outputs/step3_u27 \
    --seed-model-mode independent \
    --offline \
    --max-steps 0

run_cmd "acceptance_final_eta_${final_eta}" \
  "$PYTHON_BIN" scripts/step6_check_acceptance.py \
    --paired "${final_root}/paired_delta.csv" \
    --aggregate "${final_root}/aggregate.csv" \
    --check2-min-positive-seeds "$CHECK2_MIN_POSITIVE_SEEDS" \
    --out-md "${final_root}/acceptance_report.md" \
    --out-json "${final_root}/acceptance_report.json" \
    --no-fail-exit

append_docs "Step6 Final Test Completed (${JOB_TS})" <<EOF
- final_eta_selected: 0.${final_eta}
- final_config: ${final_cfg}
- final_root: ${final_root}
- final_acceptance_json: ${final_root}/acceptance_report.json
- pipeline_log: ${MASTER_LOG}
EOF

log "[DONE] step6 gate/final pipeline completed :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "[DONE] pipeline_log=${MASTER_LOG}"
