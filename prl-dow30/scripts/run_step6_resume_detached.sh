#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
TRAIN_CONFIG="configs/prl_100k_signals_u27.yaml"
MODEL_ROOT="outputs/step3_u27"
CHECK2_MIN_POSITIVE_SEEDS="${CHECK2_MIN_POSITIVE_SEEDS:-6}"
ETA_LIST=(079 080 082)
RESULT_ANALYSIS_FILE="/workspace/execution-aware-portfolio-rl/결과 분석"
RESULT_ANALYSIS_SCRIPT="scripts/update_result_analysis_log.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  echo "[HINT] Create venv first and set PYTHON_BIN=/path/to/python if needed."
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/step6_resume_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
INVENTORY_MD="outputs/reports/next_run_inventory_${JOB_TS}.md"
INVENTORY_JSON="outputs/reports/next_run_inventory_${JOB_TS}.json"
COMPARE_CSV="outputs/reports/validation_eta_compare_${JOB_TS}.csv"
COMPARE_MD="outputs/reports/validation_eta_compare_${JOB_TS}.md"

mkdir -p "$LOG_DIR" "outputs/reports"

log() {
  echo "$1" | tee -a "$MASTER_LOG"
}

run_step() {
  local step_name="$1"
  shift
  log "[STEP-START] ${step_name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  log "[STEP-END] ${step_name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

snapshot_result_analysis() {
  log "[STEP-START] result_analysis_snapshot :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" "$RESULT_ANALYSIS_SCRIPT" \
    --output-file "$RESULT_ANALYSIS_FILE" \
    --mode append \
    --job-ts "$JOB_TS" \
    --run-log "$MASTER_LOG" 2>&1 | tee -a "$MASTER_LOG"
  log "[STEP-END] result_analysis_snapshot :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

ensure_selected_signals() {
  local selected_path="../outputs/diagnostics/signal_scan_u27/selected_signals.json"
  if [[ -f "$selected_path" ]]; then
    log "[INFO] selected_signals already exists: ${selected_path}"
    return
  fi
  mkdir -p "$(dirname "$selected_path")"
  cat > "$selected_path" <<'JSON'
{
  "selected_signals": [
    "reversal_5d",
    "short_term_reversal"
  ],
  "ic_start": "2010-01-01",
  "ic_end": "2021-12-31",
  "note": "bootstrap file for step6 resume workflow"
}
JSON
  log "[INFO] created bootstrap selected_signals: ${selected_path}"
}

refresh_inventory() {
  local stage="$1"
  STAGE="$stage" \
  JOB_TS="$JOB_TS" \
  TRAIN_CONFIG="$TRAIN_CONFIG" \
  MODEL_ROOT="$MODEL_ROOT" \
  INVENTORY_MD="$INVENTORY_MD" \
  INVENTORY_JSON="$INVENTORY_JSON" \
  "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
job_ts = os.environ["JOB_TS"]
stage = os.environ["STAGE"]
train_config = os.environ["TRAIN_CONFIG"]
model_root = Path(os.environ["MODEL_ROOT"])
inventory_md = Path(os.environ["INVENTORY_MD"])
inventory_json = Path(os.environ["INVENTORY_JSON"])
meta_dir = model_root / "reports"

def _safe(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, cwd=root).strip()
    except Exception:
        return "unknown"

branch = _safe(["git", "branch", "--show-current"])
head = _safe(["git", "rev-parse", "--short", "HEAD"])

latest_by_seed: dict[int, dict] = {}
if meta_dir.exists():
    for path in sorted(meta_dir.glob("run_metadata_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("model_type") != "prl":
            continue
        if payload.get("config_path") != train_config:
            continue
        seed = payload.get("seed")
        if not isinstance(seed, int):
            continue
        artifact_paths = payload.get("artifact_paths") or payload.get("artifacts") or {}
        model_path_raw = artifact_paths.get("model_path")
        if not model_path_raw:
            continue
        model_path = Path(model_path_raw)
        if not model_path.is_absolute():
            model_path = (root / model_path).resolve()
        if not model_path.exists():
            continue
        created_at = str(payload.get("created_at", ""))
        prev = latest_by_seed.get(seed)
        if prev is None or created_at > prev["created_at"]:
            latest_by_seed[seed] = {
                "seed": seed,
                "created_at": created_at,
                "run_id": payload.get("run_id"),
                "model_path": str(model_path),
                "metadata_path": str(path),
                "obs_dim_expected": payload.get("obs_dim_expected"),
                "env_signature_hash": payload.get("env_signature_hash"),
            }

found_seeds = sorted(latest_by_seed.keys())
missing_seeds = [seed for seed in range(10) if seed not in latest_by_seed]

payload = {
    "job_ts": job_ts,
    "stage": stage,
    "branch": branch,
    "head": head,
    "train_config": train_config,
    "model_root": str(model_root),
    "found_seeds": found_seeds,
    "missing_seeds": missing_seeds,
    "latest_records": [latest_by_seed[seed] for seed in found_seeds],
}
inventory_json.write_text(json.dumps(payload, indent=2))

lines: list[str] = []
lines.append(f"# Next Run Inventory ({job_ts})")
lines.append("")
lines.append(f"- stage: {stage}")
lines.append(f"- branch: {branch}")
lines.append(f"- head: {head}")
lines.append(f"- train_config: {train_config}")
lines.append(f"- model_root: {model_root}")
lines.append(f"- found_seeds: {found_seeds}")
lines.append(f"- missing_seeds: {missing_seeds}")
lines.append("")
lines.append("## latest metadata per seed")
if found_seeds:
    for seed in found_seeds:
        row = latest_by_seed[seed]
        lines.append(
            f"- seed={seed}, run_id={row['run_id']}, created_at={row['created_at']}, "
            f"obs_dim_expected={row['obs_dim_expected']}, env_signature_hash={row['env_signature_hash']}"
        )
else:
    lines.append("- none")
inventory_md.write_text("\n".join(lines) + "\n")

print(",".join(str(seed) for seed in missing_seeds))
PYCODE
}

run_validation_eta() {
  local eta="$1"
  local cfg="configs/step6_fixedeta_tune_2022_2023_eta${eta}_seed10.yaml"
  local out_root="outputs/step6_fixedeta_tune_2022_2023_eta${eta}"
  run_step "step6_run_eta_${eta}" \
    "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$cfg" \
      --model-type prl \
      --model-root "$MODEL_ROOT" \
      --seed-model-mode independent \
      --offline \
      --max-steps 0

  run_step "acceptance_eta_${eta}" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "${out_root}/paired_delta.csv" \
      --aggregate "${out_root}/aggregate.csv" \
      --check2-min-positive-seeds "$CHECK2_MIN_POSITIVE_SEEDS" \
      --out-md "${out_root}/acceptance_report.md" \
      --out-json "${out_root}/acceptance_report.json" \
      --no-fail-exit

  snapshot_result_analysis
}

build_validation_summary() {
  ETAS="${ETA_LIST[*]}" \
  CHECK2_MIN_POSITIVE_SEEDS="$CHECK2_MIN_POSITIVE_SEEDS" \
  COMPARE_CSV="$COMPARE_CSV" \
  COMPARE_MD="$COMPARE_MD" \
  "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

etas = [token.strip() for token in os.environ["ETAS"].split() if token.strip()]
threshold = int(os.environ["CHECK2_MIN_POSITIVE_SEEDS"])
compare_csv = Path(os.environ["COMPARE_CSV"])
compare_md = Path(os.environ["COMPARE_MD"])

rows: list[dict[str, object]] = []

def _extract_check2_min_positive(report: dict) -> int:
    for check in report.get("checks", []):
        name = str(check.get("name", ""))
        if name.startswith("check2_mode="):
            values = []
            for item in check.get("details", {}).get("per_kappa", []):
                try:
                    values.append(int(item.get("n_positive_delta_sharpe")))
                except Exception:
                    continue
            return min(values) if values else -1
    return -1

for eta in etas:
    run_root = Path(f"outputs/step6_fixedeta_tune_2022_2023_eta{eta}")
    report_path = run_root / "acceptance_report.json"
    aggregate_path = run_root / "aggregate.csv"
    collapse_rate = None
    median_turnover_exec = None
    median_delta_sharpe_pos_kappa = None
    if aggregate_path.exists():
        with aggregate_path.open(newline="") as handle:
            aggregate_rows = list(csv.DictReader(handle))
        main_pos = []
        for item in aggregate_rows:
            if item.get("arm") != "main":
                continue
            try:
                kappa = float(item.get("kappa", "nan"))
            except Exception:
                continue
            if kappa <= 0.0:
                continue
            main_pos.append(item)
        if main_pos:
            try:
                collapse_rate = max(float(item.get("collapse_rate", "nan")) for item in main_pos)
            except Exception:
                collapse_rate = None
            try:
                median_turnover_exec = sum(float(item.get("median_turnover_exec", "nan")) for item in main_pos) / len(main_pos)
            except Exception:
                median_turnover_exec = None
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for check in report.get("checks", []):
            if str(check.get("name", "")).startswith("check2_mode="):
                vals = []
                for item in check.get("details", {}).get("per_kappa", []):
                    try:
                        vals.append(float(item.get("median_delta_sharpe")))
                    except Exception:
                        continue
                if vals:
                    median_delta_sharpe_pos_kappa = min(vals)
                break
    overall_pass = bool(report.get("overall_pass")) if report else False
    check2_min_positive = _extract_check2_min_positive(report) if report else -1
    rows.append(
        {
            "eta": f"0.{eta}",
            "overall_pass": overall_pass,
            "check2_min_positive_seeds": check2_min_positive,
            "check2_threshold": threshold,
            "check2_gap": check2_min_positive - threshold,
            "median_delta_sharpe_pos_kappa_floor": median_delta_sharpe_pos_kappa,
            "collapse_rate_pos_kappa_max": collapse_rate,
            "median_turnover_exec_pos_kappa_mean": median_turnover_exec,
            "run_root": str(run_root),
        }
    )

fieldnames = list(rows[0].keys()) if rows else []
compare_csv.parent.mkdir(parents=True, exist_ok=True)
with compare_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = ["# Validation ETA Compare", ""]
lines.append(f"- check2_threshold: {threshold}")
lines.append("")
lines.append("| eta | overall_pass | check2_min_positive_seeds | gap | median_delta_sharpe_pos_kappa_floor | collapse_rate_pos_kappa_max |")
lines.append("| --- | --- | --- | --- | --- | --- |")
for row in rows:
    lines.append(
        f"| {row['eta']} | {row['overall_pass']} | {row['check2_min_positive_seeds']} | "
        f"{row['check2_gap']} | {row['median_delta_sharpe_pos_kappa_floor']} | {row['collapse_rate_pos_kappa_max']} |"
    )
lines.append("")
lines.append(f"- csv: {compare_csv}")
compare_md.write_text("\n".join(lines) + "\n")
PYCODE
}

log "[START] step6 resume workflow :: ${JOB_TS}"
log "[INFO] python=${PYTHON_BIN}"
log "[INFO] train_config=${TRAIN_CONFIG}"
log "[INFO] model_root=${MODEL_ROOT}"
log "[INFO] etas=${ETA_LIST[*]}"

run_step "ensure_selected_signals" ensure_selected_signals
snapshot_result_analysis

missing_csv="$(refresh_inventory "before_spec_b")"
log "[INFO] initial_missing_seeds=${missing_csv:-<none>}"
snapshot_result_analysis

if [[ -n "$missing_csv" ]]; then
  IFS=',' read -r -a missing_seeds <<< "$missing_csv"
  for seed in "${missing_seeds[@]}"; do
    [[ -z "$seed" ]] && continue
    run_step "train_seed_${seed}" \
      "$PYTHON_BIN" scripts/run_train.py \
        --config "$TRAIN_CONFIG" \
        --model-type prl \
        --seed "$seed" \
        --offline
    snapshot_result_analysis
  done
else
  log "[INFO] seed 0~9 already complete. Skip Spec-B train."
fi

missing_after_csv="$(refresh_inventory "after_spec_b")"
log "[INFO] missing_seeds_after_spec_b=${missing_after_csv:-<none>}"
snapshot_result_analysis
if [[ -n "$missing_after_csv" ]]; then
  log "[ERROR] Still missing seeds after Spec-B training: ${missing_after_csv}"
  exit 2
fi

for eta in "${ETA_LIST[@]}"; do
  run_validation_eta "$eta"
done

run_step "build_validation_summary" build_validation_summary
snapshot_result_analysis

log "[DONE] step6 resume workflow completed :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "[DONE] inventory_md=${INVENTORY_MD}"
log "[DONE] compare_csv=${COMPARE_CSV}"
log "[DONE] compare_md=${COMPARE_MD}"
