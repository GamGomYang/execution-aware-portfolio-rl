#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
PHASEC_TAG="${PHASEC_TAG:-u27_eta082_m070_cg03_100k_r1}"
PHASEC_CONFIG="${PHASEC_CONFIG:-configs/exp/${PHASEC_TAG}.yaml}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
FULL_SEEDS_RAW="${FULL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CHECK2_HARD="${CHECK2_HARD:-6}"
CHECK2_SOFT="${CHECK2_SOFT:-5}"
MAX_STEPS="${MAX_STEPS:-252}"
RUN_FULL_AUDIT="${RUN_FULL_AUDIT:-0}"
FULL_AUDIT_MAX_STEPS="${FULL_AUDIT_MAX_STEPS:-0}"
EVAL_START="${EVAL_START:-2022-01-01}"
EVAL_END="${EVAL_END:-2023-12-31}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."
read -r -a FULL_SEEDS <<< "$FULL_SEEDS_RAW"

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_eta082_phaseC_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR" "outputs/reports"

MODELSWAP_ROOT="outputs/modelswap_${PHASEC_TAG}"
STEP6_ROOT="outputs/step6_${PHASEC_TAG}"
FULL_AUDIT_ROOT="outputs/step6_${PHASEC_TAG}_fullwindow"
SUMMARY_MD="outputs/reports/u27_eta082_phaseC_summary_${JOB_TS}.md"

log() {
  echo "$1" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "[STEP-START] ${name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  log "[STEP-END] ${name} rc=${rc} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$rc"
}

ensure_selected_signals() {
  local selected_path="outputs/diagnostics/signal_scan_u27/selected_signals.json"
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
  ]
}
JSON
  log "[INFO] created bootstrap selected_signals: ${selected_path}"
}

if [[ ! -f "$PHASEC_CONFIG" ]]; then
  log "[ERROR] missing config: ${PHASEC_CONFIG}"
  exit 1
fi

ensure_selected_signals
log "[INFO] phase=C"
log "[INFO] phasec_tag=${PHASEC_TAG}"
log "[INFO] phasec_config=${PHASEC_CONFIG}"
log "[INFO] full_seeds=${FULL_SEEDS[*]}"
log "[INFO] max_steps=${MAX_STEPS}"
log "[INFO] run_full_audit=${RUN_FULL_AUDIT}"
log "[INFO] eval_start=${EVAL_START}"
log "[INFO] eval_end=${EVAL_END}"

run_step "modelswap_${PHASEC_TAG}" \
  "$PYTHON_BIN" scripts/run_model_swap_matrix.py \
    --config "$PHASEC_CONFIG" \
    --output-root "$MODELSWAP_ROOT" \
    --algos prl \
    --seeds "${FULL_SEEDS[@]}" \
    --etas 0.082 \
    --kappas 0.0 0.0005 0.001 \
    --rl-timesteps 100000 \
    --eval-start "${EVAL_START}" \
    --eval-end "${EVAL_END}" \
    --offline

run_step "step6_${PHASEC_TAG}" \
  "$PYTHON_BIN" scripts/step6_run_matrix.py \
    --config "$STEP6_CONFIG" \
    --model-type prl \
    --model-root "$MODELSWAP_ROOT" \
    --seed-model-mode independent \
    --seeds "${FULL_SEEDS[@]}" \
    --kappas 0.0 0.0005 0.001 \
    --etas 0.082 \
    --out "$STEP6_ROOT" \
    --offline \
    --max-steps "$MAX_STEPS"

run_step "acceptance_hard_${PHASEC_TAG}" \
  "$PYTHON_BIN" scripts/step6_check_acceptance.py \
    --paired "${STEP6_ROOT}/paired_delta.csv" \
    --aggregate "${STEP6_ROOT}/aggregate.csv" \
    --check2-min-positive-seeds "$CHECK2_HARD" \
    --out-md "${STEP6_ROOT}/acceptance_report_hard6.md" \
    --out-json "${STEP6_ROOT}/acceptance_report_hard6.json" \
    --no-fail-exit

run_step "acceptance_soft_${PHASEC_TAG}" \
  "$PYTHON_BIN" scripts/step6_check_acceptance.py \
    --paired "${STEP6_ROOT}/paired_delta.csv" \
    --aggregate "${STEP6_ROOT}/aggregate.csv" \
    --check2-min-positive-seeds "$CHECK2_SOFT" \
    --out-md "${STEP6_ROOT}/acceptance_report_soft5.md" \
    --out-json "${STEP6_ROOT}/acceptance_report_soft5.json" \
    --no-fail-exit

if [[ "$RUN_FULL_AUDIT" == "1" ]]; then
  run_step "step6_full_audit_${PHASEC_TAG}" \
    "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$STEP6_CONFIG" \
      --model-type prl \
      --model-root "$MODELSWAP_ROOT" \
      --seed-model-mode independent \
      --seeds "${FULL_SEEDS[@]}" \
      --kappas 0.0 0.0005 0.001 \
      --etas 0.082 \
      --out "$FULL_AUDIT_ROOT" \
      --offline \
      --max-steps "$FULL_AUDIT_MAX_STEPS"
fi

PHASEC_TAG="$PHASEC_TAG" STEP6_ROOT="$STEP6_ROOT" MODELSWAP_ROOT="$MODELSWAP_ROOT" SUMMARY_MD="$SUMMARY_MD" "$PYTHON_BIN" - <<'PYCODE' | tee -a "$MASTER_LOG"
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

phasec_tag = os.environ["PHASEC_TAG"]
step6_root = Path(os.environ["STEP6_ROOT"])
modelswap_root = Path(os.environ["MODELSWAP_ROOT"])
out_md = Path(os.environ["SUMMARY_MD"])

hard = json.loads((step6_root / "acceptance_report_hard6.json").read_text())
soft = json.loads((step6_root / "acceptance_report_soft5.json").read_text())
agg = list(csv.DictReader((step6_root / "aggregate.csv").open()))
paired = list(csv.DictReader((step6_root / "paired_delta.csv").open()))
main_k001 = next(r for r in agg if r.get("arm") == "main" and abs(float(r["kappa"]) - 0.001) < 1e-12)
base_k001 = next(r for r in agg if r.get("arm") == "baseline" and abs(float(r["kappa"]) - 0.001) < 1e-12)
deltas_k001 = [float(r["delta_sharpe"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12]
neg_seeds = [int(r["seed"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12 and float(r["delta_sharpe"]) < 0.0]
lines = []
lines.append("# U27 ETA082 Phase C Summary")
lines.append("")
lines.append(f"- phasec_tag: {phasec_tag}")
lines.append(f"- modelswap_root: {modelswap_root}")
lines.append(f"- step6_root: {step6_root}")
lines.append(f"- hard6_pass: {hard['overall_pass']}")
lines.append(f"- soft5_pass: {soft['overall_pass']}")
lines.append(f"- k001_main_median_sharpe: {main_k001['median_sharpe']}")
lines.append(f"- k001_baseline_median_sharpe: {base_k001['median_sharpe']}")
lines.append(f"- k001_mean_delta_sharpe: {sum(deltas_k001)/len(deltas_k001) if deltas_k001 else None}")
lines.append(f"- k001_positive: {sum(1 for x in deltas_k001 if x > 0.0)}")
lines.append(f"- k001_negative_seeds: {neg_seeds}")
lines.append(f"- k001_collapse_rate: {main_k001['collapse_rate']}")
out_md.write_text("\n".join(lines) + "\n")
print(f"SUMMARY_MD={out_md}")
PYCODE

log "[DONE] phase=C complete"
log "[DONE] master_log=${MASTER_LOG}"
