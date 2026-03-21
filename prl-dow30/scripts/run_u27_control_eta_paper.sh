#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
CURRENT_CONFIG="${CURRENT_CONFIG:-configs/prl_100k_signals_u27_eta082_current.yaml}"
SEEDS_RAW="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ETAS_RAW="${ETAS:-1.0 0.5 0.2 0.1 0.082 0.05 0.02}"
KAPPAS_RAW="${KAPPAS:-0.0 0.0005 0.001}"
SELECTED_ETA="${SELECTED_ETA:-0.082}"
BASELINE_ETA="${BASELINE_ETA:-1.0}"
FINAL_START="${FINAL_START:-2024-01-01}"
FINAL_END="${FINAL_END:-2025-12-31}"
FORWARD_START="${FORWARD_START:-2026-01-01}"
MAX_STEPS="${MAX_STEPS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

read -r -a SEEDS <<< "$SEEDS_RAW"
read -r -a ETAS <<< "$ETAS_RAW"
read -r -a KAPPAS <<< "$KAPPAS_RAW"

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="outputs/paper_control_eta_${JOB_TS}"
TRAIN_ROOT="${RUN_ROOT}/train_control_100k"
FINAL_ROOT="${RUN_ROOT}/step6_final_eta_frontier"
FORWARD_ROOT="${RUN_ROOT}/step6_forward_eta082"
PACK_ROOT="outputs/releases/u27_control_eta_paper_${JOB_TS}"
LOG_DIR="outputs/logs/u27_control_eta_paper_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
SNAPSHOT_CONFIG="configs/snapshots/prl_100k_signals_u27_eta082_control_paper_${JOB_TS}.yaml"
SIGNAL_SNAPSHOT="configs/signal_sets/paper_control/u27_eta082_control_paper_${JOB_TS}.json"
FINAL_CONFIG="configs/eval/u27_control_eta_frontier_final_${JOB_TS}.yaml"
FORWARD_CONFIG="configs/eval/u27_control_eta_forward_${JOB_TS}.yaml"
META_OUT="outputs/reports/u27_control_eta_paper_materialization_${JOB_TS}.json"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$PACK_ROOT"

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

log "[INFO] phase=control_eta_paper"
log "[INFO] current_config=${CURRENT_CONFIG}"
log "[INFO] seeds=${SEEDS[*]}"
log "[INFO] etas=${ETAS[*]}"
log "[INFO] kappas=${KAPPAS[*]}"
log "[INFO] selected_eta=${SELECTED_ETA}"
log "[INFO] final_window=${FINAL_START}~${FINAL_END}"
log "[INFO] forward_start=${FORWARD_START}"

run_step "materialize_control_eta_paper_configs" \
  "$PYTHON_BIN" scripts/materialize_u27_control_eta_paper_configs.py \
    --current-config "$CURRENT_CONFIG" \
    --snapshot-config-out "$SNAPSHOT_CONFIG" \
    --signal-snapshot-out "$SIGNAL_SNAPSHOT" \
    --final-config-out "$FINAL_CONFIG" \
    --forward-config-out "$FORWARD_CONFIG" \
    --meta-out "$META_OUT" \
    --job-ts "$JOB_TS" \
    --final-start "$FINAL_START" \
    --final-end "$FINAL_END" \
    --forward-start "$FORWARD_START" \
    --train-output-root "$TRAIN_ROOT" \
    --final-output-root "$FINAL_ROOT" \
    --forward-output-root "$FORWARD_ROOT"

for seed in "${SEEDS[@]}"; do
  run_step "train_control_seed${seed}" \
    "$PYTHON_BIN" scripts/run_train.py \
      --config "$SNAPSHOT_CONFIG" \
      --model-type prl \
      --seed "$seed" \
      --offline \
      --output-root "$TRAIN_ROOT"
done

run_step "step6_final_eta_frontier" \
  "$PYTHON_BIN" scripts/step6_run_matrix.py \
    --config "$FINAL_CONFIG" \
    --model-type prl \
    --model-root "$TRAIN_ROOT" \
    --seed-model-mode independent \
    --seeds "${SEEDS[@]}" \
    --kappas "${KAPPAS[@]}" \
    --etas "${ETAS[@]}" \
    --out "$FINAL_ROOT" \
    --offline \
    --max-steps "$MAX_STEPS"

run_step "step6_final_eta_frontier_reports" \
  "$PYTHON_BIN" scripts/step6_build_reports.py \
    --root "$FINAL_ROOT"

run_step "step6_forward_eta082" \
  "$PYTHON_BIN" scripts/step6_run_matrix.py \
    --config "$FORWARD_CONFIG" \
    --model-type prl \
    --model-root "$TRAIN_ROOT" \
    --seed-model-mode independent \
    --seeds "${SEEDS[@]}" \
    --kappas "${KAPPAS[@]}" \
    --etas "$SELECTED_ETA" \
    --out "$FORWARD_ROOT" \
    --offline \
    --max-steps "$MAX_STEPS"

run_step "step6_forward_eta082_reports" \
  "$PYTHON_BIN" scripts/step6_build_reports.py \
    --root "$FORWARD_ROOT"

run_step "build_control_eta_paper_pack" \
  "$PYTHON_BIN" scripts/build_u27_control_eta_paper_pack.py \
    --final-root "$FINAL_ROOT" \
    --forward-root "$FORWARD_ROOT" \
    --output-dir "$PACK_ROOT" \
    --selected-eta "$SELECTED_ETA" \
    --baseline-eta "$BASELINE_ETA" \
    --meta-json "$META_OUT"

log "[DONE] phase=control_eta_paper complete"
log "[DONE] master_log=${MASTER_LOG}"
log "[DONE] train_root=${TRAIN_ROOT}"
log "[DONE] final_root=${FINAL_ROOT}"
log "[DONE] forward_root=${FORWARD_ROOT}"
log "[DONE] pack_root=${PACK_ROOT}"
