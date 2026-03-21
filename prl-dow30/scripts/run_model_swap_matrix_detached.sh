#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

CONFIG="${CONFIG:-configs/prl_100k_signals_u27.yaml}"
SEEDS="${SEEDS:-0 1 2}"
ETAS="${ETAS:-0.079 0.080 0.082}"
KAPPAS="${KAPPAS:-0.0005 0.001}"
ALGOS="${ALGOS:-prl sac ppo td3}"
RL_TIMESTEPS="${RL_TIMESTEPS:-100000}"
EVAL_START="${EVAL_START:-2022-01-01}"
EVAL_END="${EVAL_END:-2023-12-31}"
MINVAR_LOOKBACK="${MINVAR_LOOKBACK:-252}"
WAIT_SESSION="${WAIT_SESSION:-step6_resume_20260308}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="outputs/model_swap_matrix_${JOB_TS}"
LOG_DIR="outputs/logs/model_swap_matrix_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR" "$OUT_ROOT/reports"

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

log "[START] model-swap detached matrix :: ${JOB_TS}"
log "[INFO] python=${PYTHON_BIN}"
log "[INFO] config=${CONFIG}"
log "[INFO] algos=${ALGOS}"
log "[INFO] seeds=${SEEDS}"
log "[INFO] etas=${ETAS}"
log "[INFO] kappas=${KAPPAS}"
log "[INFO] rl_timesteps=${RL_TIMESTEPS}"
log "[INFO] eval_window=${EVAL_START}~${EVAL_END}"
log "[INFO] wait_session=${WAIT_SESSION:-<none>}"
log "[INFO] out_root=${OUT_ROOT}"

if [[ -n "$WAIT_SESSION" ]]; then
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    log "[WAIT] session '${WAIT_SESSION}' still running; sleep 120s"
    sleep 120
  done
  log "[WAIT-DONE] session '${WAIT_SESSION}' finished"
fi

run_step "run_model_swap_matrix" \
  "$PYTHON_BIN" scripts/run_model_swap_matrix.py \
    --config "$CONFIG" \
    --output-root "$OUT_ROOT" \
    --algos $ALGOS \
    --seeds $SEEDS \
    --etas $ETAS \
    --kappas $KAPPAS \
    --rl-timesteps "$RL_TIMESTEPS" \
    --eval-start "$EVAL_START" \
    --eval-end "$EVAL_END" \
    --minvar-lookback "$MINVAR_LOOKBACK" \
    --offline \
    --append-result-analysis \
    --result-analysis-file "/workspace/execution-aware-portfolio-rl/결과 분석"

log "[DONE] model-swap detached matrix completed :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "[DONE] reports=${OUT_ROOT}/reports"
