#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
CONFIG_PATH="${CONFIG_PATH:-configs/prl_100k_signals_u27_eta082_current.yaml}"
JOB_TAG_BASE="${JOB_TAG_BASE:-u27_eta082_current}"
JOB_TAG="${JOB_TAG:-${JOB_TAG_BASE}_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_STEPS="${MAX_STEPS:-252}"
RUN_FULL_AUDIT="${RUN_FULL_AUDIT:-1}"
FULL_AUDIT_MAX_STEPS="${FULL_AUDIT_MAX_STEPS:-0}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
EVAL_START="${EVAL_START:-2022-01-01}"
EVAL_END="${EVAL_END:-2023-12-31}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$ROOT/$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

PYTHON_BIN="$PYTHON_BIN" \
PHASEC_TAG="$JOB_TAG" \
PHASEC_CONFIG="$CONFIG_PATH" \
STEP6_CONFIG="$STEP6_CONFIG" \
MAX_STEPS="$MAX_STEPS" \
RUN_FULL_AUDIT="$RUN_FULL_AUDIT" \
FULL_AUDIT_MAX_STEPS="$FULL_AUDIT_MAX_STEPS" \
EVAL_START="$EVAL_START" \
EVAL_END="$EVAL_END" \
bash scripts/run_u27_eta082_phaseC.sh
