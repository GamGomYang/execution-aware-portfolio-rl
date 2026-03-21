#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
PHASEA_SUMMARY_CSV="${PHASEA_SUMMARY_CSV:-}"
PHASEA_TOP_K="${PHASEA_TOP_K:-2}"
AUTO_SELECT="${AUTO_SELECT:-1}"
BASELINE_TAG="${BASELINE_TAG:-u27_eta082_alpha_ctrl_20k_r1}"
PHASEB_CANDIDATES_RAW="${PHASEB_CANDIDATES:-}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_tune_2022_2023_eta082_seed10.yaml}"
FULL_SEEDS_RAW="${FULL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CHECK2_HARD="${CHECK2_HARD:-6}"
CHECK2_SOFT="${CHECK2_SOFT:-5}"
MAX_STEPS="${MAX_STEPS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

if [[ -z "$PHASEB_CANDIDATES_RAW" && "$AUTO_SELECT" == "1" ]]; then
  SELECTOR_CMD=(
    "$PYTHON_BIN" scripts/select_u27_alpha_first_batch_phaseA.py
    --baseline-tag "$BASELINE_TAG"
    --top-k "$PHASEA_TOP_K"
    --print-shell
  )
  if [[ -n "$PHASEA_SUMMARY_CSV" ]]; then
    SELECTOR_CMD+=(--summary-csv "$PHASEA_SUMMARY_CSV")
  fi

  SELECTED_TAGS=""
  while IFS='=' read -r key value; do
    case "$key" in
      BASELINE_TAG) BASELINE_TAG="$value" ;;
      SELECTED_TAGS) SELECTED_TAGS="$value" ;;
      PHASEA_SUMMARY_CSV) PHASEA_SUMMARY_CSV="$value" ;;
    esac
  done < <("${SELECTOR_CMD[@]}")
  PHASEB_CANDIDATES_RAW="$SELECTED_TAGS"
fi

if [[ -z "$PHASEB_CANDIDATES_RAW" ]]; then
  echo "[ERROR] No Phase B candidates resolved. Provide PHASEB_CANDIDATES or enable AUTO_SELECT with a valid Phase A summary."
  exit 1
fi

echo "[INFO] alpha_phaseB baseline_tag=${BASELINE_TAG}"
echo "[INFO] alpha_phaseB candidates=${PHASEB_CANDIDATES_RAW}"
if [[ -n "$PHASEA_SUMMARY_CSV" ]]; then
  echo "[INFO] alpha_phaseB phaseA_summary_csv=${PHASEA_SUMMARY_CSV}"
fi

env \
  PYTHON_BIN="$PYTHON_BIN" \
  STEP6_CONFIG="$STEP6_CONFIG" \
  FULL_SEEDS="$FULL_SEEDS_RAW" \
  BASELINE_TAG="$BASELINE_TAG" \
  PHASEB_CANDIDATES="$PHASEB_CANDIDATES_RAW" \
  CHECK2_HARD="$CHECK2_HARD" \
  CHECK2_SOFT="$CHECK2_SOFT" \
  MAX_STEPS="$MAX_STEPS" \
  bash scripts/run_u27_eta082_phaseB.sh
