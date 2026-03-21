#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
WINNER_TAG_20K="${WINNER_TAG_20K:-}"
PHASEB_SUMMARY_CSV="${PHASEB_SUMMARY_CSV:-}"
AUTO_SELECT_WINNER="${AUTO_SELECT_WINNER:-1}"
AUTO_PROMOTE="${AUTO_PROMOTE:-1}"
PROMOTION_META="${PROMOTION_META:-outputs/reports/u27_alpha_first_batch_promoted_100k_materialization.json}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
MAX_STEPS="${MAX_STEPS:-0}"
RUN_FULL_AUDIT="${RUN_FULL_AUDIT:-1}"
FULL_AUDIT_MAX_STEPS="${FULL_AUDIT_MAX_STEPS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

if [[ "$AUTO_PROMOTE" == "1" ]]; then
  PROMOTE_CMD=(
    env
    PYTHON_BIN="$PYTHON_BIN"
    PHASEB_SUMMARY_CSV="$PHASEB_SUMMARY_CSV"
    WINNER_TAG_20K="$WINNER_TAG_20K"
    AUTO_SELECT_WINNER="$AUTO_SELECT_WINNER"
    META_OUT="$PROMOTION_META"
    bash scripts/run_u27_alpha_first_batch_promote_100k.sh
  )

  while IFS='=' read -r key value; do
    case "$key" in
      WINNER_TAG_20K) WINNER_TAG_20K="$value" ;;
      PHASEB_SUMMARY_CSV) PHASEB_SUMMARY_CSV="$value" ;;
      PROMOTION_META) PROMOTION_META="$value" ;;
    esac
  done < <("${PROMOTE_CMD[@]}")
fi

if [[ -z "$WINNER_TAG_20K" ]]; then
  echo "[ERROR] No alpha winner resolved for Phase C."
  exit 1
fi

WINNER_TAG_100K="${WINNER_TAG_20K/_20k_r1/_100k_r1}"
PHASEC_CONFIG="configs/exp/${WINNER_TAG_100K}.yaml"

if [[ ! -f "$PHASEC_CONFIG" ]]; then
  echo "[ERROR] Missing promoted config for Phase C: ${PHASEC_CONFIG}"
  echo "[ERROR] Run scripts/run_u27_alpha_first_batch_promote_100k.sh first or set AUTO_PROMOTE=1."
  exit 1
fi

echo "[INFO] alpha_phaseC winner_tag_20k=${WINNER_TAG_20K}"
echo "[INFO] alpha_phaseC winner_tag_100k=${WINNER_TAG_100K}"
if [[ -n "$PHASEB_SUMMARY_CSV" ]]; then
  echo "[INFO] alpha_phaseC phaseB_summary_csv=${PHASEB_SUMMARY_CSV}"
fi
echo "[INFO] alpha_phaseC phasec_config=${PHASEC_CONFIG}"

env \
  PYTHON_BIN="$PYTHON_BIN" \
  PHASEC_TAG="$WINNER_TAG_100K" \
  PHASEC_CONFIG="$PHASEC_CONFIG" \
  STEP6_CONFIG="$STEP6_CONFIG" \
  MAX_STEPS="$MAX_STEPS" \
  RUN_FULL_AUDIT="$RUN_FULL_AUDIT" \
  FULL_AUDIT_MAX_STEPS="$FULL_AUDIT_MAX_STEPS" \
  bash scripts/run_u27_eta082_phaseC.sh
