#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
PHASEB_SUMMARY_CSV="${PHASEB_SUMMARY_CSV:-}"
WINNER_TAG_20K="${WINNER_TAG_20K:-}"
WINNER_TAG_100K="${WINNER_TAG_100K:-}"
BASELINE_TAG_20K="${BASELINE_TAG_20K:-u27_eta082_alpha_ctrl_20k_r1}"
CURRENT_CONFIG_IN="${CURRENT_CONFIG_IN:-configs/prl_100k_signals_u27_eta082_current.yaml}"
CURRENT_CONFIG="${CURRENT_CONFIG:-configs/prl_100k_signals_u27_eta082_current.yaml}"
FORWARD_CONFIG="${FORWARD_CONFIG:-configs/step6_fixedeta_forward_2026ytd_eta082_seed10.yaml}"
OPERATIONAL_CONFIG="${OPERATIONAL_CONFIG:-configs/prl_100k_signals_u27_eta082_operational_2026q1.yaml}"
STEP6_TEMPLATE="${STEP6_TEMPLATE:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
MATERIALIZE_META="${MATERIALIZE_META:-outputs/reports/u27_eta082_adoption_materialization.json}"
ADOPTION_META="${ADOPTION_META:-}"
FORWARD_START="${FORWARD_START:-2026-01-01}"
FORWARD_OUT="${FORWARD_OUT:-outputs/step6_u27_eta082_forward_2026ytd}"
FORWARD_RELEASE_ROOT="${FORWARD_RELEASE_ROOT:-outputs/releases/u27_eta082_forward_2026ytd}"
REFRESH_CACHE="${REFRESH_CACHE:-1}"
AUTO_ADOPT_CURRENT="${AUTO_ADOPT_CURRENT:-1}"
MODEL_ROOT="${MODEL_ROOT:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

if [[ "$AUTO_ADOPT_CURRENT" == "1" ]]; then
  ADOPT_CMD=(
    "$PYTHON_BIN"
    scripts/materialize_u27_alpha_first_batch_current_adoption.py
    --baseline-tag-20k "$BASELINE_TAG_20K"
    --current-config-in "$CURRENT_CONFIG_IN"
    --current-config-out "$CURRENT_CONFIG"
    --forward-config-out "$FORWARD_CONFIG"
    --operational-config-out "$OPERATIONAL_CONFIG"
    --materialize-meta-out "$MATERIALIZE_META"
    --step6-template "$STEP6_TEMPLATE"
    --forward-start "$FORWARD_START"
    --forward-output-root "$FORWARD_OUT"
    --print-shell
  )
  if [[ -n "$PHASEB_SUMMARY_CSV" ]]; then
    ADOPT_CMD+=(--phaseb-summary-csv "$PHASEB_SUMMARY_CSV")
  fi
  if [[ -n "$WINNER_TAG_20K" ]]; then
    ADOPT_CMD+=(--winner-tag-20k "$WINNER_TAG_20K")
  fi
  if [[ -n "$WINNER_TAG_100K" ]]; then
    ADOPT_CMD+=(--winner-tag-100k "$WINNER_TAG_100K")
  fi
  if [[ -n "$ADOPTION_META" ]]; then
    ADOPT_CMD+=(--adoption-meta-out "$ADOPTION_META")
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      PHASEB_SUMMARY_CSV) PHASEB_SUMMARY_CSV="$value" ;;
      WINNER_TAG_20K) WINNER_TAG_20K="$value" ;;
      WINNER_TAG_100K) WINNER_TAG_100K="$value" ;;
      MODEL_ROOT) MODEL_ROOT="$value" ;;
      CURRENT_CONFIG) CURRENT_CONFIG="$value" ;;
      FORWARD_CONFIG) FORWARD_CONFIG="$value" ;;
      OPERATIONAL_CONFIG) OPERATIONAL_CONFIG="$value" ;;
      MATERIALIZE_META) MATERIALIZE_META="$value" ;;
      ADOPTION_META) ADOPTION_META="$value" ;;
    esac
  done < <("${ADOPT_CMD[@]}")
fi

if [[ -z "$WINNER_TAG_100K" && -n "$WINNER_TAG_20K" ]]; then
  WINNER_TAG_100K="${WINNER_TAG_20K/_20k_r1/_100k_r1}"
fi
if [[ -z "$MODEL_ROOT" && -n "$WINNER_TAG_100K" ]]; then
  MODEL_ROOT="outputs/modelswap_${WINNER_TAG_100K}"
fi

if [[ -z "$MODEL_ROOT" || ! -d "$MODEL_ROOT" ]]; then
  echo "[ERROR] Missing winner model root for forward OOS: ${MODEL_ROOT:-<unset>}"
  exit 1
fi

echo "[INFO] alpha_forward winner_tag_20k=${WINNER_TAG_20K:-unset}"
echo "[INFO] alpha_forward winner_tag_100k=${WINNER_TAG_100K:-unset}"
echo "[INFO] alpha_forward current_config=${CURRENT_CONFIG}"
echo "[INFO] alpha_forward model_root=${MODEL_ROOT}"
echo "[INFO] alpha_forward forward_config=${FORWARD_CONFIG}"
echo "[INFO] alpha_forward materialize_meta=${MATERIALIZE_META}"

env \
  PYTHON_BIN="$PYTHON_BIN" \
  CURRENT_CONFIG="$CURRENT_CONFIG" \
  MODEL_ROOT="$MODEL_ROOT" \
  STEP6_TEMPLATE="$STEP6_TEMPLATE" \
  FORWARD_CONFIG="$FORWARD_CONFIG" \
  OPERATIONAL_CONFIG="$OPERATIONAL_CONFIG" \
  MATERIALIZE_META="$MATERIALIZE_META" \
  FORWARD_START="$FORWARD_START" \
  FORWARD_OUT="$FORWARD_OUT" \
  FORWARD_RELEASE_ROOT="$FORWARD_RELEASE_ROOT" \
  REFRESH_CACHE="$REFRESH_CACHE" \
  bash scripts/run_u27_eta082_forward_oos.sh
