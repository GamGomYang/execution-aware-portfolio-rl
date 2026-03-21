#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
WINNER_TAG_20K="${WINNER_TAG_20K:-}"
PHASEB_SUMMARY_CSV="${PHASEB_SUMMARY_CSV:-}"
AUTO_SELECT_WINNER="${AUTO_SELECT_WINNER:-1}"
BASELINE_TAG_20K="${BASELINE_TAG_20K:-u27_eta082_alpha_ctrl_20k_r1}"
TAG_SUFFIX="${TAG_SUFFIX:-100k_r1}"
TIMESTEPS="${TIMESTEPS:-100000}"
META_OUT="${META_OUT:-outputs/reports/u27_alpha_first_batch_promoted_100k_materialization.json}"
SKIP_RATIONALE="${SKIP_RATIONALE:-1}"
SKIP_MANIFESTS="${SKIP_MANIFESTS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

if [[ -z "$WINNER_TAG_20K" && "$AUTO_SELECT_WINNER" == "1" ]]; then
  SELECTOR_CMD=(
    "$PYTHON_BIN" scripts/select_u27_alpha_first_batch_phaseB.py
    --baseline-tag "$BASELINE_TAG_20K"
    --print-shell
  )
  if [[ -n "$PHASEB_SUMMARY_CSV" ]]; then
    SELECTOR_CMD+=(--summary-csv "$PHASEB_SUMMARY_CSV")
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      PHASEB_SUMMARY_CSV) PHASEB_SUMMARY_CSV="$value" ;;
      WINNER_TAG_20K) WINNER_TAG_20K="$value" ;;
    esac
  done < <("${SELECTOR_CMD[@]}")
fi

if [[ -z "$WINNER_TAG_20K" ]]; then
  echo "[ERROR] No winner tag resolved. Provide WINNER_TAG_20K or enable AUTO_SELECT_WINNER with a valid Phase B summary."
  exit 1
fi

WINNER_TAG_100K="${WINNER_TAG_20K/_20k_r1/_100k_r1}"
BASELINE_TAG_100K="${BASELINE_TAG_20K/_20k_r1/_100k_r1}"

CMD=(
  "$PYTHON_BIN" scripts/materialize_u27_alpha_first_batch_configs.py
  --candidates "$BASELINE_TAG_20K" "$WINNER_TAG_20K"
  --timesteps "$TIMESTEPS"
  --tag-suffix "$TAG_SUFFIX"
  --meta-out "$META_OUT"
)
if [[ "$SKIP_RATIONALE" == "1" ]]; then
  CMD+=(--skip-rationale)
fi
if [[ "$SKIP_MANIFESTS" == "1" ]]; then
  CMD+=(--skip-manifests)
fi

echo "[INFO] promote_alpha_winner winner_tag_20k=${WINNER_TAG_20K}"
echo "[INFO] promote_alpha_winner winner_tag_100k=${WINNER_TAG_100K}"
echo "[INFO] promote_alpha_winner baseline_tag_100k=${BASELINE_TAG_100K}"
if [[ -n "$PHASEB_SUMMARY_CSV" ]]; then
  echo "[INFO] promote_alpha_winner phaseB_summary_csv=${PHASEB_SUMMARY_CSV}"
fi

"${CMD[@]}"

echo "WINNER_TAG_20K=${WINNER_TAG_20K}"
echo "WINNER_TAG_100K=${WINNER_TAG_100K}"
echo "BASELINE_TAG_20K=${BASELINE_TAG_20K}"
echo "BASELINE_TAG_100K=${BASELINE_TAG_100K}"
echo "PROMOTION_META=${META_OUT}"
