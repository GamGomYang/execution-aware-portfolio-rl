#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
PHASE="${1:-phaseA}"

case "$PHASE" in
  phaseA) SCRIPT="scripts/run_u27_eta082_phaseA.sh" ;;
  alphaA) SCRIPT="scripts/run_u27_alpha_first_batch_phaseA.sh" ;;
  phaseB) SCRIPT="scripts/run_u27_eta082_phaseB.sh" ;;
  phaseC) SCRIPT="scripts/run_u27_eta082_phaseC.sh" ;;
  smoke) SCRIPT="scripts/run_u27_eta082_phaseC_smoke.sh" ;;
  overnight) SCRIPT="scripts/run_u27_eta082_overnight.sh" ;;
  forward) SCRIPT="scripts/run_u27_eta082_forward_oos.sh" ;;
  postadopt) SCRIPT="scripts/run_u27_eta082_post_adoption.sh" ;;
  *)
    echo "[ERROR] unknown phase: $PHASE"
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found"
  exit 1
fi

SESSION="${SESSION:-u27_eta082_${PHASE}_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$ROOT"

ENV_VARS=(
  PYTHON_BIN
  CONFIG_PATH
  JOB_TAG
  JOB_TAG_BASE
  SMOKE_TAG
  SMOKE_CONFIG
  SMOKE_SEED
  RUN_MATRIX
  PHASEC_TAG
  PHASEC_CONFIG
  STEP6_CONFIG
  CANDIDATES
  PILOT_SEEDS
  CHECK2_HARD
  CHECK2_SOFT
  MATERIALIZE
  MAX_STEPS
  RUN_FULL_AUDIT
  FULL_AUDIT_MAX_STEPS
  EVAL_START
  EVAL_END
  BASELINE_TAG
  PHASEB_CANDIDATES
  FALLBACK_ENABLED
  FALLBACK_TAG
  PRIMARY_BASELINE_TAG
  PRIMARY_CANDIDATES
  CURRENT_CONFIG
  STEP6_TEMPLATE
  MODEL_ROOT
  FORWARD_CONFIG
  OPERATIONAL_CONFIG
  MATERIALIZE_META
  FORWARD_START
  FORWARD_OUT
  FORWARD_RELEASE_ROOT
  OPERATIONAL_RELEASE_ROOT
  REFRESH_CACHE
)

CMD="cd $ROOT &&"
for var_name in "${ENV_VARS[@]}"; do
  if [[ -v $var_name ]]; then
    value="${!var_name}"
    CMD+=" $(printf '%q=%q' "$var_name" "$value")"
  fi
done
CMD+=" bash $SCRIPT"

tmux new-session -d -s "$SESSION" "$CMD"
echo "SESSION=$SESSION"
echo "SCRIPT=$SCRIPT"
