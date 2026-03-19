#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

PRIMARY_BASELINE_TAG="${PRIMARY_BASELINE_TAG:-u27_eta082_ctrl20k_r1}"
PRIMARY_CANDIDATES_RAW="${PRIMARY_CANDIDATES:-u27_eta082_m085_20k_r1 u27_eta082_m070_cg06_20k_r1}"

FALLBACK_ENABLED="${FALLBACK_ENABLED:-1}"
FALLBACK_TAG="${FALLBACK_TAG:-u27_eta082_m070_cg03_lr1e4_20k_r1}"

PHASEB_MAX_STEPS="${PHASEB_MAX_STEPS:-252}"
PHASEC_MAX_STEPS="${PHASEC_MAX_STEPS:-252}"
RUN_FULL_AUDIT="${RUN_FULL_AUDIT:-1}"
FULL_AUDIT_MAX_STEPS="${FULL_AUDIT_MAX_STEPS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

read -r -a PRIMARY_CANDIDATES <<< "$PRIMARY_CANDIDATES_RAW"

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_eta082_overnight_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
SUMMARY_MD="outputs/reports/u27_eta082_overnight_summary_${JOB_TS}.md"
mkdir -p "$LOG_DIR" "outputs/reports"

PRIMARY_PHASEB_SUMMARY=""
FALLBACK_PHASEB_SUMMARY=""
PROMOTED_20K_TAG=""
PROMOTED_100K_TAG=""
PROMOTION_REASON=""

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

latest_phaseb_summary() {
  ls -1t outputs/reports/u27_eta082_phaseB_summary_*.csv 2>/dev/null | head -1
}

map_20k_to_100k() {
  local tag="$1"
  local mapped="${tag/_20k_r1/_100k_r1}"
  if [[ "$mapped" == "$tag" ]]; then
    return 1
  fi
  printf '%s\n' "$mapped"
}

pick_best_candidate() {
  local summary_csv="$1"
  local exclude_tag="${2:-}"
  SUMMARY_CSV="$summary_csv" EXCLUDE_TAG="$exclude_tag" "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import csv
import os
from pathlib import Path

summary_csv = Path(os.environ["SUMMARY_CSV"])
exclude_tag = os.environ.get("EXCLUDE_TAG", "").strip()
rows = list(csv.DictReader(summary_csv.open()))
if exclude_tag:
    rows = [r for r in rows if r["tag"] != exclude_tag]

def key(row: dict[str, str]) -> tuple:
    hard = row.get("hard_pass", "") == "True"
    soft = row.get("soft_pass", "") == "True"
    mean = float(row["k001_mean_delta_sharpe"]) if row.get("k001_mean_delta_sharpe") not in ("", None) else float("-inf")
    main = float(row["k001_main_median_sharpe"]) if row.get("k001_main_median_sharpe") not in ("", None) else float("-inf")
    return (hard, soft, mean, main)

if not rows:
    print("BEST_TAG=")
    print("BEST_HARD_PASS=0")
    print("BEST_SOFT_PASS=0")
    print("BEST_K001_MEAN=")
    raise SystemExit(0)

best = max(rows, key=key)
print(f"BEST_TAG={best['tag']}")
print(f"BEST_HARD_PASS={1 if best['hard_pass'] == 'True' else 0}")
print(f"BEST_SOFT_PASS={1 if best['soft_pass'] == 'True' else 0}")
print(f"BEST_K001_MEAN={best.get('k001_mean_delta_sharpe', '')}")
PYCODE
}

write_summary() {
  {
    echo "# U27 ETA082 Overnight Summary"
    echo
    echo "- primary_baseline_tag: ${PRIMARY_BASELINE_TAG}"
    echo "- primary_candidates: ${PRIMARY_CANDIDATES[*]}"
    echo "- fallback_enabled: ${FALLBACK_ENABLED}"
    echo "- fallback_tag: ${FALLBACK_TAG}"
    echo "- promoted_20k_tag: ${PROMOTED_20K_TAG:-none}"
    echo "- promoted_100k_tag: ${PROMOTED_100K_TAG:-none}"
    echo "- promotion_reason: ${PROMOTION_REASON:-none}"
    echo "- primary_phaseb_summary: ${PRIMARY_PHASEB_SUMMARY:-none}"
    echo "- fallback_phaseb_summary: ${FALLBACK_PHASEB_SUMMARY:-none}"
    echo "- master_log: ${MASTER_LOG}"
  } > "$SUMMARY_MD"
  log "SUMMARY_MD=${SUMMARY_MD}"
}

run_primary_phaseb() {
  local before latest
  before="$(latest_phaseb_summary || true)"
  run_step "phaseB_primary" env \
    PYTHON_BIN="$PYTHON_BIN" \
    BASELINE_TAG="$PRIMARY_BASELINE_TAG" \
    PHASEB_CANDIDATES="${PRIMARY_CANDIDATES[*]}" \
    MAX_STEPS="$PHASEB_MAX_STEPS" \
    bash scripts/run_u27_eta082_phaseB.sh
  latest="$(latest_phaseb_summary || true)"
  if [[ -z "$latest" || "$latest" == "$before" ]]; then
    log "[ERROR] could not resolve primary Phase B summary"
    exit 1
  fi
  PRIMARY_PHASEB_SUMMARY="$latest"
  log "[INFO] primary_phaseb_summary=${PRIMARY_PHASEB_SUMMARY}"
}

run_fallback_phaseb() {
  local before latest
  before="$(latest_phaseb_summary || true)"
  run_step "phaseB_fallback" env \
    PYTHON_BIN="$PYTHON_BIN" \
    BASELINE_TAG="$FALLBACK_TAG" \
    PHASEB_CANDIDATES="" \
    MAX_STEPS="$PHASEB_MAX_STEPS" \
    bash scripts/run_u27_eta082_phaseB.sh
  latest="$(latest_phaseb_summary || true)"
  if [[ -z "$latest" || "$latest" == "$before" ]]; then
    log "[ERROR] could not resolve fallback Phase B summary"
    exit 1
  fi
  FALLBACK_PHASEB_SUMMARY="$latest"
  log "[INFO] fallback_phaseb_summary=${FALLBACK_PHASEB_SUMMARY}"
}

promote_candidate() {
  local tag20="$1"
  local reason="$2"
  local tag100
  if ! tag100="$(map_20k_to_100k "$tag20")"; then
    log "[ERROR] cannot map 20k tag to 100k tag: ${tag20}"
    exit 1
  fi
  if [[ ! -f "configs/exp/${tag100}.yaml" ]]; then
    log "[ERROR] missing 100k config for promoted tag: configs/exp/${tag100}.yaml"
    exit 1
  fi

  PROMOTED_20K_TAG="$tag20"
  PROMOTED_100K_TAG="$tag100"
  PROMOTION_REASON="$reason"

  run_step "phaseC_${tag100}" env \
    PYTHON_BIN="$PYTHON_BIN" \
    PHASEC_TAG="$tag100" \
    MAX_STEPS="$PHASEC_MAX_STEPS" \
    RUN_FULL_AUDIT="$RUN_FULL_AUDIT" \
    FULL_AUDIT_MAX_STEPS="$FULL_AUDIT_MAX_STEPS" \
    bash scripts/run_u27_eta082_phaseC.sh
}

log "[INFO] overnight_job_ts=${JOB_TS}"
log "[INFO] primary_baseline_tag=${PRIMARY_BASELINE_TAG}"
log "[INFO] primary_candidates=${PRIMARY_CANDIDATES[*]}"
log "[INFO] fallback_enabled=${FALLBACK_ENABLED}"
log "[INFO] fallback_tag=${FALLBACK_TAG}"
log "[INFO] phaseb_max_steps=${PHASEB_MAX_STEPS}"
log "[INFO] phasec_max_steps=${PHASEC_MAX_STEPS}"
log "[INFO] run_full_audit=${RUN_FULL_AUDIT}"

run_primary_phaseb

declare BEST_TAG="" BEST_HARD_PASS="0" BEST_SOFT_PASS="0" BEST_K001_MEAN=""
while IFS='=' read -r key value; do
  case "$key" in
    BEST_TAG) BEST_TAG="$value" ;;
    BEST_HARD_PASS) BEST_HARD_PASS="$value" ;;
    BEST_SOFT_PASS) BEST_SOFT_PASS="$value" ;;
    BEST_K001_MEAN) BEST_K001_MEAN="$value" ;;
  esac
done < <(pick_best_candidate "$PRIMARY_PHASEB_SUMMARY" "${PRIMARY_BASELINE_TAG}_full10")

if [[ -n "$BEST_TAG" ]]; then
  BEST_TAG="${BEST_TAG%_full10}"
  log "[INFO] primary_best_candidate=${BEST_TAG} hard=${BEST_HARD_PASS} soft=${BEST_SOFT_PASS} k001_mean=${BEST_K001_MEAN}"
fi

if [[ "$BEST_HARD_PASS" == "1" || "$BEST_SOFT_PASS" == "1" ]]; then
  promote_candidate "$BEST_TAG" "primary_phaseB_pass"
  write_summary
  log "[DONE] overnight complete via primary promotion"
  exit 0
fi

if [[ "$FALLBACK_ENABLED" != "1" ]]; then
  log "[INFO] no promotable primary candidate and fallback disabled"
  write_summary
  log "[DONE] overnight complete without promotion"
  exit 0
fi

run_fallback_phaseb

BEST_TAG=""
BEST_HARD_PASS="0"
BEST_SOFT_PASS="0"
BEST_K001_MEAN=""
while IFS='=' read -r key value; do
  case "$key" in
    BEST_TAG) BEST_TAG="$value" ;;
    BEST_HARD_PASS) BEST_HARD_PASS="$value" ;;
    BEST_SOFT_PASS) BEST_SOFT_PASS="$value" ;;
    BEST_K001_MEAN) BEST_K001_MEAN="$value" ;;
  esac
done < <(pick_best_candidate "$FALLBACK_PHASEB_SUMMARY")

if [[ -n "$BEST_TAG" ]]; then
  BEST_TAG="${BEST_TAG%_full10}"
  log "[INFO] fallback_best_candidate=${BEST_TAG} hard=${BEST_HARD_PASS} soft=${BEST_SOFT_PASS} k001_mean=${BEST_K001_MEAN}"
fi

if [[ "$BEST_HARD_PASS" == "1" || "$BEST_SOFT_PASS" == "1" ]]; then
  promote_candidate "$BEST_TAG" "fallback_phaseB_pass"
  write_summary
  log "[DONE] overnight complete via fallback promotion"
  exit 0
fi

log "[INFO] no promotable fallback candidate"
write_summary
log "[DONE] overnight complete without Phase C promotion"
