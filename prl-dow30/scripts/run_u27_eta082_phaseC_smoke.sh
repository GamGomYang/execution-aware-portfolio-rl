#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

SMOKE_TAG="${SMOKE_TAG:-u27_eta082_m070_cg06_100k_r1}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/exp/${SMOKE_TAG}.yaml}"
SMOKE_SEED="${SMOKE_SEED:-0}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
MAX_STEPS="${MAX_STEPS:-252}"
RUN_MATRIX="${RUN_MATRIX:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_eta082_phaseC_smoke_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
TRAIN_ROOT="outputs/smoke_${SMOKE_TAG}_seed${SMOKE_SEED}_r1"
STEP6_ROOT="outputs/step6_smoke_${SMOKE_TAG}_seed${SMOKE_SEED}_r1"
SUMMARY_MD="outputs/reports/u27_eta082_phaseC_smoke_summary_${JOB_TS}.md"
mkdir -p "$LOG_DIR" "outputs/reports"

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

if [[ ! -f "$SMOKE_CONFIG" ]]; then
  log "[ERROR] missing config: ${SMOKE_CONFIG}"
  exit 1
fi

log "[INFO] phase=C_smoke"
log "[INFO] smoke_tag=${SMOKE_TAG}"
log "[INFO] smoke_config=${SMOKE_CONFIG}"
log "[INFO] smoke_seed=${SMOKE_SEED}"
log "[INFO] max_steps=${MAX_STEPS}"
log "[INFO] run_matrix=${RUN_MATRIX}"
log "[INFO] train_root=${TRAIN_ROOT}"

run_step "train_${SMOKE_TAG}_seed${SMOKE_SEED}" \
  "$PYTHON_BIN" scripts/run_train.py \
    --config "$SMOKE_CONFIG" \
    --model-type prl \
    --seed "$SMOKE_SEED" \
    --offline \
    --output-root "$TRAIN_ROOT"

run_step "sanity_${SMOKE_TAG}_seed${SMOKE_SEED}" \
  "$PYTHON_BIN" scripts/step6_sanity.py \
    --config "$STEP6_CONFIG" \
    --model-type prl \
    --seed "$SMOKE_SEED" \
    --model-root "$TRAIN_ROOT" \
    --offline \
    --max-steps "$MAX_STEPS"

if [[ "$RUN_MATRIX" == "1" ]]; then
  run_step "step6_${SMOKE_TAG}_seed${SMOKE_SEED}" \
    "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$STEP6_CONFIG" \
      --model-type prl \
      --model-root "$TRAIN_ROOT" \
      --seed-model-mode independent \
      --seeds "$SMOKE_SEED" \
      --kappas 0.0 0.0005 0.001 \
      --etas 0.082 \
      --out "$STEP6_ROOT" \
      --offline \
      --max-steps "$MAX_STEPS"
fi

SMOKE_TAG="$SMOKE_TAG" \
SMOKE_SEED="$SMOKE_SEED" \
TRAIN_ROOT="$TRAIN_ROOT" \
STEP6_ROOT="$STEP6_ROOT" \
RUN_MATRIX="$RUN_MATRIX" \
SUMMARY_MD="$SUMMARY_MD" \
"$PYTHON_BIN" - <<'PYCODE' | tee -a "$MASTER_LOG"
from __future__ import annotations

import csv
import os
from pathlib import Path

smoke_tag = os.environ["SMOKE_TAG"]
smoke_seed = os.environ["SMOKE_SEED"]
train_root = Path(os.environ["TRAIN_ROOT"])
step6_root = Path(os.environ["STEP6_ROOT"])
run_matrix = os.environ["RUN_MATRIX"] == "1"
out_md = Path(os.environ["SUMMARY_MD"])

lines = []
lines.append("# U27 ETA082 Phase C Smoke Summary")
lines.append("")
lines.append(f"- smoke_tag: {smoke_tag}")
lines.append(f"- smoke_seed: {smoke_seed}")
lines.append(f"- train_root: {train_root}")
lines.append(f"- run_matrix: {run_matrix}")

if run_matrix and (step6_root / "aggregate.csv").exists() and (step6_root / "paired_delta.csv").exists():
    agg = list(csv.DictReader((step6_root / "aggregate.csv").open()))
    paired = list(csv.DictReader((step6_root / "paired_delta.csv").open()))
    main_k001 = next((r for r in agg if r.get("arm") == "main" and abs(float(r["kappa"]) - 0.001) < 1e-12), None)
    base_k001 = next((r for r in agg if r.get("arm") == "baseline" and abs(float(r["kappa"]) - 0.001) < 1e-12), None)
    deltas_k001 = [float(r["delta_sharpe"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12]
    lines.append(f"- step6_root: {step6_root}")
    lines.append(f"- k001_positive: {sum(1 for x in deltas_k001 if x > 0.0)}")
    lines.append(f"- k001_mean_delta_sharpe: {sum(deltas_k001)/len(deltas_k001) if deltas_k001 else None}")
    lines.append(f"- k001_main_median_sharpe: {main_k001['median_sharpe'] if main_k001 else None}")
    lines.append(f"- k001_baseline_median_sharpe: {base_k001['median_sharpe'] if base_k001 else None}")

out_md.write_text("\n".join(lines) + "\n")
print(f"SUMMARY_MD={out_md}")
PYCODE

log "[DONE] phase=C_smoke complete"
log "[DONE] master_log=${MASTER_LOG}"
