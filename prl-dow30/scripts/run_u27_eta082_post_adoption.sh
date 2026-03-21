#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
CURRENT_CONFIG="${CURRENT_CONFIG:-configs/prl_100k_signals_u27_eta082_current.yaml}"
STEP6_TEMPLATE="${STEP6_TEMPLATE:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
FORWARD_CONFIG="${FORWARD_CONFIG:-configs/step6_fixedeta_forward_2026ytd_eta082_seed10.yaml}"
OPERATIONAL_CONFIG="${OPERATIONAL_CONFIG:-configs/prl_100k_signals_u27_eta082_operational_2026q1.yaml}"
MATERIALIZE_META="${MATERIALIZE_META:-outputs/reports/u27_eta082_adoption_materialization.json}"
FORWARD_RELEASE_ROOT="${FORWARD_RELEASE_ROOT:-outputs/releases/u27_eta082_forward_2026ytd}"
OPERATIONAL_RELEASE_ROOT="${OPERATIONAL_RELEASE_ROOT:-outputs/releases/u27_eta082_operational_2026q1}"
FORWARD_OUT="${FORWARD_OUT:-outputs/step6_u27_eta082_forward_2026ytd}"
FORWARD_START="${FORWARD_START:-2026-01-01}"
JOB_TAG_BASE="${JOB_TAG_BASE:-u27_eta082_operational_2026q1}"
REFRESH_CACHE="${REFRESH_CACHE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

PIPELINE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_eta082_post_adoption_${PIPELINE_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
PIPELINE_SUMMARY_MD="outputs/reports/u27_eta082_post_adoption_summary_${PIPELINE_TS}.md"
mkdir -p "$LOG_DIR" "outputs/reports" "$OPERATIONAL_RELEASE_ROOT"

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

run_step "forward_oos_pipeline" \
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    CURRENT_CONFIG="$CURRENT_CONFIG" \
    STEP6_TEMPLATE="$STEP6_TEMPLATE" \
    FORWARD_CONFIG="$FORWARD_CONFIG" \
    OPERATIONAL_CONFIG="$OPERATIONAL_CONFIG" \
    MATERIALIZE_META="$MATERIALIZE_META" \
    FORWARD_START="$FORWARD_START" \
    FORWARD_OUT="$FORWARD_OUT" \
    FORWARD_RELEASE_ROOT="$FORWARD_RELEASE_ROOT" \
    REFRESH_CACHE="$REFRESH_CACHE" \
    MAX_STEPS=0 \
    bash scripts/run_u27_eta082_forward_oos.sh

CACHE_MAX_DATE="$(MATERIALIZE_META="$MATERIALIZE_META" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
meta = json.loads(Path(os.environ["MATERIALIZE_META"]).read_text())
print(meta["cache_max_date"])
PY
)"

OPERATIONAL_TAG="${JOB_TAG_BASE}_${PIPELINE_TS}"
log "[INFO] cache_max_date=${CACHE_MAX_DATE}"
log "[INFO] operational_tag=${OPERATIONAL_TAG}"
log "[INFO] operational_config=${OPERATIONAL_CONFIG}"

run_step "operational_phasec" \
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    CONFIG_PATH="$OPERATIONAL_CONFIG" \
    JOB_TAG="$OPERATIONAL_TAG" \
    JOB_TAG_BASE="$JOB_TAG_BASE" \
    STEP6_CONFIG="$FORWARD_CONFIG" \
    MAX_STEPS=0 \
    RUN_FULL_AUDIT=0 \
    EVAL_START="$FORWARD_START" \
    EVAL_END="$CACHE_MAX_DATE" \
    bash scripts/run_u27_eta082_current_phaseC.sh

OP_STEP6_ROOT="outputs/step6_${OPERATIONAL_TAG}"
OP_MODELSWAP_ROOT="outputs/modelswap_${OPERATIONAL_TAG}"
OP_HARD_JSON="${OP_STEP6_ROOT}/acceptance_report_hard6.json"
OP_SOFT_JSON="${OP_STEP6_ROOT}/acceptance_report_soft5.json"

if [[ ! -f "$OP_HARD_JSON" ]]; then
  log "[ERROR] missing operational acceptance file: ${OP_HARD_JSON}"
  exit 1
fi

cp "$OPERATIONAL_CONFIG" "$OPERATIONAL_RELEASE_ROOT/"
cp "$FORWARD_CONFIG" "$OPERATIONAL_RELEASE_ROOT/"
cp "$MATERIALIZE_META" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_STEP6_ROOT}/aggregate.csv" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_STEP6_ROOT}/paired_delta.csv" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_STEP6_ROOT}/acceptance_report_hard6.md" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_STEP6_ROOT}/acceptance_report_hard6.json" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_STEP6_ROOT}/acceptance_report_soft5.md" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_STEP6_ROOT}/acceptance_report_soft5.json" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_MODELSWAP_ROOT}/reports/model_swap_runs.csv" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_MODELSWAP_ROOT}/reports/model_swap_rl_aggregate.csv" "$OPERATIONAL_RELEASE_ROOT/"
cp "${OP_MODELSWAP_ROOT}/reports/model_swap_summary.md" "$OPERATIONAL_RELEASE_ROOT/"

OPERATIONAL_RELEASE_ROOT="$OPERATIONAL_RELEASE_ROOT" FORWARD_RELEASE_ROOT="$FORWARD_RELEASE_ROOT" OP_STEP6_ROOT="$OP_STEP6_ROOT" OP_MODELSWAP_ROOT="$OP_MODELSWAP_ROOT" CACHE_MAX_DATE="$CACHE_MAX_DATE" OPERATIONAL_TAG="$OPERATIONAL_TAG" PIPELINE_SUMMARY_MD="$PIPELINE_SUMMARY_MD" "$PYTHON_BIN" - <<'PYCODE' | tee -a "$MASTER_LOG"
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

release_root = Path(os.environ["OPERATIONAL_RELEASE_ROOT"])
forward_release_root = os.environ["FORWARD_RELEASE_ROOT"]
step6_root = Path(os.environ["OP_STEP6_ROOT"])
modelswap_root = Path(os.environ["OP_MODELSWAP_ROOT"])
cache_max_date = os.environ["CACHE_MAX_DATE"]
operational_tag = os.environ["OPERATIONAL_TAG"]
summary_md = Path(os.environ["PIPELINE_SUMMARY_MD"])

hard = json.loads((step6_root / "acceptance_report_hard6.json").read_text())
soft = json.loads((step6_root / "acceptance_report_soft5.json").read_text())
agg = list(csv.DictReader((step6_root / "aggregate.csv").open()))
paired = list(csv.DictReader((step6_root / "paired_delta.csv").open()))
main_k001 = next(r for r in agg if r.get("arm") == "main" and abs(float(r["kappa"]) - 0.001) < 1e-12)
base_k001 = next(r for r in agg if r.get("arm") == "baseline" and abs(float(r["kappa"]) - 0.001) < 1e-12)
deltas_k001 = [float(r["delta_sharpe"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12]
neg_seeds = [int(r["seed"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12 and float(r["delta_sharpe"]) < 0.0]

readme_lines = []
readme_lines.append("# U27 ETA082 Operational 2026Q1 Release")
readme_lines.append("")
readme_lines.append(f"- operational_tag: `{operational_tag}`")
readme_lines.append(f"- evaluation_window: `2026-01-01~{cache_max_date}`")
readme_lines.append(f"- modelswap_root: `{modelswap_root}`")
readme_lines.append(f"- step6_root: `{step6_root}`")
readme_lines.append(f"- hard6_pass: `{hard['overall_pass']}`")
readme_lines.append(f"- soft5_pass: `{soft['overall_pass']}`")
readme_lines.append(f"- k001_mean_delta_sharpe: `{sum(deltas_k001)/len(deltas_k001) if deltas_k001 else None}`")
readme_lines.append(f"- k001_positive: `{sum(1 for x in deltas_k001 if x > 0.0)}`")
readme_lines.append(f"- k001_negative_seeds: `{neg_seeds}`")
readme_lines.append(f"- k001_main_median_sharpe: `{main_k001['median_sharpe']}`")
readme_lines.append(f"- k001_baseline_median_sharpe: `{base_k001['median_sharpe']}`")
readme_lines.append(f"- k001_collapse_rate: `{main_k001['collapse_rate']}`")
release_root.joinpath("README.md").write_text("\n".join(readme_lines) + "\n")

summary_lines = []
summary_lines.append("# U27 ETA082 Post-Adoption Summary")
summary_lines.append("")
summary_lines.append(f"- forward_release_root: {forward_release_root}")
summary_lines.append(f"- operational_release_root: {release_root}")
summary_lines.append(f"- evaluation_window: 2026-01-01~{cache_max_date}")
summary_lines.append(f"- operational_hard6_pass: {hard['overall_pass']}")
summary_lines.append(f"- operational_soft5_pass: {soft['overall_pass']}")
summary_lines.append(f"- operational_k001_mean_delta_sharpe: {sum(deltas_k001)/len(deltas_k001) if deltas_k001 else None}")
summary_lines.append(f"- operational_k001_positive: {sum(1 for x in deltas_k001 if x > 0.0)}")
summary_lines.append(f"- operational_k001_negative_seeds: {neg_seeds}")
summary_md.write_text("\n".join(summary_lines) + "\n")
print(f"PIPELINE_SUMMARY_MD={summary_md}")
PYCODE

log "[DONE] post_adoption pipeline complete"
log "[DONE] master_log=${MASTER_LOG}"
