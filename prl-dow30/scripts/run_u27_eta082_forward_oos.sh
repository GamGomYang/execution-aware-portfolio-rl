#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
CURRENT_CONFIG="${CURRENT_CONFIG:-configs/prl_100k_signals_u27_eta082_current.yaml}"
STEP6_TEMPLATE="${STEP6_TEMPLATE:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
MODEL_ROOT="${MODEL_ROOT:-outputs/modelswap_u27_eta082_m070_cg03_lr1e4_100k_r1}"
FORWARD_CONFIG="${FORWARD_CONFIG:-configs/step6_fixedeta_forward_2026ytd_eta082_seed10.yaml}"
OPERATIONAL_CONFIG="${OPERATIONAL_CONFIG:-configs/prl_100k_signals_u27_eta082_operational_2026q1.yaml}"
MATERIALIZE_META="${MATERIALIZE_META:-outputs/reports/u27_eta082_adoption_materialization.json}"
FORWARD_START="${FORWARD_START:-2026-01-01}"
FORWARD_OUT="${FORWARD_OUT:-outputs/step6_u27_eta082_forward_2026ytd}"
FORWARD_RELEASE_ROOT="${FORWARD_RELEASE_ROOT:-outputs/releases/u27_eta082_forward_2026ytd}"
REFRESH_CACHE="${REFRESH_CACHE:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
CHECK2_HARD="${CHECK2_HARD:-6}"
CHECK2_SOFT="${CHECK2_SOFT:-5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_eta082_forward_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
SUMMARY_MD="outputs/reports/u27_eta082_forward_summary_${JOB_TS}.md"
mkdir -p "$LOG_DIR" "outputs/reports" "$FORWARD_RELEASE_ROOT"

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

if [[ "$REFRESH_CACHE" == "1" ]]; then
  run_step "build_cache_current_alias" \
    "$PYTHON_BIN" scripts/build_cache.py \
      --config "$CURRENT_CONFIG"
else
  log "[INFO] skipping cache refresh because REFRESH_CACHE=${REFRESH_CACHE}"
fi

run_step "materialize_forward_and_operational_configs" \
  "$PYTHON_BIN" scripts/materialize_u27_eta082_adoption_configs.py \
    --current-config "$CURRENT_CONFIG" \
    --step6-template "$STEP6_TEMPLATE" \
    --forward-config-out "$FORWARD_CONFIG" \
    --operational-config-out "$OPERATIONAL_CONFIG" \
    --meta-out "$MATERIALIZE_META" \
    --forward-start "$FORWARD_START" \
    --forward-output-root "$FORWARD_OUT"

CACHE_MAX_DATE="$(MATERIALIZE_META="$MATERIALIZE_META" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
meta = json.loads(Path(os.environ["MATERIALIZE_META"]).read_text())
print(meta["cache_max_date"])
PY
)"
log "[INFO] cache_max_date=${CACHE_MAX_DATE}"
log "[INFO] forward_config=${FORWARD_CONFIG}"
log "[INFO] operational_config=${OPERATIONAL_CONFIG}"

run_step "forward_sanity_seed0" \
  "$PYTHON_BIN" scripts/step6_sanity.py \
    --config "$FORWARD_CONFIG" \
    --model-type prl \
    --seed 0 \
    --model-root "$MODEL_ROOT" \
    --offline \
    --max-steps "$MAX_STEPS"

run_step "forward_step6_matrix" \
  "$PYTHON_BIN" scripts/step6_run_matrix.py \
    --config "$FORWARD_CONFIG" \
    --model-type prl \
    --model-root "$MODEL_ROOT" \
    --seed-model-mode independent \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --kappas 0.0 0.0005 0.001 \
    --etas 0.082 \
    --out "$FORWARD_OUT" \
    --offline \
    --max-steps "$MAX_STEPS"

run_step "forward_acceptance_hard6" \
  "$PYTHON_BIN" scripts/step6_check_acceptance.py \
    --paired "${FORWARD_OUT}/paired_delta.csv" \
    --aggregate "${FORWARD_OUT}/aggregate.csv" \
    --check2-min-positive-seeds "$CHECK2_HARD" \
    --out-md "${FORWARD_OUT}/acceptance_report_hard6.md" \
    --out-json "${FORWARD_OUT}/acceptance_report_hard6.json" \
    --no-fail-exit

run_step "forward_acceptance_soft5" \
  "$PYTHON_BIN" scripts/step6_check_acceptance.py \
    --paired "${FORWARD_OUT}/paired_delta.csv" \
    --aggregate "${FORWARD_OUT}/aggregate.csv" \
    --check2-min-positive-seeds "$CHECK2_SOFT" \
    --out-md "${FORWARD_OUT}/acceptance_report_soft5.md" \
    --out-json "${FORWARD_OUT}/acceptance_report_soft5.json" \
    --no-fail-exit

FORWARD_OUT="$FORWARD_OUT" CACHE_MAX_DATE="$CACHE_MAX_DATE" SUMMARY_MD="$SUMMARY_MD" "$PYTHON_BIN" - <<'PYCODE' | tee -a "$MASTER_LOG"
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

forward_out = Path(os.environ["FORWARD_OUT"])
cache_max_date = os.environ["CACHE_MAX_DATE"]
summary_md = Path(os.environ["SUMMARY_MD"])

hard = json.loads((forward_out / "acceptance_report_hard6.json").read_text())
soft = json.loads((forward_out / "acceptance_report_soft5.json").read_text())
agg = list(csv.DictReader((forward_out / "aggregate.csv").open()))
paired = list(csv.DictReader((forward_out / "paired_delta.csv").open()))
main_k001 = next(r for r in agg if r.get("arm") == "main" and abs(float(r["kappa"]) - 0.001) < 1e-12)
base_k001 = next(r for r in agg if r.get("arm") == "baseline" and abs(float(r["kappa"]) - 0.001) < 1e-12)
deltas_k001 = [float(r["delta_sharpe"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12]
neg_seeds = [int(r["seed"]) for r in paired if abs(float(r["kappa"]) - 0.001) < 1e-12 and float(r["delta_sharpe"]) < 0.0]

lines = []
lines.append("# U27 ETA082 Forward 2026 YTD Summary")
lines.append("")
lines.append("- evaluation_window: 2026-01-01~" + cache_max_date)
lines.append(f"- forward_root: {forward_out}")
lines.append(f"- hard6_pass: {hard['overall_pass']}")
lines.append(f"- soft5_pass: {soft['overall_pass']}")
lines.append(f"- k001_main_median_sharpe: {main_k001['median_sharpe']}")
lines.append(f"- k001_baseline_median_sharpe: {base_k001['median_sharpe']}")
lines.append(f"- k001_mean_delta_sharpe: {sum(deltas_k001)/len(deltas_k001) if deltas_k001 else None}")
lines.append(f"- k001_positive: {sum(1 for x in deltas_k001 if x > 0.0)}")
lines.append(f"- k001_negative_seeds: {neg_seeds}")
lines.append(f"- k001_collapse_rate: {main_k001['collapse_rate']}")
summary_md.write_text("\n".join(lines) + "\n")
print(f"SUMMARY_MD={summary_md}")
PYCODE

cp "$FORWARD_CONFIG" "$FORWARD_RELEASE_ROOT/"
cp "$MATERIALIZE_META" "$FORWARD_RELEASE_ROOT/"
cp "$SUMMARY_MD" "$FORWARD_RELEASE_ROOT/"
cp "${FORWARD_OUT}/aggregate.csv" "$FORWARD_RELEASE_ROOT/"
cp "${FORWARD_OUT}/paired_delta.csv" "$FORWARD_RELEASE_ROOT/"
cp "${FORWARD_OUT}/acceptance_report_hard6.md" "$FORWARD_RELEASE_ROOT/"
cp "${FORWARD_OUT}/acceptance_report_hard6.json" "$FORWARD_RELEASE_ROOT/"
cp "${FORWARD_OUT}/acceptance_report_soft5.md" "$FORWARD_RELEASE_ROOT/"
cp "${FORWARD_OUT}/acceptance_report_soft5.json" "$FORWARD_RELEASE_ROOT/"

FORWARD_RELEASE_ROOT="$FORWARD_RELEASE_ROOT" FORWARD_OUT="$FORWARD_OUT" CACHE_MAX_DATE="$CACHE_MAX_DATE" MODEL_ROOT="$MODEL_ROOT" FORWARD_CONFIG="$FORWARD_CONFIG" SUMMARY_MD="$SUMMARY_MD" "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import os
from pathlib import Path

release_root = Path(os.environ["FORWARD_RELEASE_ROOT"])
forward_out = Path(os.environ["FORWARD_OUT"])
cache_max_date = os.environ["CACHE_MAX_DATE"]
model_root = os.environ["MODEL_ROOT"]
forward_config = os.environ["FORWARD_CONFIG"]
summary_md = os.environ["SUMMARY_MD"]

lines = []
lines.append("# U27 ETA082 Forward 2026 YTD Release")
lines.append("")
lines.append(f"- forward_config: `{forward_config}`")
lines.append(f"- authoritative_model_root: `{model_root}`")
lines.append(f"- evaluation_window: `2026-01-01~{cache_max_date}`")
lines.append(f"- forward_step6_root: `{forward_out}`")
lines.append(f"- summary: `{summary_md}`")
lines.append("")
lines.append("## Notes")
lines.append("- This is a locked forward OOS run using the already-adopted winner models.")
lines.append("- No hyperparameter changes were made in this track.")
release_root.joinpath("README.md").write_text("\n".join(lines) + "\n")
PYCODE

log "[DONE] forward_oos complete"
log "[DONE] master_log=${MASTER_LOG}"
