#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
if [[ -n "${PYTHON_CMD:-}" ]]; then
  read -r -a PYTHON <<< "$PYTHON_CMD"
elif [[ -x "/workspace/execution-aware-portfolio-rl/.venv/bin/python" ]]; then
  PYTHON=("/workspace/execution-aware-portfolio-rl/.venv/bin/python")
else
  PYTHON=("python3")
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON=("$PYTHON_BIN")
fi

CURRENT_CONFIG="${CURRENT_CONFIG:-configs/prl_100k_signals_u27_eta082_current.yaml}"
SEEDS_RAW="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ETAS_RAW="${ETAS:-1.0 0.5 0.2 0.1 0.082 0.05 0.02}"
KAPPAS_RAW="${KAPPAS:-0.0 0.0005 0.001}"
VALIDATION_START="${VALIDATION_START:-2022-01-01}"
VALIDATION_END="${VALIDATION_END:-2023-12-31}"
FINAL_START="${FINAL_START:-2024-01-01}"
FINAL_END="${FINAL_END:-2025-12-31}"
MAX_STEPS="${MAX_STEPS:-0}"
SAC_TOTAL_TIMESTEPS="${SAC_TOTAL_TIMESTEPS:-0}"
BASELINE_ETA="${BASELINE_ETA:-1.0}"
POSITIVE_KAPPAS="${POSITIVE_KAPPAS:-0.0005,0.001}"
RELATIVE_THRESHOLD="${RELATIVE_THRESHOLD:-0.95}"
FINAL_MODE="${FINAL_MODE:-selected_plus_baseline}"
FINAL_ETAS_RAW="${FINAL_ETAS:-}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
RUN_SELECT="${RUN_SELECT:-1}"
RUN_FINAL="${RUN_FINAL:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_PACK="${RUN_PACK:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$ROOT"
export PYTHONPATH="."

read -r -a SEEDS <<< "$SEEDS_RAW"
read -r -a ETAS <<< "$ETAS_RAW"
read -r -a KAPPAS <<< "$KAPPAS_RAW"

JOB_TS="${JOB_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-outputs/paper_rebuild_${JOB_TS}}"
TRAIN_ROOT="${RUN_ROOT}/train_control"
VALIDATION_ROOT="${RUN_ROOT}/validation_eta"
FINAL_ROOT="${RUN_ROOT}/final_eta"
BASELINES_ROOT="${RUN_ROOT}/external_baselines"
PACK_ROOT="${RUN_ROOT}/paper_pack"
PACK_STATS_ROOT="${PACK_ROOT}/stats"
PACK_DIAGNOSTICS_ROOT="${PACK_ROOT}/diagnostics"
PACK_FIGURES_ROOT="${PACK_ROOT}/figures"
CONFIG_ROOT="${RUN_ROOT}/configs"
LOG_DIR="${RUN_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/master.log"

SNAPSHOT_CONFIG="${CONFIG_ROOT}/snapshot_control.yaml"
SIGNAL_SNAPSHOT="${CONFIG_ROOT}/selected_signals_snapshot.json"
VALIDATION_CONFIG="${CONFIG_ROOT}/validation_eta.yaml"
FINAL_CONFIG="${CONFIG_ROOT}/final_eta.yaml"
MATERIALIZATION_META="${CONFIG_ROOT}/materialization_meta.json"
SELECTION_DIR="${VALIDATION_ROOT}/selection"

mkdir -p "$TRAIN_ROOT" "$VALIDATION_ROOT" "$FINAL_ROOT" "$BASELINES_ROOT" "$PACK_ROOT" "$CONFIG_ROOT" "$LOG_DIR"

log() {
  echo "$1" | tee -a "$MASTER_LOG"
}

run_cmd() {
  local name="$1"
  shift
  log "[STEP-START] ${name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY-RUN] $*"
    log "[STEP-END] ${name} rc=0 :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  log "[STEP-END] ${name} rc=${rc} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$rc"
}

selected_eta_from_json() {
  "${PYTHON[@]}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
payload = json.loads(path.read_text())
value = payload.get("selected_eta")
print("" if value is None else value)
PY
}

build_pack() {
  mkdir -p "$PACK_ROOT/configs" "$PACK_ROOT/validation" "$PACK_ROOT/final" "$PACK_ROOT/checklists"
  cp "$SNAPSHOT_CONFIG" "$PACK_ROOT/configs/" 2>/dev/null || true
  cp "$VALIDATION_CONFIG" "$PACK_ROOT/configs/" 2>/dev/null || true
  cp "$FINAL_CONFIG" "$PACK_ROOT/configs/" 2>/dev/null || true
  cp "$MATERIALIZATION_META" "$PACK_ROOT/configs/" 2>/dev/null || true

  cp "$VALIDATION_ROOT/aggregate.csv" "$PACK_ROOT/validation/" 2>/dev/null || true
  cp "$VALIDATION_ROOT/paired_delta.csv" "$PACK_ROOT/validation/" 2>/dev/null || true
  cp "$VALIDATION_ROOT/fig_frontier.png" "$PACK_ROOT/validation/" 2>/dev/null || true
  cp "$VALIDATION_ROOT/fig_misalignment.png" "$PACK_ROOT/validation/" 2>/dev/null || true
  cp "$SELECTION_DIR/validation_eta_selection.json" "$PACK_ROOT/validation/" 2>/dev/null || true
  cp "$SELECTION_DIR/validation_eta_selection.md" "$PACK_ROOT/validation/" 2>/dev/null || true

  cp "$FINAL_ROOT/aggregate.csv" "$PACK_ROOT/final/" 2>/dev/null || true
  cp "$FINAL_ROOT/paired_delta.csv" "$PACK_ROOT/final/" 2>/dev/null || true
  cp "$FINAL_ROOT/fig_frontier.png" "$PACK_ROOT/final/" 2>/dev/null || true
  cp "$FINAL_ROOT/fig_misalignment.png" "$PACK_ROOT/final/" 2>/dev/null || true

  cp "$BASELINES_ROOT/aggregate.csv" "$PACK_ROOT/final/" 2>/dev/null || true
  cp "$BASELINES_ROOT/report.md" "$PACK_ROOT/final/" 2>/dev/null || true
  cp "$BASELINES_ROOT/protocol.json" "$PACK_ROOT/final/" 2>/dev/null || true
  cp "docs/CONTROL_ETA_REBUILD_SPEC.md" "$PACK_ROOT/" 2>/dev/null || true

  cat > "$PACK_ROOT/README.md" <<EOF
# Control Eta Validation-First Pack

- run_root: ${RUN_ROOT}
- train_root: ${TRAIN_ROOT}
- validation_root: ${VALIDATION_ROOT}
- final_root: ${FINAL_ROOT}
- baselines_root: ${BASELINES_ROOT}
- selected_eta_json: ${SELECTION_DIR}/validation_eta_selection.json
- notes:
  - eta grid fixed a priori
  - validation first, then eta selection, then held-out test evaluation
  - test results are not used for eta selection
  - heuristic baselines are matched on window, kappa, annualization, rf, and executed-path metrics
EOF
}

append_unique_eta() {
  local candidate="$1"
  local item
  for item in "${FINAL_ETAS_ARGS[@]:-}"; do
    if [[ "$item" == "$candidate" ]]; then
      return 0
    fi
  done
  FINAL_ETAS_ARGS+=("$candidate")
}

log "[INFO] phase=control_eta_validation_first"
log "[INFO] current_config=${CURRENT_CONFIG}"
log "[INFO] seeds=${SEEDS[*]}"
log "[INFO] etas=${ETAS[*]}"
log "[INFO] kappas=${KAPPAS[*]}"
log "[INFO] validation_window=${VALIDATION_START}~${VALIDATION_END}"
log "[INFO] final_window=${FINAL_START}~${FINAL_END}"
log "[INFO] run_root=${RUN_ROOT}"
log "[INFO] python_cmd=${PYTHON[*]}"
log "[INFO] final_mode=${FINAL_MODE}"

if [[ "$RUN_TRAIN" == "1" && "$SAC_TOTAL_TIMESTEPS" != "0" && "$SAC_TOTAL_TIMESTEPS" -lt "100000" ]]; then
  log "[ERROR] paper_mode training requires SAC_TOTAL_TIMESTEPS >= 100000, or 0 to keep the frozen config default."
  exit 2
fi

run_cmd "materialize_validation_and_final_configs" \
  "${PYTHON[@]}" scripts/materialize_u27_control_eta_paper_configs.py \
    --current-config "$CURRENT_CONFIG" \
    --snapshot-config-out "$SNAPSHOT_CONFIG" \
    --signal-snapshot-out "$SIGNAL_SNAPSHOT" \
    --validation-config-out "$VALIDATION_CONFIG" \
    --final-config-out "$FINAL_CONFIG" \
    --meta-out "$MATERIALIZATION_META" \
    --job-ts "$JOB_TS" \
    --validation-start "$VALIDATION_START" \
    --validation-end "$VALIDATION_END" \
    --final-start "$FINAL_START" \
    --final-end "$FINAL_END" \
    --train-output-root "$TRAIN_ROOT" \
    --validation-output-root "$VALIDATION_ROOT" \
    --final-output-root "$FINAL_ROOT" \
    --sac-total-timesteps "$SAC_TOTAL_TIMESTEPS"

if [[ "$RUN_TRAIN" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_cmd "train_control_seed${seed}" \
      "${PYTHON[@]}" scripts/run_train.py \
        --config "$SNAPSHOT_CONFIG" \
        --model-type prl \
        --seed "$seed" \
        --offline \
        --output-root "$TRAIN_ROOT"
  done
else
  log "[INFO] skip training phase (RUN_TRAIN=${RUN_TRAIN})"
fi

if [[ "$RUN_VALIDATION" == "1" ]]; then
  validation_cmd=(
    "${PYTHON[@]}" scripts/step6_run_matrix.py
    --config "$VALIDATION_CONFIG"
    --model-type prl
    --model-root "$TRAIN_ROOT"
    --seed-model-mode independent
    --seeds "${SEEDS[@]}"
    --kappas "${KAPPAS[@]}"
    --etas "${ETAS[@]}"
    --out "$VALIDATION_ROOT"
    --offline
  )
  if [[ "$MAX_STEPS" != "0" ]]; then
    validation_cmd+=(--max-steps "$MAX_STEPS")
  fi
  run_cmd "validation_eta_frontier" "${validation_cmd[@]}"
  run_cmd "validation_eta_reports" \
    "${PYTHON[@]}" scripts/step6_build_reports.py \
      --root "$VALIDATION_ROOT"
else
  log "[INFO] skip validation eta frontier phase (RUN_VALIDATION=${RUN_VALIDATION})"
fi

SELECTED_ETA=""
if [[ "$RUN_SELECT" == "1" ]]; then
  run_cmd "select_validation_eta" \
    "${PYTHON[@]}" scripts/select_eta_from_validation.py \
      --root "$VALIDATION_ROOT" \
      --output-dir "$SELECTION_DIR" \
      --baseline-eta "$BASELINE_ETA" \
      --positive-kappas "$POSITIVE_KAPPAS" \
      --relative-threshold "$RELATIVE_THRESHOLD"
  if [[ "$DRY_RUN" != "1" ]]; then
    SELECTED_ETA="$(selected_eta_from_json "$SELECTION_DIR/validation_eta_selection.json")"
  fi
  log "[INFO] selected_eta=${SELECTED_ETA:-<none>}"
else
  log "[INFO] skip selection phase (RUN_SELECT=${RUN_SELECT})"
fi

if [[ "$RUN_FINAL" == "1" ]]; then
  FINAL_ETAS_ARGS=()
  if [[ "$FINAL_MODE" == "selected_only" ]]; then
    if [[ -z "$SELECTED_ETA" ]]; then
      log "[ERROR] selected eta is empty; cannot run final selected_only mode."
      exit 2
    fi
    FINAL_ETAS_ARGS=("$SELECTED_ETA")
  elif [[ "$FINAL_MODE" == "selected_plus_baseline" ]]; then
    if [[ -z "$SELECTED_ETA" ]]; then
      log "[ERROR] selected eta is empty; cannot run final selected_plus_baseline mode."
      exit 2
    fi
    append_unique_eta "$BASELINE_ETA"
    append_unique_eta "$SELECTED_ETA"
  elif [[ "$FINAL_MODE" == "full_grid" ]]; then
    FINAL_ETAS_ARGS=("${ETAS[@]}")
  elif [[ "$FINAL_MODE" == "explicit" ]]; then
    if [[ -z "$FINAL_ETAS_RAW" ]]; then
      log "[ERROR] FINAL_MODE=explicit requires FINAL_ETAS."
      exit 2
    fi
    read -r -a FINAL_ETAS_ARGS <<< "$FINAL_ETAS_RAW"
  else
    log "[ERROR] unsupported FINAL_MODE=${FINAL_MODE}"
    exit 2
  fi

  final_cmd=(
    "${PYTHON[@]}" scripts/step6_run_matrix.py
    --config "$FINAL_CONFIG"
    --model-type prl
    --model-root "$TRAIN_ROOT"
    --seed-model-mode independent
    --seeds "${SEEDS[@]}"
    --kappas "${KAPPAS[@]}"
    --etas "${FINAL_ETAS_ARGS[@]}"
    --out "$FINAL_ROOT"
    --offline
  )
  if [[ "$MAX_STEPS" != "0" ]]; then
    final_cmd+=(--max-steps "$MAX_STEPS")
  fi
  run_cmd "final_eta_eval" "${final_cmd[@]}"
  run_cmd "final_eta_reports" \
    "${PYTHON[@]}" scripts/step6_build_reports.py \
      --root "$FINAL_ROOT"
else
  log "[INFO] skip final/test phase (RUN_FINAL=${RUN_FINAL})"
fi

if [[ "$RUN_BASELINES" == "1" ]]; then
  run_cmd "external_heuristic_baselines" \
    "${PYTHON[@]}" scripts/run_external_heuristic_baselines.py \
      --config "$FINAL_CONFIG" \
      --kappas "${KAPPAS[@]}" \
      --out "$BASELINES_ROOT" \
      --offline
else
  log "[INFO] skip external baseline phase (RUN_BASELINES=${RUN_BASELINES})"
fi

if [[ "$RUN_PACK" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY-RUN] build paper pack at ${PACK_ROOT}"
  else
    run_cmd "build_selected_eta_stats" \
      "${PYTHON[@]}" scripts/build_selected_eta_stats.py \
        --final-root "$FINAL_ROOT" \
        --selection-json "$SELECTION_DIR/validation_eta_selection.json" \
        --output-dir "$PACK_STATS_ROOT" \
        --baseline-eta "$BASELINE_ETA"
    run_cmd "build_misalignment_v2" \
      "${PYTHON[@]}" scripts/build_misalignment_v2.py \
        --final-root "$FINAL_ROOT" \
        --selection-json "$SELECTION_DIR/validation_eta_selection.json" \
        --output-dir "$PACK_DIAGNOSTICS_ROOT"
    run_cmd "build_validation_first_tables" \
      "${PYTHON[@]}" scripts/build_control_eta_validation_first_tables.py \
        --validation-root "$VALIDATION_ROOT" \
        --selection-json "$SELECTION_DIR/validation_eta_selection.json" \
        --final-root "$FINAL_ROOT" \
        --baselines-root "$BASELINES_ROOT" \
        --output-dir "$PACK_ROOT" \
        --baseline-eta "$BASELINE_ETA" \
        --selected-stats-csv "$PACK_STATS_ROOT/selected_eta_vs_eta1_stats.csv" \
        --diagnostic-v2-csv "$PACK_DIAGNOSTICS_ROOT/diagnostic_selected_eta_v2.csv"
    run_cmd "build_paper_figures" \
      "${PYTHON[@]}" scripts/build_paper_figures.py \
        --validation-root "$VALIDATION_ROOT" \
        --selection-json "$SELECTION_DIR/validation_eta_selection.json" \
        --final-root "$FINAL_ROOT" \
        --output-dir "$PACK_FIGURES_ROOT" \
        --representative-json "$PACK_DIAGNOSTICS_ROOT/representative_seed_metrics.json" \
        --seedwise-stats-csv "$PACK_STATS_ROOT/selected_eta_seedwise_deltas.csv"
    build_pack
  fi
else
  log "[INFO] skip pack phase (RUN_PACK=${RUN_PACK})"
fi

run_cmd "check_validation_first_outputs" \
  "${PYTHON[@]}" scripts/check_control_eta_validation_first.py \
    --run-root "$RUN_ROOT"

log "[DONE] phase=control_eta_validation_first complete"
log "[DONE] master_log=${MASTER_LOG}"
log "[DONE] run_root=${RUN_ROOT}"
