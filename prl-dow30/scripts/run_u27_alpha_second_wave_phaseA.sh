#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_tune_2022_2023_eta082_seed10.yaml}"
PILOT_SEEDS_RAW="${PILOT_SEEDS:-0 2 5 6}"
CHECK2_HARD="${CHECK2_HARD:-3}"
CHECK2_SOFT="${CHECK2_SOFT:-2}"
MAX_STEPS="${MAX_STEPS:-0}"
CANDIDATES_RAW="${CANDIDATES:-u27_eta082_alpha2_ctrl_20k_r1 u27_eta082_alpha2_cs312_20k_r1 u27_eta082_alpha2_cs61vsmom_20k_r1 u27_eta082_alpha2_cs61resid_20k_r1 u27_eta082_alpha2_cs312resid_20k_r1}"
MATERIALIZE="${MATERIALIZE:-1}"
MATERIALIZE_META="${MATERIALIZE_META:-outputs/reports/u27_alpha_second_wave_materialization.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

read -r -a PILOT_SEEDS <<< "$PILOT_SEEDS_RAW"
read -r -a CANDIDATES <<< "$CANDIDATES_RAW"

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_alpha_second_wave_phaseA_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
SUMMARY_CSV="outputs/reports/u27_alpha_second_wave_phaseA_summary_${JOB_TS}.csv"
SUMMARY_MD="outputs/reports/u27_alpha_second_wave_phaseA_summary_${JOB_TS}.md"
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

materialize_batch() {
  run_step "materialize_alpha_second_wave" \
    "$PYTHON_BIN" scripts/materialize_u27_alpha_second_wave_configs.py \
      --candidates "${CANDIDATES[@]}" \
      --meta-out "$MATERIALIZE_META"
}

train_candidate() {
  local tag="$1"
  local cfg="$2"
  local train_root="$3"
  local seed
  for seed in "${PILOT_SEEDS[@]}"; do
    run_step "train_${tag}_seed${seed}" \
      "$PYTHON_BIN" scripts/run_train.py \
        --config "$cfg" \
        --model-type prl \
        --seed "$seed" \
        --offline \
        --output-root "$train_root"
  done
}

build_summary() {
  CANDIDATES_STR="${CANDIDATES[*]}" \
  CHECK2_HARD="$CHECK2_HARD" \
  CHECK2_SOFT="$CHECK2_SOFT" \
  SUMMARY_CSV="$SUMMARY_CSV" \
  SUMMARY_MD="$SUMMARY_MD" \
  STEP6_CONFIG="$STEP6_CONFIG" \
  "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd
import yaml

candidates = [token for token in os.environ["CANDIDATES_STR"].split() if token]
hard_thr = int(os.environ["CHECK2_HARD"])
soft_thr = int(os.environ["CHECK2_SOFT"])
out_csv = Path(os.environ["SUMMARY_CSV"])
out_md = Path(os.environ["SUMMARY_MD"])
step6_config = os.environ["STEP6_CONFIG"]


def _extract_check2_min_positive(payload: dict) -> int | None:
    for check in payload.get("checks", []):
        name = str(check.get("name", ""))
        if not name.startswith("check2_mode="):
            continue
        vals = []
        for item in check.get("details", {}).get("per_kappa", []):
            try:
                vals.append(int(item.get("n_positive_delta_sharpe")))
            except Exception:
                continue
        if vals:
            return min(vals)
    return None


rows: list[dict[str, object]] = []
for tag in candidates:
    cfg_path = Path("configs/exp") / f"{tag}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    signal_names = list(((cfg.get("signals", {}) or {}).get("signal_names", [])))

    step6_root = Path("outputs") / f"step6_{tag}"
    hard_path = step6_root / "acceptance_report_hard3.json"
    soft_path = step6_root / "acceptance_report_soft2.json"
    aggregate_path = step6_root / "aggregate.csv"
    paired_path = step6_root / "paired_delta.csv"

    hard_payload = json.loads(hard_path.read_text()) if hard_path.exists() else None
    soft_payload = json.loads(soft_path.read_text()) if soft_path.exists() else None
    agg_rows = list(csv.DictReader(aggregate_path.open())) if aggregate_path.exists() else []
    paired_rows = list(csv.DictReader(paired_path.open())) if paired_path.exists() else []

    def _agg_row(arm: str, kappa: float):
        return next((r for r in agg_rows if r.get("arm") == arm and abs(float(r.get("kappa", "nan")) - kappa) < 1e-12), None)

    def _deltas(kappa: float):
        return [float(r["delta_sharpe"]) for r in paired_rows if abs(float(r["kappa"]) - kappa) < 1e-12]

    deltas_k0005 = _deltas(0.0005)
    deltas_k001 = _deltas(0.001)
    k001_negative = [int(r["seed"]) for r in paired_rows if abs(float(r["kappa"]) - 0.001) < 1e-12 and float(r["delta_sharpe"]) <= 0.0]
    main_k0005 = _agg_row("main", 0.0005)
    main_k001 = _agg_row("main", 0.001)

    row = {
        "tag": tag,
        "signals": ",".join(signal_names),
        "hard_pass": bool(hard_payload.get("overall_pass")) if hard_payload else False,
        "soft_pass": bool(soft_payload.get("overall_pass")) if soft_payload else False,
        "hard_check2_min_positive": _extract_check2_min_positive(hard_payload) if hard_payload else None,
        "soft_check2_min_positive": _extract_check2_min_positive(soft_payload) if soft_payload else None,
        "k0005_positive": sum(1 for x in deltas_k0005 if x > 0.0),
        "k0005_mean_delta_sharpe": float(pd.Series(deltas_k0005).mean()) if deltas_k0005 else None,
        "k001_positive": sum(1 for x in deltas_k001 if x > 0.0),
        "k001_mean_delta_sharpe": float(pd.Series(deltas_k001).mean()) if deltas_k001 else None,
        "k001_median_delta_sharpe": float(pd.Series(deltas_k001).median()) if deltas_k001 else None,
        "k001_negative_seeds": json.dumps(k001_negative),
        "k001_main_median_sharpe": float(main_k001["median_sharpe"]) if main_k001 else None,
        "k001_collapse_rate": float(main_k001["collapse_rate"]) if main_k001 else None,
        "k0005_main_median_sharpe": float(main_k0005["median_sharpe"]) if main_k0005 else None,
        "step6_root": str(step6_root),
    }
    rows.append(row)

df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(
        ["hard_pass", "k001_positive", "k001_mean_delta_sharpe", "k0005_positive", "k0005_mean_delta_sharpe"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

lines: list[str] = []
lines.append("# U27 Alpha Second Wave Phase A Summary")
lines.append("")
lines.append(f"- validation_step6_config: {step6_config}")
lines.append(f"- hard_threshold: {hard_thr}")
lines.append(f"- soft_threshold: {soft_thr}")
lines.append(f"- csv: {out_csv}")
lines.append("")
lines.append("| rank | tag | signals | hard_pass | soft_pass | k001_positive | k001_mean_delta | k0005_positive | k0005_mean_delta | k001_negative_seeds |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for idx, row in enumerate(df.to_dict(orient="records"), start=1):
    lines.append(
        f"| {idx} | {row['tag']} | {row['signals']} | {row['hard_pass']} | {row['soft_pass']} | "
        f"{row['k001_positive']} | {row['k001_mean_delta_sharpe']} | {row['k0005_positive']} | "
        f"{row['k0005_mean_delta_sharpe']} | {row['k001_negative_seeds']} |"
    )
if not df.empty:
    best = df.iloc[0]
    lines.append("")
    lines.append(
        f"- top_candidate: {best['tag']} :: signals={best['signals']} :: "
        f"hard_pass={best['hard_pass']} :: k001_positive={best['k001_positive']} :: "
        f"k001_mean_delta_sharpe={best['k001_mean_delta_sharpe']}"
    )
out_md.write_text("\n".join(lines) + "\n")
print(f"SUMMARY_CSV={out_csv}")
print(f"SUMMARY_MD={out_md}")
PYCODE
}

if [[ "$MATERIALIZE" == "1" ]]; then
  materialize_batch
fi

log "[INFO] phase=alpha_second_wave_A"
log "[INFO] candidates=${CANDIDATES[*]}"
log "[INFO] pilot_seeds=${PILOT_SEEDS[*]}"
log "[INFO] step6_config=${STEP6_CONFIG}"
log "[INFO] max_steps=${MAX_STEPS}"
log "[INFO] materialize_meta=${MATERIALIZE_META}"

for tag in "${CANDIDATES[@]}"; do
  cfg="configs/exp/${tag}.yaml"
  train_root="outputs/${tag}"
  step6_root="outputs/step6_${tag}"

  if [[ ! -f "$cfg" ]]; then
    log "[ERROR] missing config: ${cfg}"
    exit 1
  fi

  train_candidate "$tag" "$cfg" "$train_root"

  run_step "step6_${tag}" \
    "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$STEP6_CONFIG" \
      --model-type prl \
      --model-root "$train_root" \
      --seed-model-mode independent \
      --seeds "${PILOT_SEEDS[@]}" \
      --kappas 0.0 0.0005 0.001 \
      --etas 0.082 \
      --out "$step6_root" \
      --offline \
      --max-steps "$MAX_STEPS"

  run_step "acceptance_hard_${tag}" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "${step6_root}/paired_delta.csv" \
      --aggregate "${step6_root}/aggregate.csv" \
      --check2-min-positive-seeds "$CHECK2_HARD" \
      --out-md "${step6_root}/acceptance_report_hard3.md" \
      --out-json "${step6_root}/acceptance_report_hard3.json" \
      --no-fail-exit

  run_step "acceptance_soft_${tag}" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "${step6_root}/paired_delta.csv" \
      --aggregate "${step6_root}/aggregate.csv" \
      --check2-min-positive-seeds "$CHECK2_SOFT" \
      --out-md "${step6_root}/acceptance_report_soft2.md" \
      --out-json "${step6_root}/acceptance_report_soft2.json" \
      --no-fail-exit
done

summary_raw="$(build_summary)"
printf '%s\n' "$summary_raw" | tee -a "$MASTER_LOG"
log "[DONE] phase=alpha_second_wave_A complete"
log "[DONE] master_log=${MASTER_LOG}"
