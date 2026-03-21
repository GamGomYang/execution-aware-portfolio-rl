#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
STEP6_CONFIG="${STEP6_CONFIG:-configs/step6_fixedeta_final_test_eta082_seed10.yaml}"
FULL_SEEDS_RAW="${FULL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
BASELINE_TAG="${BASELINE_TAG:-u27_eta082_ctrl20k_r1}"
PHASEB_CANDIDATES_RAW="${PHASEB_CANDIDATES:-u27_eta082_m070_cg03_20k_r1 u27_eta082_m070_cg06_20k_r1}"
CHECK2_HARD="${CHECK2_HARD:-6}"
CHECK2_SOFT="${CHECK2_SOFT:-5}"
MAX_STEPS="${MAX_STEPS:-252}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

read -r -a FULL_SEEDS <<< "$FULL_SEEDS_RAW"
read -r -a PHASEB_CANDIDATES <<< "$PHASEB_CANDIDATES_RAW"
CANDIDATES=("$BASELINE_TAG")
for item in "${PHASEB_CANDIDATES[@]}"; do
  [[ "$item" == "$BASELINE_TAG" ]] && continue
  CANDIDATES+=("$item")
done

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_eta082_phaseB_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
SUMMARY_CSV="outputs/reports/u27_eta082_phaseB_summary_${JOB_TS}.csv"
SUMMARY_MD="outputs/reports/u27_eta082_phaseB_summary_${JOB_TS}.md"
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

train_candidate() {
  local tag="$1"
  local cfg="$2"
  local train_root="$3"
  local seed
  for seed in "${FULL_SEEDS[@]}"; do
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
  "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd

candidates = [token for token in os.environ["CANDIDATES_STR"].split() if token]
hard_thr = int(os.environ["CHECK2_HARD"])
soft_thr = int(os.environ["CHECK2_SOFT"])
out_csv = Path(os.environ["SUMMARY_CSV"])
out_md = Path(os.environ["SUMMARY_MD"])


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
for base_tag in candidates:
    tag = f"{base_tag}_full10"
    step6_root = Path("outputs") / f"step6_{tag}"
    hard_path = step6_root / "acceptance_report_hard6.json"
    soft_path = step6_root / "acceptance_report_soft5.json"
    aggregate_path = step6_root / "aggregate.csv"
    paired_path = step6_root / "paired_delta.csv"

    hard_payload = json.loads(hard_path.read_text()) if hard_path.exists() else None
    soft_payload = json.loads(soft_path.read_text()) if soft_path.exists() else None
    agg_rows = list(csv.DictReader(aggregate_path.open())) if aggregate_path.exists() else []
    paired_rows = list(csv.DictReader(paired_path.open())) if paired_path.exists() else []

    main_k001 = next((r for r in agg_rows if r.get("arm") == "main" and abs(float(r.get("kappa", "nan")) - 0.001) < 1e-12), None)
    base_k001 = next((r for r in agg_rows if r.get("arm") == "baseline" and abs(float(r.get("kappa", "nan")) - 0.001) < 1e-12), None)
    deltas_k001 = [float(r["delta_sharpe"]) for r in paired_rows if abs(float(r["kappa"]) - 0.001) < 1e-12]

    rows.append(
        {
            "tag": tag,
            "hard_threshold": hard_thr,
            "soft_threshold": soft_thr,
            "hard_pass": bool(hard_payload.get("overall_pass")) if hard_payload else False,
            "soft_pass": bool(soft_payload.get("overall_pass")) if soft_payload else False,
            "hard_check2_min_positive": _extract_check2_min_positive(hard_payload) if hard_payload else None,
            "soft_check2_min_positive": _extract_check2_min_positive(soft_payload) if soft_payload else None,
            "k001_positive": sum(1 for x in deltas_k001 if x > 0.0),
            "k001_mean_delta_sharpe": float(pd.Series(deltas_k001).mean()) if deltas_k001 else None,
            "k001_median_delta_sharpe": float(pd.Series(deltas_k001).median()) if deltas_k001 else None,
            "k001_main_median_sharpe": float(main_k001["median_sharpe"]) if main_k001 else None,
            "k001_baseline_median_sharpe": float(base_k001["median_sharpe"]) if base_k001 else None,
            "k001_main_minus_base": (float(main_k001["median_sharpe"]) - float(base_k001["median_sharpe"])) if main_k001 and base_k001 else None,
            "k001_collapse_rate": float(main_k001["collapse_rate"]) if main_k001 else None,
            "step6_root": str(step6_root),
        }
    )

df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(
        ["hard_pass", "soft_pass", "k001_mean_delta_sharpe", "k001_main_median_sharpe"],
        ascending=[False, False, False, False],
        na_position="last",
    )
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

lines: list[str] = []
lines.append("# U27 ETA082 Phase B Summary")
lines.append("")
lines.append(f"- hard_threshold: {hard_thr}")
lines.append(f"- soft_threshold: {soft_thr}")
lines.append(f"- csv: {out_csv}")
lines.append("")
lines.append("| tag | hard_pass | soft_pass | k001_positive | k001_mean_delta | k001_main_minus_base | k001_collapse_rate |")
lines.append("| --- | --- | --- | --- | --- | --- | --- |")
for row in df.to_dict(orient="records"):
    lines.append(
        f"| {row['tag']} | {row['hard_pass']} | {row['soft_pass']} | {row['k001_positive']} | "
        f"{row['k001_mean_delta_sharpe']} | {row['k001_main_minus_base']} | {row['k001_collapse_rate']} |"
    )
if not df.empty:
    best = df.iloc[0]
    lines.append("")
    lines.append(
        f"- top_candidate: {best['tag']} (hard_pass={best['hard_pass']}, soft_pass={best['soft_pass']}, "
        f"k001_mean_delta_sharpe={best['k001_mean_delta_sharpe']})"
    )
out_md.write_text("\n".join(lines) + "\n")
print(f"SUMMARY_CSV={out_csv}")
print(f"SUMMARY_MD={out_md}")
PYCODE
}

log "[INFO] phase=B"
log "[INFO] candidates=${CANDIDATES[*]}"
log "[INFO] full_seeds=${FULL_SEEDS[*]}"
log "[INFO] step6_config=${STEP6_CONFIG}"

for base_tag in "${CANDIDATES[@]}"; do
  cfg="configs/exp/${base_tag}.yaml"
  tag="${base_tag}_full10"
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
      --seeds "${FULL_SEEDS[@]}" \
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
      --out-md "${step6_root}/acceptance_report_hard6.md" \
      --out-json "${step6_root}/acceptance_report_hard6.json" \
      --no-fail-exit

  run_step "acceptance_soft_${tag}" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "${step6_root}/paired_delta.csv" \
      --aggregate "${step6_root}/aggregate.csv" \
      --check2-min-positive-seeds "$CHECK2_SOFT" \
      --out-md "${step6_root}/acceptance_report_soft5.md" \
      --out-json "${step6_root}/acceptance_report_soft5.json" \
      --no-fail-exit
done

summary_raw="$(build_summary)"
printf '%s\n' "$summary_raw" | tee -a "$MASTER_LOG"
log "[DONE] phase=B complete"
log "[DONE] master_log=${MASTER_LOG}"
