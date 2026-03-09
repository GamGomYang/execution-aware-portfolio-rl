#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN_DEFAULT="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

SPEC_FILE="/workspace/execution-aware-portfolio-rl/명세"
RESULT_FILE="/workspace/execution-aware-portfolio-rl/결과 분석"

PILOT_SESSION="${PILOT_SESSION:-model_swap_final_pilot_20260309}"
PILOT_WATCH_SESSION="${PILOT_WATCH_SESSION:-model_swap_final_pilot_log_20260309}"
POLL_SEC="${POLL_SEC:-600}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not found: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/spec_n1_n4_${JOB_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
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

append_both() {
  {
    echo ""
    echo "## [$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1"
    cat
  } | tee -a "$SPEC_FILE" "$RESULT_FILE" >> "$MASTER_LOG"
}

append_both "Spec-N1 정책 고정 (${JOB_TS})" <<'EOF'
- hard rule (논문/최종보고): check2_min_positive_seeds >= 6 유지
- soft rule (탐색 전용 라벨): check2_min_positive_seeds >= 5 병행 기록
- 보고 원칙:
  - 본 결론은 hard rule(>=6) 기준으로 작성
  - soft rule(>=5)은 보조 신호로만 표기
EOF

for ETA in 079 080; do
  run_step "spec_n2_final_run_eta_${ETA}" \
    "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "configs/step6_fixedeta_final_test_eta${ETA}_seed10.yaml" \
      --model-type prl \
      --model-root outputs/step3_u27 \
      --seed-model-mode independent \
      --offline \
      --max-steps 0

  run_step "spec_n2_acceptance_hard6_eta_${ETA}" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "outputs/step6_fixedeta_final_test_eta${ETA}/paired_delta.csv" \
      --aggregate "outputs/step6_fixedeta_final_test_eta${ETA}/aggregate.csv" \
      --check2-min-positive-seeds 6 \
      --out-md "outputs/step6_fixedeta_final_test_eta${ETA}/acceptance_report.md" \
      --out-json "outputs/step6_fixedeta_final_test_eta${ETA}/acceptance_report.json" \
      --no-fail-exit

  run_step "spec_n2_acceptance_soft5_eta_${ETA}" \
    "$PYTHON_BIN" scripts/step6_check_acceptance.py \
      --paired "outputs/step6_fixedeta_final_test_eta${ETA}/paired_delta.csv" \
      --aggregate "outputs/step6_fixedeta_final_test_eta${ETA}/aggregate.csv" \
      --check2-min-positive-seeds 5 \
      --out-md "outputs/step6_fixedeta_final_test_eta${ETA}/acceptance_report_min5.md" \
      --out-json "outputs/step6_fixedeta_final_test_eta${ETA}/acceptance_report_min5.json" \
      --no-fail-exit
done

summary_raw="$(
  JOB_TS="$JOB_TS" "$PYTHON_BIN" - <<'PYCODE'
from __future__ import annotations

import json
from pathlib import Path
import csv
import os

job_ts = os.environ["JOB_TS"]
root = Path("outputs")
etas = ["079", "080", "082"]
rows: list[dict[str, object]] = []
for eta in etas:
    run = root / f"step6_fixedeta_final_test_eta{eta}"
    acc6 = json.loads((run / "acceptance_report.json").read_text())
    acc5_path = run / "acceptance_report_min5.json"
    acc5 = json.loads(acc5_path.read_text()) if acc5_path.exists() else None
    with (run / "aggregate.csv").open(newline="") as f:
        agg = list(csv.DictReader(f))

    for k in [0.0005, 0.001]:
        main = [r for r in agg if float(r["kappa"]) == k and r["arm"] == "main"][0]
        base = [r for r in agg if float(r["kappa"]) == k and r["arm"] == "baseline"][0]
        rows.append(
            {
                "eta": float(f"0.{eta}"),
                "kappa": k,
                "hard6_overall_pass": bool(acc6["overall_pass"]),
                "soft5_overall_pass": bool(acc5["overall_pass"]) if acc5 else None,
                "main_median_sharpe": float(main["median_sharpe"]),
                "baseline_median_sharpe": float(base["median_sharpe"]),
                "delta_median_sharpe_main_minus_base": float(main["median_sharpe"]) - float(base["median_sharpe"]),
                "main_turnover_exec": float(main["median_turnover_exec"]),
                "baseline_turnover_exec": float(base["median_turnover_exec"]),
                "collapse_rate_main": float(main["collapse_rate"]),
            }
        )

csv_path = Path(f"outputs/reports/spec_n2_final_eta_compare_{job_ts}.csv")
md_path = Path(f"outputs/reports/spec_n2_final_eta_compare_{job_ts}.md")
json_path = Path(f"outputs/reports/spec_n2_final_eta_compare_{job_ts}.json")

import pandas as pd
df = pd.DataFrame(rows).sort_values(["eta", "kappa"])
df.to_csv(csv_path, index=False)
json_path.write_text(df.to_json(orient="records", indent=2))

best_hard = (
    df[df["hard6_overall_pass"] == True]
    .sort_values(["delta_median_sharpe_main_minus_base"], ascending=False)
    .head(1)
)
best_soft = (
    df[df["soft5_overall_pass"] == True]
    .sort_values(["delta_median_sharpe_main_minus_base"], ascending=False)
    .head(1)
)

lines: list[str] = []
lines.append(f"# Spec-N2 Final ETA Compare ({job_ts})")
lines.append("")
lines.append("| eta | kappa | hard6_pass | soft5_pass | main_median_sharpe | baseline_median_sharpe | delta_main_minus_base | main_turnover | baseline_turnover | collapse_rate |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for row in df.to_dict(orient="records"):
    lines.append(
        f"| {row['eta']} | {row['kappa']} | {row['hard6_overall_pass']} | {row['soft5_overall_pass']} | "
        f"{row['main_median_sharpe']:.9f} | {row['baseline_median_sharpe']:.9f} | "
        f"{row['delta_median_sharpe_main_minus_base']:+.9f} | {row['main_turnover_exec']:.12f} | "
        f"{row['baseline_turnover_exec']:.12f} | {row['collapse_rate_main']:.3f} |"
    )
lines.append("")
if len(best_hard):
    r = best_hard.iloc[0]
    lines.append(f"- best_hard6: eta={r['eta']}, kappa={r['kappa']}, delta={r['delta_median_sharpe_main_minus_base']:+.9f}")
else:
    lines.append("- best_hard6: <none>")
if len(best_soft):
    r = best_soft.iloc[0]
    lines.append(f"- best_soft5: eta={r['eta']}, kappa={r['kappa']}, delta={r['delta_median_sharpe_main_minus_base']:+.9f}")
else:
    lines.append("- best_soft5: <none>")
lines.append(f"- csv: {csv_path}")
lines.append(f"- json: {json_path}")
md_path.write_text("\n".join(lines) + "\n")

print(f"SUMMARY_MD={md_path}")
print(f"SUMMARY_CSV={csv_path}")
print(f"SUMMARY_JSON={json_path}")
PYCODE
)"
printf '%s\n' "$summary_raw" | tee -a "$MASTER_LOG"
SUMMARY_MD="$(printf '%s\n' "$summary_raw" | awk -F= '/^SUMMARY_MD=/{print $2}')"
SUMMARY_CSV="$(printf '%s\n' "$summary_raw" | awk -F= '/^SUMMARY_CSV=/{print $2}')"
SUMMARY_JSON="$(printf '%s\n' "$summary_raw" | awk -F= '/^SUMMARY_JSON=/{print $2}')"

append_both "Spec-N2 실행 완료 (${JOB_TS})" <<EOF
- summary_md: ${SUMMARY_MD}
- summary_csv: ${SUMMARY_CSV}
- summary_json: ${SUMMARY_JSON}
- hard6는 최종 기준, soft5는 탐색 라벨로 병행 기록 완료
EOF

if tmux has-session -t "$PILOT_SESSION" 2>/dev/null; then
  log "[WARN] existing pilot session found. killing: ${PILOT_SESSION}"
  tmux kill-session -t "$PILOT_SESSION"
fi
if tmux has-session -t "$PILOT_WATCH_SESSION" 2>/dev/null; then
  log "[WARN] existing watch session found. killing: ${PILOT_WATCH_SESSION}"
  tmux kill-session -t "$PILOT_WATCH_SESSION"
fi

tmux new-session -d -s "$PILOT_SESSION" \
  "cd $ROOT && PYTHON_BIN=$PYTHON_BIN WAIT_SESSION='' ALGOS='prl sac' SEEDS='0 1 2' ETAS='0.082' KAPPAS='0.0005 0.001' EVAL_START='2024-01-01' EVAL_END='2025-12-31' RL_TIMESTEPS='100000' scripts/run_model_swap_matrix_detached.sh"

tmux new-session -d -s "$PILOT_WATCH_SESSION" \
  "cd $ROOT && SESSION='$PILOT_SESSION' RESULT_FILE='$RESULT_FILE' SPEC_FILE='$SPEC_FILE' POLL_SEC='$POLL_SEC' scripts/watch_model_swap_progress_to_result_log.sh"

append_both "Spec-N3 파일럿 시작 (${JOB_TS})" <<EOF
- pilot_session: ${PILOT_SESSION}
- watch_session: ${PILOT_WATCH_SESSION}
- command: PRL/SAC, seeds=0,1,2, eta=0.082, kappa=0.0005/0.001, eval=2024~2025, RL_TIMESTEPS=100000
- progress_logging: 결과 분석 파일에 ${POLL_SEC}초 간격 append
- stop_rule(Spec-N4): check2<6이면 eta 추가탐색 중단, 모델/신호/학습설정 변경 없는 반복 금지
EOF

log "[DONE] spec_n1_n4 workflow bootstrap completed :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "[DONE] workflow_log=${MASTER_LOG}"
