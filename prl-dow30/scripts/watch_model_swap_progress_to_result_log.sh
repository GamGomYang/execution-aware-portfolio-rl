#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
SESSION="${SESSION:?SESSION is required}"
RESULT_FILE="${RESULT_FILE:?RESULT_FILE is required}"
SPEC_FILE="${SPEC_FILE:-}"
POLL_SEC="${POLL_SEC:-600}"

cd "$ROOT"

now_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

append_result() {
  cat >> "$RESULT_FILE"
}

append_spec_if_set() {
  if [[ -n "$SPEC_FILE" ]]; then
    cat >> "$SPEC_FILE"
  else
    cat >/dev/null
  fi
}

latest_model_swap_log() {
  ls -1dt outputs/logs/model_swap_matrix_*/master.log 2>/dev/null | head -n 1 || true
}

extract_out_root() {
  local log_path="$1"
  rg -n "\\[INFO\\] out_root=" "$log_path" | tail -n 1 | sed 's/.*out_root=//' || true
}

snapshot_once() {
  local tag="$1"
  local ts
  ts="$(now_utc)"
  local log_path
  log_path="$(latest_model_swap_log)"
  local train_count="0"
  local done_count="0"
  local wait_count="0"
  local out_root="<unknown>"
  local last_line="<none>"

  if [[ -n "$log_path" && -f "$log_path" ]]; then
    train_count="$(rg -c "\\[TRAIN\\] algo=" "$log_path" || echo 0)"
    done_count="$(rg -c "\\[DONE\\] model-swap detached matrix completed" "$log_path" || echo 0)"
    wait_count="$(rg -c "\\[WAIT\\] session" "$log_path" || echo 0)"
    out_root="$(extract_out_root "$log_path")"
    last_line="$(tail -n 1 "$log_path" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
  fi

  append_result <<EOF

## [${ts}] Model-Swap Pilot Progress (${tag})
- session: ${SESSION}
- run_log: ${log_path:-<none>}
- out_root: ${out_root}
- train_events_seen: ${train_count}
- wait_events_seen: ${wait_count}
- done_markers_seen: ${done_count}
- last_log_line: ${last_line}
EOF
}

write_final_summary() {
  local log_path
  log_path="$(latest_model_swap_log)"
  if [[ -z "$log_path" || ! -f "$log_path" ]]; then
    return 0
  fi

  local out_root
  out_root="$(extract_out_root "$log_path")"
  if [[ -z "$out_root" || ! -d "$out_root/reports" ]]; then
    return 0
  fi

  local summary_out="outputs/reports/model_swap_final_pilot_summary_$(date -u +%Y%m%dT%H%M%SZ).md"
  local summary_json="${summary_out%.md}.json"

  /workspace/execution-aware-portfolio-rl/.venv/bin/python - <<'PYCODE' "$out_root" "$summary_out" "$summary_json"
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

out_root = Path(sys.argv[1])
summary_out = Path(sys.argv[2])
summary_json = Path(sys.argv[3])

reports = out_root / "reports"
rl = pd.read_csv(reports / "model_swap_rl_aggregate.csv")
vs = pd.read_csv(reports / "model_swap_vs_sac.csv")

best_overall = rl.sort_values("median_sharpe_net_lin", ascending=False).iloc[0]
subset = rl[(rl["eta"] == 0.082) & (rl["kappa"].isin([0.0005, 0.001]))]
subset = subset.sort_values(["kappa", "algo"])

vs_subset = vs[(vs["eta"] == 0.082) & (vs["kappa"].isin([0.0005, 0.001]))].sort_values(["kappa", "algo"])

payload = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "out_root": str(out_root),
    "best_overall": {
        "algo": str(best_overall["algo"]),
        "eta": float(best_overall["eta"]),
        "kappa": float(best_overall["kappa"]),
        "median_sharpe_net_lin": float(best_overall["median_sharpe_net_lin"]),
        "median_cumret_net_lin": float(best_overall["median_cumret_net_lin"]),
        "n_positive_sharpe": int(best_overall["n_positive_sharpe"]),
        "n_seeds": int(best_overall["n_seeds"]),
    },
    "eta_0082_rl_rows": subset.to_dict(orient="records"),
    "eta_0082_vs_sac_rows": vs_subset.to_dict(orient="records"),
}
summary_json.write_text(json.dumps(payload, indent=2))

lines: list[str] = []
lines.append("# Model-Swap Final Pilot Summary")
lines.append("")
lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
lines.append(f"- out_root: {out_root}")
lines.append("")
lines.append("## Best Overall")
best = payload["best_overall"]
lines.append(
    f"- algo={best['algo']}, eta={best['eta']}, kappa={best['kappa']}, "
    f"median_sharpe={best['median_sharpe_net_lin']:.6f}, "
    f"median_cumret={best['median_cumret_net_lin']:.6f}, "
    f"positive={best['n_positive_sharpe']}/{best['n_seeds']}"
)
lines.append("")
lines.append("## RL Aggregate @ eta=0.082")
lines.append("| algo | kappa | n_seeds | median_sharpe | median_cumret | n_positive_sharpe | median_turnover |")
lines.append("| --- | --- | --- | --- | --- | --- | --- |")
for row in payload["eta_0082_rl_rows"]:
    lines.append(
        f"| {row['algo']} | {row['kappa']} | {int(row['n_seeds'])} | {float(row['median_sharpe_net_lin']):.6f} | "
        f"{float(row['median_cumret_net_lin']):.6f} | {int(row['n_positive_sharpe'])} | {float(row['median_turnover_exec']):.9f} |"
    )
lines.append("")
lines.append("## Delta vs SAC @ eta=0.082")
lines.append("| algo | kappa | n | median_delta_sharpe_vs_sac | n_positive_delta_vs_sac |")
lines.append("| --- | --- | --- | --- | --- |")
for row in payload["eta_0082_vs_sac_rows"]:
    lines.append(
        f"| {row['algo']} | {row['kappa']} | {int(row['n'])} | {float(row['median_delta_sharpe_vs_sac']):+.6f} | {int(row['n_positive_delta_vs_sac'])} |"
    )
summary_out.write_text("\n".join(lines) + "\n")
print(summary_out)
print(summary_json)
PYCODE

  local md_path="$summary_out"
  local js_path="$summary_json"
  local ts
  ts="$(now_utc)"
  append_result <<EOF

## [${ts}] Model-Swap Pilot Final Summary
- out_root: ${out_root}
- summary_md: ${md_path}
- summary_json: ${js_path}
EOF
  append_spec_if_set <<EOF

## [${ts}] Model-Swap Pilot Final Summary
- out_root: ${out_root}
- summary_md: ${md_path}
- summary_json: ${js_path}
EOF
}

snapshot_once "start"
while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep "$POLL_SEC"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    snapshot_once "running"
  fi
done
snapshot_once "finished"
write_final_summary
