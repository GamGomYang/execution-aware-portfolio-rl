#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PY="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
PYTEST="/workspace/execution-aware-portfolio-rl/.venv/bin/pytest"
CONFIG="configs/prl_100k_signals.yaml"

cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/signal_state_detached_${JOB_TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
SUMMARY_JSON="outputs/reports/signal_state_detached_summary_${JOB_TS}.json"

echo "[START] job_ts=${JOB_TS}" | tee -a "$MASTER_LOG"
echo "[INFO] root=${ROOT}" | tee -a "$MASTER_LOG"

run_step() {
  local name="$1"
  shift
  echo "[STEP-START] ${name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  echo "[STEP-END] ${name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
}

# 1) 신규/회귀 테스트 + 게이트
run_step "pytest_signal_state" "$PYTEST" -q tests/test_env_signal_state_obs_dim.py tests/test_signature_includes_signal_state.py tests/test_signal_off_backcompat_signature.py
run_step "pytest_regression" "$PYTEST" -q tests/test_env.py tests/test_eval_fails_on_mismatch.py tests/test_model_naming_and_eval_defaults.py tests/test_run_metadata_contains_lock_fields.py
run_step "gate_commit1" "$PY" scripts/step6_commit1_gate.py
run_step "gate_commit2" "$PY" scripts/step6_commit2_gate.py
run_step "gate_commit3" "$PY" scripts/step6_commit3_gate.py

# 2) 스모크 (seed=0) train/eval
run_step "smoke_train_seed0" "$PY" scripts/run_train.py --config "$CONFIG" --model-type prl --seed 0 --offline
run_step "smoke_eval_seed0" "$PY" scripts/run_eval.py --config "$CONFIG" --model-type prl --seed 0 --offline

# 3) seed 1..9 학습 (seed0은 smoke로 이미 수행)
for seed in 1 2 3 4 5 6 7 8 9; do
  run_step "train_seed_${seed}" "$PY" scripts/run_train.py --config "$CONFIG" --model-type prl --seed "$seed" --offline
done

# 4) 요약 산출
"$PY" - <<'PYCODE' | tee -a "$MASTER_LOG"
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

reports = Path("outputs/reports")
metas = []
for p in reports.glob("run_metadata_*.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get("model_type") != "prl":
        continue
    if not str(d.get("config_path", "")).endswith("configs/prl_100k_signals.yaml"):
        continue
    seed = d.get("seed")
    if seed is None:
        continue
    metas.append({
        "file": str(p),
        "run_id": d.get("run_id"),
        "seed": int(seed),
        "created_at": d.get("created_at"),
        "model_path": (d.get("artifact_paths") or d.get("artifacts") or {}).get("model_path"),
        "obs_dim_expected": d.get("obs_dim_expected"),
        "env_signature_hash": d.get("env_signature_hash"),
        "feature_flags": d.get("feature_flags", {}),
    })

if not metas:
    print("[SUMMARY] no matching run_metadata found")
else:
    df = pd.DataFrame(metas).sort_values(["seed", "created_at"])
    latest = df.groupby("seed", as_index=False).tail(1).sort_values("seed")
    print("[SUMMARY] latest metadata per seed")
    print(latest[["seed", "run_id", "created_at", "obs_dim_expected"]].to_string(index=False))
    got_seeds = latest["seed"].tolist()
    print(f"[SUMMARY] unique_seeds={got_seeds} n={len(got_seeds)}")
PYCODE

# JSON summary snapshot
"$PY" - <<PYCODE
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path("$SUMMARY_JSON")
summary_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "job_ts": "$JOB_TS",
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "master_log": "$MASTER_LOG",
    "config": "$CONFIG",
    "model_type": "prl",
    "seed_scope": list(range(10)),
    "note": "seed0 trained in smoke phase, seeds1-9 trained in loop",
}
summary_path.write_text(json.dumps(payload, indent=2))
print(f"[DONE] wrote summary json: {summary_path}")
PYCODE

echo "[DONE] detached workflow completed :: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
