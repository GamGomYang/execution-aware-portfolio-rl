#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PY="/workspace/execution-aware-portfolio-rl/.venv/bin/python"
CONFIG="configs/prl_100k_signals_u27.yaml"
cd "$ROOT"
export PYTHONPATH="."

JOB_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/logs/u27_seed10_detached_${JOB_TS}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
AUDIT_JSON="outputs/reports/u27_seed10_audit_${JOB_TS}.json"
export JOB_TS
export AUDIT_JSON

echo "[START] job_ts=${JOB_TS}" | tee -a "$MASTER_LOG"

timed_step() {
  local step="$1"; shift
  echo "[STEP-START] ${step} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  echo "[STEP-END] ${step} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
}

for seed in 0 1 2 3 4 5 6 7 8 9; do
  timed_step "train_seed_${seed}" "$PY" scripts/run_train.py --config "$CONFIG" --model-type prl --seed "$seed" --offline
done

timed_step "final_audit" "$PY" - <<'PYCODE'
from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
import yaml

root = Path('.')
config_path = 'configs/prl_100k_signals_u27.yaml'
meta_dir = root / 'outputs' / 'reports'
log_dir = root / 'outputs' / 'logs'

cfg = yaml.safe_load((root / config_path).read_text())
fixed_assets = cfg['universe']['fixed_asset_list']

rows = []
for p in meta_dir.glob('run_metadata_*.json'):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get('model_type') != 'prl':
        continue
    if d.get('config_path') != config_path:
        continue
    seed = d.get('seed')
    if seed is None:
        continue
    model_path = Path((d.get('artifact_paths') or d.get('artifacts') or {}).get('model_path', ''))
    rows.append({
        'path': str(p),
        'seed': int(seed),
        'run_id': d.get('run_id'),
        'created_at': d.get('created_at'),
        'num_assets': d.get('num_assets'),
        'asset_count': len(d.get('asset_list') or []),
        'obs_dim_expected': d.get('obs_dim_expected'),
        'feature_flags': d.get('feature_flags') or {},
        'data_manifest_hash': d.get('data_manifest_hash'),
        'env_signature_hash': d.get('env_signature_hash'),
        'model_exists': model_path.exists(),
    })

if not rows:
    raise SystemExit('No run_metadata found for u27 config')

df = pd.DataFrame(rows).sort_values(['seed', 'created_at'])
latest = df.groupby('seed', as_index=False).tail(1).sort_values('seed').reset_index(drop=True)

sig_names_ok = latest['feature_flags'].apply(lambda x: x.get('signal_names') == ['reversal_5d', 'short_term_reversal'])
sig_state_ok = latest['feature_flags'].apply(lambda x: bool(x.get('signal_state', False)))

manifest = json.loads((root / cfg['data']['processed_dir'] / 'data_manifest.json').read_text())
manifest_assets = manifest.get('asset_list') or manifest.get('kept_tickers') or []

selected = json.loads((root / 'outputs' / 'diagnostics' / 'signal_scan_u27' / 'selected_signals.json').read_text())

checks = {
    'metadata_n_seeds_10': bool(len(latest) == 10 and set(latest['seed']) == set(range(10))),
    'metadata_num_assets_27_all': bool((latest['num_assets'] == 27).all()),
    'metadata_asset_count_27_all': bool((latest['asset_count'] == 27).all()),
    'metadata_obs_dim_918_all': bool((latest['obs_dim_expected'] == 918).all()),
    'metadata_signal_state_true_all': bool(sig_state_ok.all()),
    'metadata_signal_names_fixed_all': bool(sig_names_ok.all()),
    'metadata_data_manifest_hash_single': bool(latest['data_manifest_hash'].nunique() == 1),
    'metadata_env_signature_hash_single': bool(latest['env_signature_hash'].nunique() == 1),
    'metadata_model_files_exist_all': bool(latest['model_exists'].all()),
    'config_fixed_list_count_27': bool(len(fixed_assets) == 27),
    'config_fixed_list_no_trv': bool('TRV' not in fixed_assets),
    'manifest_num_assets_27': bool(int(manifest.get('num_assets', -1)) == 27),
    'manifest_asset_count_27': bool(len(manifest_assets) == 27),
    'diagnose_ic_window_train_only': bool(selected.get('ic_start') == '2010-01-01' and selected.get('ic_end') == '2021-12-31'),
    'diagnose_selected_exact_target': bool(selected.get('selected_signals') == ['reversal_5d', 'short_term_reversal']),
}

print('[AUDIT] latest metadata rows')
print(latest[['seed','run_id','num_assets','asset_count','obs_dim_expected']].to_string(index=False))
print('\n[AUDIT] checks')
for k, v in checks.items():
    print(f'{k}={v}')

payload = {
    'config_path': config_path,
    'n_latest_rows': int(len(latest)),
    'latest_rows': latest[['seed','run_id','created_at','num_assets','asset_count','obs_dim_expected','data_manifest_hash','env_signature_hash']].to_dict(orient='records'),
    'checks': checks,
    'all_pass': bool(all(checks.values())),
}

out_path = root / os.environ['AUDIT_JSON']
out_path.write_text(json.dumps(payload, indent=2))
print(f'\n[AUDIT] json={out_path}')
print(f'[AUDIT] all_pass={payload["all_pass"]}')
PYCODE

echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"
