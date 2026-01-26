# Gate3 screening (risk-align, W1+W2)

## 운영 원칙
- 타임아웃 회피를 위해 seed 분할 실행(per-seed `--output-root`)을 기본으로 한다.
- 분석/판정은 반드시 PACK(합본) root 기준으로 수행한다. run_index를 concat해 `metrics.csv`/`regime_metrics.csv`와 동일한 seeds 집합을 가진다.
- 항상 `--run-index` 필터를 사용한다.
- eval window는 W1(2024-01-01~2025-12-30) / W2(2022-02-15~2023-12-29) 고정이며, 정책 문서(`outputs/exp_runs/gate3/gate3_window_policy.md`)와 일치해야 한다.

## 실행 커맨드(예시)
- Baseline per-seed: `for s in 0 1 2 3 4; do TS=$(date +%Y%m%d_%H%M%S); python -m scripts.run_all --config configs/exp/gate3_reference_baseline_sac_W1W2.yaml --model-types baseline --seeds $s --output-root outputs/exp_runs/gate3/reference_baseline_sac_seed${s}/${TS}; done`
- Candidate A(C1 beta=0.3): 동일 루프로 `configs/exp/gate3_m07_C1_varbeta_03.yaml` + `--model-types prl`
- Candidate B(C2 gamma=0.3): 동일 루프로 `configs/exp/gate3_m07_C2_cvargamma_03.yaml` + `--model-types prl`
- PACK(합본) 생성(예시, ref): seed별 latest를 찾아 concat → `outputs/exp_runs/gate3/reference_baseline_sac_PACK/reports/{metrics.csv,regime_metrics.csv,run_index.json}`
- Analyze(각 PACK): `python -m scripts.analyze_paper_results --metrics <pack>/reports/metrics.csv --regime-metrics <pack>/reports/regime_metrics.csv --run-index <pack>/reports/run_index.json --output-dir <pack>/reports`
- Leaderboard: ref PACK vs cand PACK들을 `scripts.build_gate3_leaderboard.py`(gate2와 동일 인터페이스)로 전달하거나, 임시로 pandas concat 후 아래 PASS 규칙을 적용해 CSV/MD를 생성한다.

## 판정 규칙
- PASS: ΔSharpe_net_exp ≥ +0.20 (cand - ref, seeds 평균, all window) AND MDD_net_exp 악화 없음(guardrail 예: ref 대비 -0.20 이하로 악화 금지).
- FAIL: ΔSharpe_net_exp ≤ -0.05 OR MDD_net_exp가 guardrail 이하로 악화 OR turnover > 1.2x ref 반복.
- BORDERLINE: 위 두 조건 사이에 위치(±0.05~0.20 구간). 필요 시 추가 seed/긴 timesteps 또는 beta/gamma 0.1/1.0 실험으로 재평가.
- Score(정렬용): `Score = sharpe_net_exp - 0.25*|mdd_net_exp| - 0.10*avg_turnover` (tail 개선 확인 시 보조 참고).

## 체크리스트(강제)
- [ ] 모든 seed output_root에 `reports/metrics.csv`, `regime_metrics.csv`, `run_index.json` 존재
- [ ] PACK에 seeds=0..4 전부 포함(run_id 집합 일치)
- [ ] eval_window 값이 W1/W2로 기록되어 있음
- [ ] analyze는 run_index 필터 사용
- [ ] leaderboard/summary에서 PASS/FAIL/BORDERLINE, Score, ΔSharpe, ΔMDD를 ref 대비로 제시
