# Gate1 스크리닝 운영 가이드 (STEP 2 확장 명세 v1.1)

기존 STEP 2 명세를 유지하면서, baseline 기준·재현성·속도·랭킹 규칙을 고정한다. Gate1은 **방향성 확인 단계**이며 통계 검정은 참고용이다.

## 2-A) 고정 운영 규칙
- **force_refresh=false 고정** (Gate1~Gate3). 데이터/피처 파이프라인을 의도적으로 바꿀 때만 true.
- **Reference baseline_sac 1회**: Gate1 시작 시 baseline_sac만 1회 실행해 기준값을 고정(W1, seeds=1, 10k~20k timesteps, trace off, no baselines, no step4, force_refresh=false).
- **후보(PRL)**: 이후 9~10개 후보는 PRL만 실행. Gate2부터 상위 1~2개만 baseline_sac 재실행.
- **Seed**: Gate1 1차 스크리닝 seeds=1. 필요 시 상위 2~3개만 seeds=2 미니 재검증(순위 뒤집힘 여부만 확인).
- **통계 검정 해석**: n_seeds<=2에서는 p-value/CI는 참고용. 판단은 지표(Sharpe_net_exp, turnover, MDD)와 스코어로만 수행.

## 2-B) 공통 FAST 설정 (원문 유지 + 강제 키)
- timesteps: 10k~20k, seeds: 1(최대 2), eval window: W1만.
- eval.write_trace=false(또는 trace_stride>=10), eval.run_baselines=false, eval.write_step4=false.
- **force_refresh=false**, **--output-root 필수** (e.g., `outputs/exp_runs/gate1/<EXP>/<TS>/`).
- 분석 시 **반드시 --run-index** 사용.

## 2-C) 실행 순서
1) **Reference baseline_sac** (필수)  
   - config: `configs/exp/gate1_reference_baseline_sac_W1.yaml`  
   - run: `python -m scripts.run_all --config configs/exp/gate1_reference_baseline_sac_W1.yaml --model-types baseline --output-root outputs/exp_runs/gate1/reference_baseline_sac/<TS>`  
   - analyze (필터): `python -m scripts.analyze_paper_results --metrics <root>/reports/metrics.csv --regime-metrics <root>/reports/regime_metrics.csv --run-index <root>/reports/run_index.json --output-dir <root>/reports`  
   - diagnosis: `python -m scripts.diagnosis_decomposition --metrics <root>/reports/metrics.csv --regime-metrics <root>/reports/regime_metrics.csv --output-dir <root>/reports`  
   - reference_row.csv는 leaderboard 스크립트가 생성.
2) **후보 9~10개(PRL)**  
   - run_all → analyze(--run-index) → diagnosis 를 후보별 반복.  
   - output_root: `outputs/exp_runs/gate1/<EXP_NAME>/<TS>/`.

## 2-D) 산출물
- 각 output_root/reports: metrics.csv, regime_metrics.csv, run_index.json, analysis(Δ/CI), diagnosis_decomposition.md.
- Gate1 종료 시:
  - **Gate1_leaderboard.csv** (PRL 후보 vs reference baseline 비교치 포함)
  - **gate1_summary.md** (PASS/FAIL 사유, 상위 1~2개 선정 근거, “Gate1은 방향성 확인 단계” 문구)

## 2-E) 스코어링/판정 규칙
- 비교 기준은 항상 reference baseline_sac.
- PASS (둘 중 하나):
  - **T1**: avg_turnover <= 0.70 * baseline_ref_avg_turnover AND sharpe_net_exp >= baseline_ref_sharpe_net_exp
  - **T2**: sharpe_net_exp >= baseline_ref_sharpe_net_exp + 0.10
- FAIL 즉시 컷:
  - sharpe_net_exp <= baseline_ref_sharpe_net_exp - 0.05
  - avg_turnover >= 1.10 * baseline_ref_avg_turnover (반복될 경우)
- **Score** = sharpe_net_exp - 0.25 * |max_drawdown_net_exp| - 0.10 * avg_turnover  
  - mid Sharpe가 baseline보다 낮으면 -0.05 페널티.
- PASS 후보 중 Score 상위 1~2개를 Gate2로 승격.

## 2-F) 자동화 도구
- **Gate1 리더보드 생성**  
  - `python -m scripts.build_gate1_leaderboard --reference-run-index outputs/exp_runs/gate1/reference_baseline_sac/<TS>/reports/run_index.json --candidate-run-indexes "outputs/exp_runs/gate1/*/*/reports/run_index.json" --output-dir outputs/exp_runs/gate1`
  - 산출: `Gate1_leaderboard.csv`, `gate1_summary.md`, `reference_row.csv` (ref trace)
- (선택) run_gate1_sweep.sh를 추가해 baseline → 후보 실행/분석을 일괄 처리 가능.

## 2-G) 종료 체크리스트
- Reference baseline_sac 1회 실행 완료 (force_refresh=false).
- 후보 9~10개 실행 완료, 모두 output_root 분리 저장.
- analyze가 --run-index 기반으로 생성됨.
- Gate1_leaderboard.csv 확장 컬럼 포함, gate1_summary.md 작성됨.
- PASS 후보 중 상위 1~2개를 Gate2로 승격.
