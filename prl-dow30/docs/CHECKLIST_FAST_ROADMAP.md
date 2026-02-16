# FAST Roadmap Compliance Checklist

이 문서는 2주 로드맵 v1.0 요구사항(목표/지표/게이트/산출물)과 FAST PHASED SPEC v1.0 요구사항(출력 분리/필터/평가창/시간 옵션)이 모두 충족되는지 검증하는 체크리스트다. 아래 커맨드들은 재현 가능한 형태로 작성됐다.

## 1) output_root 분리 및 run_index 필터 동작
- 커맨드 (예):  
  `python -m scripts.run_all --config configs/exp/gate0_smoke_W1.yaml --output-root outputs/exp_runs/gate0_smoke/<TS>`
- 확인:
  - `<output_root>/reports/run_index.json` 생성 여부
  - 같은 repo에서 다른 `<output_root>`로 실행 시 산출물이 섞이지 않는지 확인
  - `python -m scripts.analyze_paper_results --metrics <root>/reports/metrics.csv --regime-metrics <root>/reports/regime_metrics.csv --run-index <root>/reports/run_index.json --output-dir <root>/reports` 실행 시 run_id 필터가 적용되는지 확인

## 2) W1/W2 평가창 확인
- 커맨드 (예):  
  `python -m scripts.run_all --config configs/exp/gate2_confirm_W1W2.yaml --output-root outputs/exp_runs/gate2_confirm/<TS>`
- 확인:
  - `metrics.csv`와 `regime_metrics.csv`에 `eval_window` 컬럼이 존재하고 W1/W2 모두 기록되는지 확인
  - `trace_*_W1.parquet`, `trace_*_W2.parquet` 등 윈도우별 산출물이 생성되는지 확인

## 3) 시간 단축 옵션 (no-trace / no-baselines / no-step4 / trace_stride)
- 커맨드 (예):  
  `python -m scripts.run_all --config configs/exp/gate1_screen_W1.yaml --output-root outputs/exp_runs/gate1_fast/<TS>`
- 설정:
  - `eval.write_trace=false` 또는 `trace_stride>=5`
  - `eval.run_baselines=false`
  - `eval.write_step4=false`
- 확인:
  - `metrics.csv`/`regime_metrics.csv`는 생성되고 net_exp 컬럼이 유지되는지
  - trace parquet이 없거나 stride에 따라 다운샘플됐는지
  - step4 리포트가 생성되지 않는지

## 4) diagnosis_decomposition 산출
- 커맨드 (예):  
  `python -m scripts.diagnosis_decomposition --metrics <root>/reports/metrics.csv --regime-metrics <root>/reports/regime_metrics.csv --trace <root>/reports/trace_<run_id>.parquet --output-dir <root>/reports`
- 확인:
  - `<root>/reports/diagnosis_decomposition.md`
  - `<root>/reports/turnover_distribution.csv`
  - `<root>/reports/regime_breakdown_net.csv`
  - 내용에 gross→net_exp 하락폭, turn-over 분포, mid regime 요약, Sharpe 개선 필요량이 포함되었는지

## 5) roadmap report 산출
- 커맨드 (예):  
  `python -m scripts.make_roadmap_report --run-index-paths outputs/exp_runs/*/reports/run_index.json --output-dir outputs/reports`
- 확인:
  - `outputs/reports/roadmap_results.md`
  - `outputs/reports/final_candidates_table.csv`
  - `outputs/reports/final_decision.md`
  - Gate별 결과가 net_exp 기준으로 정렬/요약되었는지, mid regime 개선 여부 언급이 있는지

## 6) 종합 PASS 조건
- 위 항목 1~5의 커맨드 실행 후 모든 산출물이 존재하고 net_exp 기준 지표가 생성되었음을 확인하면 PASS.
- 출력 경로, run_index 필터, 평가창(W1/W2), 시간 절약 옵션, 진단/로드맵 리포트가 모두 정상 동작해야 함.
