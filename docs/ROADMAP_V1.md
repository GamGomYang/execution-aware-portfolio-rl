[Sharpe ≥ 2.0 (Net 기준) 달성을 위한 2주 실험 로드맵 & 구현/검증 명세 v1.0]
(전제: 현재 파이프라인은 gross/net 지표가 모두 생성되며, net_exp가 “정의 일치” 기본 지표)

────────────────────────────────────────────────────────
0) 목표/성공 기준(가장 중요)
0-1) 최종 목표(Primary Objective)
- OOS 테스트 구간(예: 2022-02-15 ~ 2025-12-30)에서
  “연율화 Sharpe_net_exp ≥ 2.0” 달성

0-2) 필수 동시 조건(안전장치)
- max_drawdown_net_exp ≥ -0.20 (또는 목표 MDD 한계 지정)
- avg_turnover가 현저히 폭증하지 않을 것 (예: baseline 대비 +50% 이내)
- 성능이 seed에 대해 안정적일 것:
  - seeds ≥ 10에서 mean Sharpe_net_exp ≥ 2.0
  - 95% CI(bootstrap)가 2.0 아래로 크게 내려가지 않을 것(예: 하한 1.8 이상 목표)

0-3) 현실 체크(중요 경고)
- Dow30 롱온리 + 비용 + 일별 리밸런싱만으로 Sharpe 2.0은 매우 높은 목표일 수 있음.
- 따라서 2주 로드맵은 “달성 가능성 극대화”를 위해
  (A) turnover 구조 제어 + (B) mid-regime 약점 개선 + (C) risk 목표 정렬
  + 필요 시 (D) 환경 확장(현금/헤지/리밸런스 빈도)까지 포함한다.

────────────────────────────────────────────────────────
1) STEP 0: “샤프 계산 정의” 고정(실험 전 1일 내 반드시 완료)
(샤프 2.0 목표가 의미 있으려면 계산 정의가 고정돼야 함)

1-1) Sharpe 계산식(고정)
- returns: net_return_exp (기본), 대조로 gross_return도 유지
- Sharpe_net_exp = sqrt(252) * mean(daily_net_return_exp) / std(daily_net_return_exp)
- risk-free는 0으로 고정(또는 명시적으로 r_f 사용 시 동일하게 적용)
- 결측/0분산 처리 규칙 명시

1-2) 검증(필수)
- metrics.csv의 sharpe_net_exp가 trace로 재계산한 값과 일치(T3 테스트 통과)
- “연율화가 적용됨”을 보고서에 문장으로 명시

산출물:
- outputs/reports/sharpe_definition.md (1페이지)
- audit_issues.json에 “Sharpe 정의 고정” 기록

────────────────────────────────────────────────────────
2) STEP 1: 원인 분해(현재 결과 기준) — “왜 0.8 근처인가”
(실험 방향을 낭비하지 않기 위해, 원인을 수치로 쪼개고 시작)

2-1) 핵심 분해 리포트(자동 생성)
- baseline_sac vs prl_sac에 대해:
  - gross 대비 net_exp 하락폭(= 비용 영향)
  - avg_turnover, turnover 분포(quantiles)
  - mid regime에서 Δ가 음수인 이유 추정:
    - (a) turnover 급증 여부
    - (b) return 분산 증가 여부
    - (c) 손실일(negative days) 비율 변화
- “Sharpe_net_exp를 2.0으로 만들려면”
  mean/std 관점에서 필요한 개선폭(대략)을 계산:
  - 현재 mean/std 수준에서 std를 얼마나 낮추거나 mean을 얼마나 올려야 하는지 숫자로 제시

산출물:
- outputs/reports/diagnosis_decomposition.md
- outputs/reports/turnover_distribution.csv
- outputs/reports/regime_breakdown_net.csv

────────────────────────────────────────────────────────
3) STEP 2: 개선 후보(레버리지 큰 순서대로) — “무조건 net을 올리는 구조”
(현재 패턴: PRL은 turnover↑ → net에서 이점 상쇄. 따라서 먼저 turnover 구조 제어가 1순위)

[개선 후보군 A: Turnover 구조 제어(최우선)]
A1) Action Smoothing (포트폴리오 관성)
- 구현: 실제 적용 weights w_t를
  w_t = (1-α) * w_{t-1} + α * w*_t
- α sweep: [0.05, 0.10, 0.20]
- 기대 효과: turnover 감소 → net 성과 회복, mid regime 안정화

A2) Δw Penalty (행동 변화 직접 패널티)
- reward = base_reward - λ * ||w_t - w_{t-1}||_1
- λ sweep: [0.1, 0.3, 1.0]  (스케일은 실제 turnover 수준 보고 조정)
- 기대 효과: “불필요한 잦은 리밸런스” 억제

A3) Rebalance Frequency Control (리밸런싱 빈도)
- 매일이 아니라 k-day 리밸런싱:
  - k ∈ [2, 5, 10] (격일/주간/2주)
- 비리밸런스 날은 w_t = w_{t-1} 유지(또는 소폭 drift만 반영)
- 기대 효과: mid regime에서 노이즈 트레이딩 감소

[개선 후보군 B: Mid-regime 약점 타겟(두 번째)]
B1) Regime-aware Plasticity Schedule
- plasticity 강도(학습률/적응률/모듈 게이트)를 regime에 따라 조절:
  - mid에서는 plasticity↓ (보수적)
  - low/high에서는 plasticity↑ 또는 유지
- 구현 형태:
  - (1) 상태에 regime one-hot(또는 vz z-score)를 입력
  - (2) PRL 내부 plasticity 계수에 regime-dependent multiplier 적용
- 기대 효과: mid 구간에서 과적응/불필요한 포지션 전환 줄이기

B2) Mid-Regime Cost Amplification (선택)
- mid 구간에서만 cost multiplier를 키워 turnover를 더 강하게 억제
- 예: cost_t = c_tc * turnover_t * m(regime)
  - m(mid)=1.5~2.0, m(low/high)=1.0
- 기대 효과: mid 취약을 직접 억제(단, 설계 정당화 필요)

[개선 후보군 C: “Sharpe 목표 정렬” 보상/리스크 페널티(세 번째)]
C1) Variance / Downside penalty
- reward = log(1+r) - cost - β * (rolling_vol)^2  또는 downside_vol
- β sweep: [0.1, 0.3, 1.0]
- 기대 효과: std 감소 → Sharpe 상승 방향

C2) CVaR penalty (리스크 꼬리 억제)
- reward = base - γ * CVaR(returns_window)
- γ sweep: [0.1, 0.3, 1.0]
- 기대 효과: tail risk 감소, MDD 감소

[개선 후보군 D: Sharpe 2 달성 가능성을 올리는 “환경 확장”(필요 시)]
D1) Cash weight 허용(현금 비중)
- 행동공간에 “cash asset” 추가(0 return 가정 or mmf proxy)
- 목적: 변동 구간에서 risk-off로 샤프 상승

D2) Hedge asset 추가(가능하면)
- 예: IEF/TLT/GLD 등 1~3개 추가(크로스에셋)
- 목적: 분산 감소로 Sharpe 상승

※ D는 “Dow30만으로 Sharpe 2”가 비현실적일 때, 목표 달성 확률을 크게 올리는 옵션.

────────────────────────────────────────────────────────
4) STEP 3: 실험 설계(2주 안에 “확실한 결론”을 내는 매트릭스)
(완전한 full factorial은 폭발하므로, Gate 기반(단계 통과형)으로 설계)

[공통 실험 파라미터]
- seeds: 10개 권장(최소 5개, 최종 10개)
- timesteps:
  - Gate1 빠른 스크리닝: 100k
  - Gate2 확증: 250k
  - Gate3 최종: 500k (가능하면)
- cost: c_tc=0.0005 기본 + 비용 스윕(0.0, 0.00025, 0.0005, 0.0010)

[Gate 0: 현 상태 베이스라인 확정(1일)]
- baseline_sac, prl_sac (현재 설정) + 전략 baseline 3종
- net_exp 기준 테이블/그래프 생성(이미 있음) + seed 10 확장

[Gate 1: Turnover 제어 1차 스크리닝(2~3일)]
- 후보: A1, A2, A3만 먼저
- 조합 폭발 방지 규칙:
  - A1 단독: α ∈ {0.05, 0.10, 0.20} (3개)
  - A2 단독: λ ∈ {0.1, 0.3, 1.0} (3개)
  - A3 단독: k ∈ {2,5,10} (3개)
  - + (선택) A1+A2 “최소 조합” 2개만(예: α=0.1 고정 후 λ 2개)
- 총 9~11개 설정 × seeds=5 × 100k
- 통과 기준:
  - mean Sharpe_net_exp가 baseline 대비 +0.2 이상 개선 OR
  - mean turnover가 baseline 대비 -30% 이상 감소 AND net 성과 악화 없음

[Gate 2: Mid-regime 타겟 개선(3~4일)]
- Gate1에서 상위 2개 설정을 가져와 B1/B2를 얹어본다.
- B1: regime input + mid에서 plasticity multiplier {0.5, 0.7}
- B2: mid cost multiplier {1.5, 2.0} (선택)
- 평가 지표(필수):
  - mid regime의 Δsharpe_net_exp가 0 이상으로 개선되는지
  - mid regime의 turnover가 줄어드는지
- 통과 기준:
  - mid Δsharpe_net_exp의 bootstrap CI 하한이 -0.005 이상(거의 0에 근접)
  - 또는 mid Δcumret_net_exp가 명확히 개선(음수→0 이상)

[Gate 3: Risk 정렬 보상(C 후보)로 “샤프를 끌어올리기”(3~4일)]
- Gate2 상위 1~2개 설정에 대해 C1 또는 C2 추가
- 목표는 mean 증가보다 “std 감소”에 초점(샤프 상승의 핵심)
- 통과 기준:
  - Sharpe_net_exp가 의미 있게 상승(+0.2 이상) AND
  - MDD_net_exp가 악화되지 않을 것

[Gate 4(옵션): 환경 확장(D 후보)(2~3일)]
- Gate3까지도 Sharpe가 2.0에 한참 못 미치면, D를 진행
- D1 cash asset 추가부터(구현 난이도 대비 효과 큼)
- D2 hedge asset은 데이터/구간 정합성 확보 후 진행
- 통과 기준:
  - Sharpe_net_exp가 +0.4 이상 개선(환경 확장은 효과가 커야 의미 있음)

[최종 확증 런(마지막 2~3일)]
- 최종 후보 1~2개를 선정하여:
  - seeds=10
  - timesteps=250k 또는 500k
  - 비용 스윕(최소 2개: 0.0005, 0.0010)
- 최종 보고서에 “gross + net_exp” 모두 제시하되, 결론은 net_exp 기반으로 쓴다.

────────────────────────────────────────────────────────
5) 구현 명세(설정 파일/실험 관리 규칙)
5-1) config naming 규칙(강제)
- configs/exp/
  - exp_A1_smooth_a010.yaml
  - exp_A2_dwp_l030.yaml
  - exp_A3_rebal_k5.yaml
  - exp_A1A2_a010_l030.yaml
  - exp_A1_midplast_m05.yaml
  - exp_A1_midcost_m20.yaml
  - exp_A1_midplast_m05_riskbeta_03.yaml
  - exp_cash_A1_midplast.yaml

5-2) output 분리 규칙(강제)
- outputs/exp_runs/<EXP_NAME>/<timestamp>/
  - reports/, traces/, models/, logs/
- run_index.json에 해당 실험의 run_id 목록 저장
- analyze_paper_results는 반드시 “이 run_index.json에 포함된 run_id만” 대상으로 분석

5-3) 자동 리포트 생성 규칙
- run_all 실행 후 자동으로:
  - summary_seed_stats.csv (gross/net 둘 다)
  - paired_seed_diffs.csv (gross/net 둘 다)
  - regime_paired_diffs.csv (gross/net 둘 다)
  - step4_report(최종 후보에 대해서만)

────────────────────────────────────────────────────────
6) 평가/리포트 지표(강제)
(샤프만 보면 위험. 반드시 아래를 함께 본다)

[공통]
- sharpe_gross, sharpe_net_exp, sharpe_net_lin
- cumulative_return_gross, cumulative_return_net_exp, cumulative_return_net_lin
- max_drawdown_gross, max_drawdown_net_exp, max_drawdown_net_lin
- avg_turnover, total_turnover, turnover_quantiles(50/75/90/99%)
- cost_sum, cost_mean (가능하면)
- “mid-regime 전용”:
  - sharpe_net_exp_mid, cumret_net_exp_mid, turnover_mid

[최종 선택 기준(스코어링 추천)]
- Score = Sharpe_net_exp
         - 0.25 * |MDD_net_exp|
         - 0.10 * avg_turnover
- (그리고 mid regime Sharpe_net_exp가 baseline보다 낮으면 페널티)

────────────────────────────────────────────────────────
7) 2주 실행 일정(현실적인 플랜)
Day 1:
- Sharpe 정의 고정 + decomposition 리포트 생성 + seeds=10 baseline/prl 재평가

Day 2~4 (Gate1):
- A1/A2/A3 스크리닝(100k, seeds=5)
- 상위 2개 후보 선정

Day 5~7 (Gate2):
- B1/B2 mid 타겟 개선(100k~250k, seeds=5)
- mid 음수 CI 문제 완화 확인

Day 8~10 (Gate3):
- C1/C2 risk 정렬(250k, seeds=5)
- 샤프 상승이 “std 감소”로 오는지 확인

Day 11~12 (Gate4 옵션):
- 여전히 샤프가 낮으면 cash/hedge 확장 실험
- 또는 비용 스윕으로 “임계점” 정리(논문에 강력한 그림)

Day 13~14 (최종 확증):
- 최종 후보 1~2개
- seeds=10, 250k~500k
- 비용 0.0005/0.0010에서 재확인
- 최종 report 생성

────────────────────────────────────────────────────────
8) 산출물(최종 제출용)
- outputs/reports/roadmap_results.md
  - 각 Gate별 상위 후보, 탈락 사유, 핵심 수치, 그래프
- outputs/reports/final_candidates_table.csv
  - 후보별 Sharpe_net_exp, MDD_net_exp, turnover, mid-regime 성과 정리
- outputs/reports/final_decision.md
  - “왜 이 후보를 선택했는지” 근거(정량 + 안정성)

────────────────────────────────────────────────────────
9) 실패/리스크 관리(명시)
- 만약 Gate3까지도 Sharpe_net_exp가 1.2~1.4 수준에서 정체:
  - Dow30 롱온리 한계 가능성이 크므로 D1(cash) 또는 D2(hedge)로 확장
  - 또는 목표를 “Sharpe 2.0 (gross)”가 아니라 “Sharpe_net_exp 1.5+ 안정”로 현실 조정(논문형)
- seeds=3은 결론 불가능 → 반드시 최소 5, 최종 10

끝.
