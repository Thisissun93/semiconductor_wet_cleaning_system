# 업무 시나리오 측정

`python scripts/benchmark_scenario.py`로 다시 만들 수 있습니다. DB 사본에서 규칙 판정 → 원인 분석 → 조치 추천을 5회 실행한 중앙값입니다.

## 처리량과 시간

| 항목 | 값 |
|---|---|
| 대상 | LOT 120개 × 판정 항목 17개 = 2,040건 |
| 판정 결과 | CRITICAL 39, WARNING 55, WATCH 72 |
| WARNING 이상 LOT | 35개 |
| 원인 후보 / 조치 추천 | 106건 / 216건 |
| 규칙 판정 | 26 ms |
| 원인 분석 | 120 ms |
| 조치 추천 | 11 ms |
| **합계** | **152 ms** |

시간은 실행 환경에 따라 달라집니다. 수작업 소요 시간과의 비교는 측정하지 않았습니다.

## 사례: LOT-202607-0114 (WC-03)

CRITICAL 항목이 가장 많은 LOT입니다.

| 항목 | 등급 | 측정값 |
|---|---|---|
| bath_age_hours | CRITICAL | 20.375 |
| concentration_deviation_percent | CRITICAL | 13.928 |
| filter_differential_pressure | CRITICAL | 2.064 |
| flow_rate | CRITICAL | 15.31 |
| particle_count | CRITICAL | 29.0 |
| nozzle_pressure | WARNING | 1.405 |
| pump_pressure | WARNING | 2.411 |
| vibration | WARNING | 1.261 |
| metal_contamination_ppb | WATCH | 0.598 |
| motor_current | WATCH | 7.777 |
| organic_residue_count | WATCH | 3.0 |
| yield_percent | WATCH | 94.843 |

| 순위 | 원인 후보 | 기여도(%) | 신뢰도 |
|---|---|---|---|
| 1 | Filter 막힘 및 교체주기 초과 | 41.4 | HIGH |
| 2 | Pump 및 Motor 성능 저하 | 30.3 | HIGH |
| 3 | 약액 Bath 노화 및 교체주기 초과 | 28.4 | HIGH |

| 순위 | 조치 | 우선순위 | 담당 |
|---|---|---|---|
| 1 | Wet Cleaning Filter 즉시 교체 | URGENT | 설비기술 |
| 2 | Filter Housing 및 배관 오염 점검 | URGENT | 설비기술 |
| 3 | Pump 및 Motor 상태 즉시 점검 | URGENT | 설비기술 |

원인 기여도는 규칙 점수의 상대 비율입니다. 통계적 인과 추정이 아니므로 현장 확인 전 우선순위로만 씁니다.
