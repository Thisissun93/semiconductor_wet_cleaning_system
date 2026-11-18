# AGENTS.md

반도체 Wet Cleaning 공정의 MES 형식 데이터(SQLite)로 규칙 기반 이상 판정, 원인 후보, 조치 추천, SPC, 예지보전(Random Forest + SHAP), 평가 설계를 보여 주는 Streamlit 앱입니다. 데이터는 포트폴리오용 생성 데이터입니다.

## 구조

- `config/process_limits.yaml`: 항목별 판정 한계 (NORMAL / WATCH / WARNING / CRITICAL)
- `src/detection/rule_detector.py` → `src/root_cause/rule_based_rca.py` → `src/recommendation/action_engine.py`: 판정 → 원인 → 조치. 결과는 `database/wet_cleaning.db`에 저장
- `src/analysis/spc_analysis.py`: I-MR 관리한계, Cp/Cpk(군내 σ), Pp/Ppk(전체 σ)
- `src/predictive_maintenance/`: 특징량, 학습, 예측, SHAP
- `src/analysis/validation.py`, `scripts/validate_stats.py`: 독립 계산 대조
- `scripts/benchmark_scenario.py`: 업무 시나리오 측정

## 확인 명령

```bash
python -m pytest tests -q
python scripts/validate_stats.py   # docs/stats_validation.md가 바뀌면 커밋 필요
```

## Review guidelines

- 판정 등급은 정상에서 멀어질수록 같거나 높아져야 합니다. 경계값(≤, <) 처리와 `process_limits.yaml` 방향(lower/higher/range) 해석을 확인하세요.
- 관리한계(공정 변동, MR̄/1.128)와 규격한계(LSL/USL)를 섞지 않아야 합니다. Cp/Cpk는 군내 σ, Pp/Ppk는 전체 σ입니다.
- I-MR 계산 입력은 시간 순서여야 합니다. 정렬이 빠지면 지적하세요.
- 예지보전: 학습·평가 데이터 누출(겹치는 윈도, 무작위 분할), 라벨 절단(이후 LOT 10개 미만), 다수 클래스 기준선 대비 성능을 확인하세요. 기준선보다 낫다는 근거 없이 성능을 주장하는 문구는 지적하세요.
- 화면, DB 저장 결과, PDF 보고서가 같은 계산 결과를 써야 합니다.
- SQL은 매개변수 바인딩을 쓰고, 사용자 입력을 문자열로 이어 붙이지 않습니다.
- 계산 로직을 바꾸면 테스트와 `docs/stats_validation.md`를 함께 갱신해야 합니다.

- README 맨 위의 영어 요약(English summary)에 있는 수치는 한국어 본문·docs와 같아야 합니다. 수치를 바꾸는 PR에서 영어 요약이 갱신되지 않았으면 지적하세요.
