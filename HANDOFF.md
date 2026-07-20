# HANDOFF — 마지막 갱신: 2026-07-20 (장소: 미기록)

## 현재 위치
- 로드맵 단계: 1. 문제와 평가 지표 확인 (완료), 2. 학습·검증 데이터 분리 설계 (완료) → 3~4. EDA / 기본 전처리 시작 예정
- 작업 중 파일: 없음 (아직 노트북 시작 전)

## 지난 세션에서 한 것
- 데이터 파일 구조 확인 (`data/data_description.md`, `info.xlsx`)
- `info.xlsx` info 시트로 KPX 그룹-터빈 매핑 확인:
  - kpx_group_1 = VESTAS wtg01-06 (V126, 3.6MW×6=21.6MW)
  - kpx_group_2 = VESTAS wtg07-12 (V126, 3.6MW×6=21.6MW)
  - kpx_group_3 = UNISON wtg01-05 (U136, 4.2MW×5=21MW)
- 대회 공식 채점 산식(`src/metric.py`)을 민석님이 대회 제공 원문 그대로 전달 → 그대로 저장 (수정 금지)
- `src/submission.py`의 `validate_submission()` 작성: 컬럼/행수(8760)/forecast_id·시각 불변/결측·음수·설비용량 초과 검사
- `requirements.txt` 작성 및 venv에 설치 완료, Jupyter 커널 `wind_forecast (venv)` 등록 완료
- train/validation 분리안 확정 (기본안 채택, 아래 "결정 사항" 참고)

## 다음 할 일 (우선순위순)
1. `notebooks/01_preprocessing.ipynb` 작성 시작: CSV 로드(utf-8-sig) → datetime 변환 → GFS/LDAPS long→wide 구조 변환 → labels 병합 → `data/processed/train_base.parquet`, `test_base.parquet` 캐시 생성 (train/test 동일 함수 적용, `preprocessing`·`leakage-guard` 스킬 기준)
2. 이어서 `notebooks/02_eda.ipynb` (`eda-checklist` 스킬 체크리스트 순서대로)
3. EDA 결과를 반영해 `03_features.ipynb`(피처 엔지니어링)로 진행

## 결정 사항 / 근거 (누적)
- `src/metric.py`는 민석님이 대회 사이트에서 받은 원문 그대로 사용, 수정 금지
- NMAE는 그룹별 설비용량으로 정규화한 오차율의 평균이며, 3개 그룹 NMAE를 단순평균 (행 개수 가중 아님)
- FICR은 그룹별 (실제 정산 단가 합) / (최대 단가 4원 기준 합)의 3개 그룹 단순평균
- 채점 시 valid 조건: 실제 발전량 ≥ 그룹 설비용량의 10% (실제값 기준이므로 로컬 검증에서만 적용 가능, 예측 시점엔 전 구간 성실 예측 필요)
- **train/validation 분리 (기본안 채택)**: train = 2022-01-01 01:00 ~ 2024-01-01 00:00, validation = 2024-01-01 01:00 ~ 2025-01-01 00:00 (test와 동일하게 "1/1 01:00 ~ 익년 1/1 00:00" 1년 구조로 리더보드 상관 확보)
  - 리스크: group_3은 라벨이 2023년부터만 있어 이 분리에서는 **2023년 한 해만** 학습에 사용됨 → 모델 선택 단계에서 데이터 부족 문제로 재검토 가능성 있음 (예: group_3 검증 구간을 더 짧게 갖거나, group_1/2와 정보 공유하는 모델 구조 고려)
  - 노트북 파일 순서는 로드맵 문서 순서(EDA→전처리)와 달리 `01_preprocessing.ipynb`(구조 변환: long→wide, 병합) → `02_eda.ipynb`(본격 탐색) 순으로 진행 — EDA가 wide 병합 테이블을 필요로 하기 때문 (`preprocessing`/`eda-checklist` 스킬 기준)

## 실험 기록
| 날짜 | 실험 | 로컬 Score | 리더보드 | 결론 |
|---|---|---|---|---|

## 미해결 질문
- group_3 데이터 부족(2023년 1년치만 학습) 문제를 모델링 단계에서 어떻게 다룰지

## 환경 메모
- venv Python 3.13.7, `requirements.txt` 설치 완료 (pandas 3.0.3, lightgbm 4.7.0, xgboost 3.3.0, catboost 1.2.10, scikit-learn 1.9.0, optuna 4.9.0 등)
- Jupyter 커널 등록 완료: `wind_forecast (venv)`
