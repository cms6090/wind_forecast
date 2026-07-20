# HANDOFF — 마지막 갱신: 2026-07-20 (장소: 미기록)

## 현재 위치
- 로드맵 단계: 1~2, 4(기본 전처리) 완료(실행 완료) → 3. EDA(`02_eda.ipynb`) 시작 예정
- 작업 중 파일: `notebooks/01_preprocessing.ipynb` (38개 셀 작성 및 실행 완료), `reports/01_preprocessing.md` 작성 완료

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
- `notebooks/01_preprocessing.ipynb` 작성 및 실행 완료, `data/processed/*.parquet` 캐시 6개 생성, `reports/01_preprocessing.md` 작성 완료 (자세한 내용은 해당 리포트 참고)

## 다음 할 일 (우선순위순)
1. `notebooks/02_eda.ipynb` 작성 시작 (`eda-checklist` 스킬 순서대로) — 최우선 확인 항목:
   - 라벨 결측(group_1 104개/group_2 103개/group_3 8766개)이 특정 기간에 몰려있는지 (터빈 점검·고장 의심)
   - test LDAPS 752개 결측이 어느 `forecast_kst_dtm`에서 발생했는지 (발표분 전체 누락 여부 판정 → leakage-guard 기준으로 ffill 허용 여부 결정)
2. 커널 경로 확인: 01_preprocessing.ipynb 실행 시 `sys.executable`이 `wind-forecast_Competition\venv\Scripts\python.exe`로 나옴 — 이 프로젝트의 `wind_forecast (venv)` 커널이 아닌 다른 폴더의 venv로 실행된 것으로 보임. 다음 세션에서 Jupyter 커널 선택을 `wind_forecast (venv)`로 명시적으로 바꿔야 함
3. EDA 결과를 반영해 `03_features.ipynb`(피처 엔지니어링)로 진행

## 01_preprocessing.ipynb 산출물 (생성 완료)
- `data/processed/train_base_wide.parquet` (26304, 801), `train_base_agg.parquet` (26304, 136)
- `data/processed/test_base_wide.parquet` (8760, 799), `test_base_agg.parquet` (8760, 134)
- `data/processed/gfs_grid_meta.parquet`, `ldaps_grid_meta.parquet`
- wide = 격자별 컬럼 보존(`gfs_g1_...` 등), agg = 격자 평균/표준편차(`gfs_mean_...`, `gfs_std_...`) — 둘 중 무엇이 나은지는 아직 미정, 모델 비교로 결정 예정

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
- 라벨 결측(group_1/2/3) 및 test LDAPS 결측 752개의 원인·처리 방침 (EDA에서 확인 예정)
- 실행 커널이 다른 경로(`wind-forecast_Competition`) venv를 사용한 것으로 보이는 문제 원인 확인 필요

## 환경 메모
- venv Python 3.13.7, `requirements.txt` 설치 완료 (pandas 3.0.3, lightgbm 4.7.0, xgboost 3.3.0, catboost 1.2.10, scikit-learn 1.9.0, optuna 4.9.0 등)
- Jupyter 커널 등록 완료: `wind_forecast (venv)`
