# HANDOFF — 마지막 갱신: 2026-07-20 (장소: 미기록)

## 현재 위치
- 로드맵 단계: 1~2, 4(기본 전처리), 3(EDA) 완료(모두 실행 완료) → 6. 피처 엔지니어링(`03_features.ipynb`) 시작 예정
- 작업 중 파일: `notebooks/02_eda.ipynb` (82개 셀 작성 및 실행 완료), `reports/02_eda.md` 작성 완료

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
- `notebooks/01_preprocessing.ipynb` 작성 및 실행 완료, `data/processed/*.parquet` 캐시 6개 생성, `reports/01_preprocessing.md` 작성 완료
- `notebooks/02_eda.ipynb` 작성·실행 완료, `reports/02_eda.md` 작성 완료 — 핵심 발견은 아래 "02_eda 핵심 발견" 참고
- 커널 경로 문제(`wind-forecast_Competition`) 민석님이 직접 수정 완료

## 다음 할 일 (우선순위순)
1. `notebooks/03_features.ipynb` 작성 시작 (`wind-domain-features`/`feature-selection`/`leakage-guard` 스킬 기준) — `reports/02_eda.md`의 "So-what" 9개 항목을 피처 설계 근거로 사용
2. test LDAPS 부분 결측 3개 시각의 정확한 `forecast_kst_dtm` 특정 후 결측 처리 방침 확정 (같은 발표분 내 다른 격자 보간 vs 인근 발표분 ffill, leakage-guard 판정 필요)
3. LDAPS 상대습도 100% 초과값 클리핑 적용 여부 결정

## 02_eda 핵심 발견 (자세한 내용은 reports/02_eda.md)
- **그룹-터빈 매핑 최종 확증**: SCADA 합계 vs 라벨 상관 = 0.9998/0.9998/0.9966
- **LDAPS가 GFS보다 발전량과 상관이 높음**(10m 기준 0.67~0.77 vs GFS 전 높이 0.61~0.63) — 허브높이 매칭보다 지형 해상도가 더 중요. 03_features에서 LDAPS 우선 고려
- **GFS가 실제 풍속을 평균 2.8~3.2 m/s 과소예측**(SCADA 대비, RMSE 3.2~4.3 m/s) — GFS 파생 피처는 편향 감안 필요
- **풍향 영향이 매우 큼**: WSW/W 방향이 E/SE 대비 최대 20배 발전량 (겨울 계절풍과 일치)
- **계절×풍속 상호작용이 비선형**: 저·중풍속은 겨울이 유리(공기밀도), 고풍속(20~30m/s)은 겨울이 오히려 최저(cut-out 정지 추정)
- **라벨 결측 패턴**: 매년 7월 반복 소규모 결측 + 2022-10 대량 결측(82개, 터빈 점검 의심)
- **`data_available_kst_dtm` 규칙**: 3년 전체 예외 없이 지켜짐 (검증 완료)
- **test LDAPS 결측 752개**: 발표분 전체 누락 아님(0건), 특정 3개 시각의 격자 일부만 부분 결측
- **VESTAS SCADA 센서 오류 868셀**: 물리범위 기준 NaN 처리 완료 (통신두절 후 보정 스파이크로 추정)
- **UNISON 출력제한(curtailment) 확인**: 100 단위 계단(0~700, 드물게 800~1000), 인위적 패턴 통계로 확증
- **`power_kw10m`은 "출력(kW)"이 아니라 "10분 에너지(kWh)"** — 단위 오해로 인한 4-4번 계산 버그 발견 및 수정

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
- **그룹 대표 격자는 "그룹 평균 좌표 → 최근접 격자 1개"가 아니라 "터빈별 최근접 격자 비율로 가중평균"(`GROUP_GRID_WEIGHTS`) 방식 채택** — LDAPS에서 group_2가 격자 두 개로 정확히 반반 갈리는 걸 확인했기 때문. GFS는 전 그룹이 격자 하나로 수렴해 영향 없음
- **SCADA `power_kw10m`은 10분 에너지(kWh)로 확정** — 라벨 대조로 검증(비율 0.98~1.0). "출력(kW)"으로 오해해 10/60을 곱하면 6배 축소되는 버그가 생김

## 실험 기록
| 날짜 | 실험 | 로컬 Score | 리더보드 | 결론 |
|---|---|---|---|---|

## 미해결 질문
- group_3 데이터 부족(2023년 1년치만 학습) 문제를 모델링 단계에서 어떻게 다룰지
- test LDAPS 부분 결측 3개 시각을 어떻게 처리할지 (같은 발표분 내 다른 격자 보간 vs 인근 발표분 ffill)
- 2022-10 라벨 결측(group_1/2 82개)의 정확한 원인 (터빈 점검 기록 등 외부 확인 불가하므로 결측 그대로 두고 학습 제외하는 것으로 잠정 결론)

## 환경 메모
- venv Python 3.13.7, `requirements.txt` 설치 완료 (pandas 3.0.3, lightgbm 4.7.0, xgboost 3.3.0, catboost 1.2.10, scikit-learn 1.9.0, optuna 4.9.0 등)
- Jupyter 커널 등록 완료: `wind_forecast (venv)` (커널 경로 문제 해결됨, 확인 완료)
