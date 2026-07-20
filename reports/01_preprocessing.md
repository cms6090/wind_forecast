# 01. 전처리 — 원본 데이터를 wide 테이블로 변환

## Why

- GFS(9격자)/LDAPS(16격자) 예보 데이터는 `forecast_kst_dtm` 하나당 격자 수만큼 행이 쌓인 long 구조라, 모델이 바로 쓸 수 없다.
- EDA와 피처 엔지니어링을 시작하려면 "한 시각 = 한 행"인 wide 테이블과, train/test에 동일하게 적용 가능한 재현 가능한 캐시(parquet)가 먼저 필요하다.
- 이 단계에서는 결측 보정·이상치 제거·스케일링을 하지 않는다 (근거 없이 임의로 처리하지 않기 위해, `leakage-guard`/`preprocessing` 스킬 규칙).

## How

1. **로드**: 모든 CSV를 `utf-8-sig`로 읽고 `forecast_kst_dtm`/`data_available_kst_dtm`을 datetime으로 변환 (데이터 설명서 5~8절 — BOM 명시).
2. **행 수·기간 검증**: `data_description.md`에 적힌 값(gfs_train 236,736행, ldaps_train 420,864행, labels 26,304행 등)과 실제 로드 결과를 assert로 대조.
3. **격자 메타 분리**: `grid_id`별 위도/경도가 시간에 따라 불변임을 확인(assert)한 뒤 별도 표로 저장 — 매 시각 반복 저장을 피하고, 추후 지도 시각화에 사용.
4. **train/test 격자 구성 동일성 확인**: 격자 배치(위도/경도 집합)가 train과 test에서 완전히 같음을 확인 — 다르면 wide 변환 후 컬럼명이 어긋나 추론이 불가능해지므로 사전 검증.
5. **long→wide 변환(격자별 컬럼 보존)**: `pivot`으로 시각 하나당 한 행이 되도록 바꾸고, 컬럼명은 `{source}_g{grid_id}_{변수}` 규칙 사용. GFS(변수 35개×격자 9개=315+2)=317컬럼, LDAPS(변수 30개×격자 16개=480+2)=482컬럼.
6. **long→wide 변환(격자 집계)**: 같은 시각의 격자들을 평균·표준편차로 집계한 대안 버전도 함께 생성(표준편차는 격자 간 풍속 차이=지역별 바람 편차 정보). GFS agg 72컬럼, LDAPS agg 62컬럼.
7. **기준축 병합**: train은 `train_labels.csv`, test는 `sample_submission.csv`를 기준 축으로 두고 기상 데이터를 left join — 기상 데이터 결측이 라벨 손실로 이어지지 않게 함.
8. **병합 후 결측치 확인**: 채우지 않고 개수만 집계해 다음 단계(EDA)에 근거로 넘김.
9. **물리 범위 간단 점검**: 기온이 켈빈 단위인지, 상대습도가 0~100 범위인지 확인.
10. **train/test 컬럼 구성 일치 검증**: 라벨 컬럼(train)·`forecast_id`(test)를 제외한 나머지 컬럼 집합이 완전히 같은지 assert.
11. **parquet 캐시 저장**: `data/processed/`에 wide/agg 버전과 격자 메타를 각각 저장, 재로드로 dtype 보존 확인.

## Result

- 행 수·기간 검증: 전부 통과 (명세서와 일치).
- 격자 구성: GFS 9개, LDAPS 16개, train/test 완전 동일.
- 병합 후 shape: `train_base_wide` (26304, 801), `train_base_agg` (26304, 136), `test_base_wide` (8760, 799), `test_base_agg` (8760, 134). train/test 컬럼 수 차이(2개)는 기준축 컬럼 차이(라벨 3개 − `forecast_id` 1개)로 설명되며, 기상 컬럼 자체는 컬럼 구성 일치 검증(10단계)을 통과함.
- **결측치 (train, 라벨 컬럼)**: `kpx_group_3` 8,766개, `kpx_group_1` 104개, `kpx_group_2` 103개. 기상 컬럼 결측은 0개.
  - 데이터 설명서 대비 예상 밖 결과: group_1/2는 "2022~2024 전체 제공"이라 했으나 실결측이 존재하고, group_3도 "2022년만 공백(8760시간)"이라 했으나 8,766개로 6개 더 많음.
- **결측치 (test, 기상 컬럼)**: 총 752개 셀, 여러 LDAPS 격자의 `surface_0_lsm` 등에 격자당 3개씩 분산 — 특정 발표분(예보 누락일) 존재 가능성.
- **물리 범위**: GFS 기온(격자1) 251.2~305.8K(약 -22~33°C), LDAPS 기온(격자1) 249.7~302.0K, GFS 상대습도(격자1) 11.4~99.9% — 모두 켈빈·퍼센트 범위로 타당.
- 컬럼 구성 일치 검증(wide/agg 모두) 통과.
- parquet 캐시 6개 파일 저장 및 재로드 shape 일치 확인 완료.

## So-what

- **다음 EDA(`02_eda.ipynb`) 최우선 확인 항목**: (1) group_1(104)/group_2(103)/group_3(8766) 라벨 결측이 특정 기간에 몰려 있는지(터빈 점검·고장 의심 구간) 시계열로 확인. (2) test LDAPS 752개 결측이 발생한 정확한 `forecast_kst_dtm`을 특정해, "발표분 전체 누락"인지 "격자 일부만 누락"인지 판정 — 전자라면 `leakage-guard` 기준으로 직전 발표분 ffill 허용 여부를 판단해야 함.
- 라벨 결측은 보간하지 않고 학습에서 제외하는 것이 원칙(`preprocessing`/`leakage-guard` 스킬) — EDA에서 결측 분포를 본 뒤 최종 처리 방침을 확정한다.
- wide(격자별) vs agg(집계) 버전 중 어느 쪽을 주력으로 쓸지는 아직 미정 — `model-selection` 단계에서 실제 비교로 결정.
- 노트북 실행 시 커널이 이 프로젝트의 `wind_forecast (venv)`가 아닌 다른 경로(`wind-forecast_Competition\venv`)의 파이썬으로 실행된 흔적이 있음 — 다음 세션에서 커널 선택을 재확인 필요 (데이터 경로 자체는 정상 확인됨).
