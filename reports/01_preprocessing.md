# 01. 전처리 — 원본 데이터 로딩 → pivot → 조인 → 캐시 저장

## Why

원본 CSV는 그대로 모델에 넣을 수 없다. 이유는 두 가지다.

1. **기상 예보(LDAPS, GFS)가 긴 표(long format)다.** 한 시각(`forecast_kst_dtm`)에 대해 여러 격자(grid)가 각각 한 행씩 차지한다(LDAPS 16격자, GFS 9격자). 모델은 "시간 하나 = 행 하나"를 원하므로 격자를 컬럼으로 펼치는 pivot이 필요하다.
2. **기상 데이터와 발전량 라벨이 서로 다른 파일에 있다.** 시각을 기준으로 조인해야 "이 시각 날씨 → 이 시각 발전량"이 한 행에 있는 학습용 표가 만들어진다.

추가로 SCADA(10분 단위 터빈 실측) 데이터의 단위(`power_kw10m`)를 라벨과 대조해 확정하는 작업도 이 단계에서 함께 했다. 이름만으로는 "평균 kW인지 10분 누적 에너지(kWh)인지"가 애매했기 때문에, 실제 라벨과 코드로 크로스체크해서 확정하기로 했다.

## How

1. **경로 구조 확인**: `data/train/`, `data/test/`, `data/processed/` 하위 폴더 구조를 실제 파일로 확인하고 상수로 고정했다.
2. **LDAPS pivot 함수 작성(`pivot_weather_wide`)**: `forecast_kst_dtm`을 행으로, `grid_id`별 변수를 `{prefix}_g{격자번호}_{변수명}` 컬럼으로 펼쳤다. LDAPS/GFS 양쪽에서 재사용하도록 `prefix` 인자로 구분했다(변수명이 겹치는 경우 컬럼 충돌을 막기 위함). 결과: 420,864행(26,304시각×16격자, long) → 26,304행×482열(wide).
3. **LDAPS test에 같은 함수 적용 — 실행 중 발견한 결측**: 결측이 없어야 한다는 assert가 실패했다. 원인은 `ldaps_test.csv` 원본 자체에 정확히 3개 시각(`2025-04-08 17:00`, `2025-06-18 18:00`, `2025-07-18 06:00`)에서 16개 격자 전체가 비어 있었던 것(총 752개 값). 이 시각도 결국 제출해야 하므로 행을 버릴 수 없어, `pivot_weather_wide` 안에 "시간순 정렬 후 선형보간" 처리를 추가했다. 결측이 없는 경우(train, GFS test)는 이 처리가 아무 일도 하지 않으므로 train/test가 완전히 같은 함수를 타면서도 안전하다.
4. **GFS train/test pivot**: 같은 함수를 재사용. 결측 0건.
5. **라벨(`train_labels.csv`) 결측 확인**: `kpx_group_1` 104개, `kpx_group_2` 103개는 산발적 결측. `kpx_group_3`은 8,766개(대부분 2022년 전체)가 결측 — 0이 아니라 진짜 결측이므로 채우지 않고 NaN 그대로 둔다(임의 보간 금지 원칙).
6. **SCADA 단위 검증(가설 A vs 가설 B)**: `power_kw10m`이 "10분간 평균 출력(kW)"인지(가설 A, 1시간=6개 평균), 아니면 이미 "10분간 발전량(kWh)"인지(가설 B, 1시간=6개 합)를 실제 라벨과 대조해서 확정했다. 결과: 가설 B의 오차율이 2% 이내로 압도적으로 낮았다. `power_kw10m`은 이미 kWh 스케일이고, 1시간 발전량은 6개 구간의 합(sum)이다.
7. **SCADA 시간 집계**: 출력은 합, 풍속·풍향은 평균으로 집계. VESTAS 26,304행×38열(n_samples==6 비율 99.996%), UNISON 17,544행×17열(100%).
8. **그룹 합계 vs 라벨 전체 기간 대조**: 비율(라벨/SCADA합계) 평균이 세 그룹 모두 1.03~1.07로 "합" 방식이 맞다는 것을 재확인했다. 표준편차는 kpx_group_3이 가장 컸다(1.10) — 저풍속 시간대의 비율 왜곡 때문에 부풀려진 값이지만, 그룹 3이 상대적으로 다루기 어려운 그룹이라는 방향과는 일치한다.
9. **학습/평가 데이터 병합**: LDAPS·GFS wide 표와 라벨을 `forecast_kst_dtm` 기준으로 조인했다. train과 test에 완전히 같은 pivot·병합 로직을 적용해, 두 파이프라인이 서로 다른 전처리를 쓰지 않도록 했다.
10. **parquet 캐시 저장**: `train_merged.parquet`, `test_merged.parquet`, `scada_vestas_hourly.parquet`, `scada_unison_hourly.parquet`.

## Result

| 산출물 | shape | 비고 |
|---|---|---|
| `data/processed/train_merged.parquet` | 26,304 × 800 | LDAPS(480) + GFS(315) + 라벨(3) + 시각/공개시각(2) |
| `data/processed/test_merged.parquet` | 8,760 × 797 | LDAPS(480) + GFS(315) + 시각/공개시각(2) |
| `data/processed/scada_vestas_hourly.parquet` | 26,304 × 38 | 터빈 12기 × (kWh, ws, wd) + n_samples |
| `data/processed/scada_unison_hourly.parquet` | 17,544 × 17 | 터빈 5기 × (kWh, ws, wd) + n_samples |

핵심 결론: **SCADA `power_kw10m`은 "합(sum)"으로 시간 집계해야 라벨과 일치한다**(오차 2% 이내). LDAPS test 원본에만 3개 시각(전체의 0.03%)의 결측이 있었고 시간 선형보간으로 메웠다.

## So-what

- 이후 노트북(`03_features.ipynb` 이후)은 원본 CSV를 다시 읽지 않고 `data/processed/*.parquet`만 읽는다.
- SCADA "합" 방식 확정은 이후 피처 엔지니어링 단계의 파워커브 추정, 예보 풍속 보정 모델 학습에 바로 쓸 수 있는 기반이 됐다.
- 그룹 3(UNISON)의 라벨 대비 SCADA 비율 표준편차가 가장 크게 나온 것은 "그룹 3이 병목이 될 가능성"과 방향이 일치한다. 다만 이 표준편차 수치 자체는 저풍속 구간의 비율 왜곡 때문에 부풀려져 있어서, EDA 단계에서 설비용량 10% 이상 구간만 필터링해 다시 확인할 필요가 있다.
- `train_merged.parquet`(127MB), `test_merged.parquet`(43MB)는 `.gitignore`의 `data/` 규칙에 이미 포함되어 자동으로 git 추적에서 제외된다.
