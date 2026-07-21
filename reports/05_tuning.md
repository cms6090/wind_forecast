# 05. 하이퍼파라미터 튜닝 — 통합모델(CatBoost)

## Why

- `04_model_selection`에서 확정한 최종 승자 구조("통합모델(이용률 타깃) + CatBoost + v2 피처셋 50개")의 하이퍼파라미터를 `model-tuning` 스킬 기준으로 탐색한다.
- `model-tuning` 스킬 1절: "튜닝 목적 함수 = 동일 CV 구조의 fold 평균 Score. 단일 fold 최적화 금지." `04_model_selection`은 비교 단계라 단일 fold(2022~2023 train/2024 validation)로 충분했지만, 실제 하이퍼파라미터를 탐색하는 단계에서는 fold 하나의 우연한 패턴에 과적합될 위험이 커서 확장 윈도우 3-fold CV로 전환한다(`timeseries-validation` 스킬 1절 보강안).
- 튜닝으로 짜내는 개선폭은 보통 작다(`model-tuning` 스킬 4절: "튜닝으로 짜낸 0.001보다 피처 하나의 0.01이 크다") — 그래서 개선이 없다는 결과도 정직하게 기록하는 게 목적 중 하나다.

## How

1. **v2 피처셋 사용**: `04_model_selection`의 `train_features_v2.parquet`/`test_features_v2.parquet`(50개 피처, `feature-selection` 결과 반영본)를 그대로 불러온다.
2. **확장 윈도우 3-fold CV 도입**: fold1(2022-01~2023-06 학습/2023-07~12 검증), fold2(~2023-12 학습/2024-01~06 검증), fold3(~2024-06 학습/2024-07~12 검증). 랜덤 K-Fold는 시계열 누수라 금지(`timeseries-validation` 1절), 확장 윈도우가 실제 운영과 가장 비슷한 구조.
3. **통합모델 학습 함수 재사용**: `04_model_selection` 8절(5b)과 동일한 구조(group_id 범주형 + 이용률 타깃, CatBoost, MAE 손실, train 내 시간순 마지막 15% early stopping)를 함수화해서 fold별/3-fold 평균 점수를 낼 수 있게 함.
4. **튜닝 전 기준점**: 04번 5b의 기본 파라미터(`iterations=2000, learning_rate=0.05`)를 이번 3-fold CV로 다시 채점 — 04번의 단일 fold 점수(0.5971)와는 CV 구조가 달라 직접 비교하면 안 되므로, 항상 "같은 구조끼리"만 비교.
5. **CatBoost 탐색 범위 설계**: `model-tuning` 스킬 2절의 LightGBM 표를 CatBoost 파라미터로 대응(`learning_rate`/`depth`/`min_data_in_leaf`/`rsm`/`subsample`+`bootstrap_type=Bernoulli`/`l2_leaf_reg`). `iterations`는 튜닝 대상에서 제외하고 early stopping(100 rounds)으로 결정(스킬 1절 원칙).
6. **Optuna(TPE) 탐색**: `N_TRIALS`(기본 30, 상수로 노출해 조절 가능)만큼 coarse 탐색. sqlite(`data/processed/optuna_catboost_pooled.db`, git 제외)에 저장해 같은 기기에서 이어서 탐색 가능(스킬 5절).
7. **seed 안정성 확인**: 최적 파라미터를 seed 3개(42/7/2024)로 재검증, 평균·표준편차 확인(스킬 3절).
8. **튜닝 전후 비교**: 같은 3-fold CV 구조에서 기본값 vs 튜닝값을 나란히 비교, 개선폭이 fold 표준편차 이내면 "효과 미미"로 정직하게 기록.

## Result

*(민석님이 `05_tuning.ipynb`를 실행한 뒤 아래를 채운다 — N_TRIALS는 처음엔 작게 돌려서 소요 시간을 가늠하는 것을 추천)*

| 구분 | 3-fold 평균 Score | 비고 |
|---|---|---|
| 튜닝 전(기본값) | | |
| 튜닝 후(최적 파라미터, seed 평균) | | |

- 최적 파라미터: (실행 후 채움)
- seed 3개 평균/표준편차: (실행 후 채움)
- 상위 trial 파라미터 분포: (실행 후 채움 — 특정 값 근처로 몰려 있으면 2차 탐색 필요)

## So-what

*(실행 결과를 보고 아래 판단을 채운다)*

- 튜닝 효과가 fold 표준편차보다 큰지(유의미한 개선인지), 작다면 그 자체를 정직하게 기록하고 다음 단계(앙상블/피처 보강)로 넘어갈지
- 2차 좁은 범위 탐색이 필요한지
- `train.ipynb`에 하드코딩할 최종 파라미터 확정
