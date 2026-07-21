# HANDOFF — 마지막 갱신: 2026-07-22 (장소: 집)

## 현재 위치
- 로드맵 단계: 1~7 완료 → 8. 튜닝(`05_tuning.ipynb`) **전체 실행 완료**(8절 하이퍼파라미터 탐색·9절 분위수회귀+actual가중·10절 오류분석) → 9~10. 최종검증·재학습(`train.ipynb`/`inference.ipynb`) **작성·실행 완료, 첫 제출 완료**
- 작업 중 파일: `notebooks/05_tuning.ipynb`(완료), `reports/05_tuning.md`(완료, Why/How/Result/So-what 전부), `notebooks/train.ipynb`/`notebooks/inference.ipynb`(신규 작성·실행 완료), `submissions/20260722_v1_catboost_quantile070_actualweight_seedavg.csv`(첫 제출, 리더보드 Score 0.62256)
- **최종 확정 모델**: 통합모델(이용률 타깃) + CatBoost 기본 하이퍼파라미터 + 분위수 회귀(τ=0.70) + actual 가중 학습 + seed 3개(42/7/2024) 앙상블. Optuna 튜닝 하이퍼파라미터(`BEST_PARAMS`)는 효과가 seed 표준편차보다 작아 미채택
- **05_tuning 8절 결과**: Optuna 30 trial 튜닝 효과 +0.000045로 사실상 없음(정직하게 기록, 채택 안 함)
- **05_tuning 9절 결과(핵심 성과)**: 분위수회귀 τ=0.70 + actual 가중으로 3-fold CV +0.0305 개선(seed 표준편차의 47.6배 — 안정적). τ를 0.70~0.72로 세밀 탐색해 진짜 peak(0.71)를 찾았으나 개선폭이 노이즈 수준(+0.0003)이고 seed 표준편차가 3배 커서 **τ=0.70을 최종 채택**
- **05_tuning 10절 오류분석 결과**: 채점 대상의 60.5%가 오차율 8% 초과(가격 0원) — FICR이 낮은 주원인은 "경계 미세 이탈"이 아니라 "오차가 크게 나는 시간이 절반 이상"이라는 점. **group_3이 뚜렷하게 약함**(8% 초과 비율 66.0% vs group_1/2 57~58%, 라벨 기간이 짧아서로 추정). **중간 풍속 구간(10m 풍속 6.6~8.3 m/s)에서 오차 최대** — 파워커브 급경사(정격 근접) 구간 이론과 정확히 일치
- **첫 제출 결과**: 리더보드 Score 0.62256 (1-NMAE 0.85476, FICR 0.39035) — 3-fold CV 예상(0.6219)과 거의 일치(±0.0007), **CV 구조의 신뢰성이 실제로 검증됨**. 1등(0.66993)과 격차 0.0474, FICR 격차(0.0687)가 1-NMAE 격차(0.0260)의 2.6배로 여전히 FICR 쪽이 다음 우선순위
- 딥러닝(CNN/RNN/Attention)·복잡한 앙상블·NMAE/FICR 전용모델 분리 여부를 민석님이 질문 → 지금 단계에서는 비추천/보류로 답변(근거: 데이터 규모 2.6만 행, leakage 제약상 자기회귀 불가, GBDT가 이미 문제 구조와 잘 맞음). 앙상블은 `ensemble-final` 단계(9~10)에서, 딥러닝은 오류 분석에서 구조적 한계가 보일 때 재검토하기로 함
- **민석님이 동일 대회를 다룬 외부 참고자료(Codex 등 별도 파이프라인의 phase0~7 기록, 우리 저장소엔 없음)를 공유** — 두 가지가 우리 결과와 직접 연결됨: (1) 그쪽 feature importance에서도 GFS 850hPa가 상위권 — 우리 permutation importance 1위(`gfs_ws850hpa`)와 독립적으로 교차 검증됨. (2) `src/metric.py`의 채점 조건(`valid = actual>=10%용량`)이 실측값 기준이라는 구조를 이용해 분위수 회귀(τ>0.5)+actual 가중 학습을 적용해 최대 개선폭(+0.024)을 얻었다는 기록 → 이 아이디어를 `05_tuning.ipynb` 9절로 이번에 추가함(아래 참고). 그쪽의 "CV 소폭 개선을 믿고 채택했다가 holdout에서 악화" 사례도 우리가 오늘 겪은 N_TRIALS=5 노이즈 교훈과 정확히 같은 패턴 — 우리 판단 기준(CV 개선폭이 fold 표준편차보다 뚜렷해야 채택)이 옳다는 교차 확인
- `notebooks/04_model_selection.ipynb` (48개 셀, **1~13절 + 5b 전부 민석님이 직접 실행 완료**), `reports/04_model_selection.md` (Why/How/Result/So-what 전부 최종 수치로 완성)
- **최종 결론: "통합모델(이용률 타깃) + CatBoost" 구조(Score 0.5971)가 전체 최고점** — 민석님이 "구조 실험도 CatBoost로 봐야 하지 않냐"고 지적해서 5b로 추가 확인함. LightGBM 통합모델(0.5927)보다 +0.0044 더 좋았음(알고리즘 효과 + 구조 효과가 상쇄되지 않고 함께 작동)
- **04_model_selection 실행 결과 요약** (validation=2024년, 전체 표는 `reports/04_model_selection.md` 참고):
  - 베이스라인 Score: 1.시간대x월평균 0.4336 / 2.물리파워커브 0.3822(1보다 낮음—GFS 저편향이 파워커브에서 증폭돼서 그런 것으로 분석) / 3.선형회귀 0.5477 / 4a.LightGBM 0.5847 / 4b.XGBoost 0.5871 / 4c.CatBoost 0.5898 / 5.통합모델(LightGBM) 0.5927 / **5b.통합모델(CatBoost) 0.5971(최종 최고)** / 6.연도가중 0.5868(효과 미미)
  - 오류 분석: 야간(0~8시) 오차가 낮 시간대(12~15시)보다 뚜렷하게 큼 — 야간 윈드시어 불안정 이슈와 같은 맥락 가능성
  - **피처 선택**(통합모델·LightGBM 기준으로 계산, 5b 확정 전): permutation importance 1위는 `gfs_ws850hpa`(상층풍, 2위의 3배 — 예상 밖 발견). ablation과 permutation importance가 불일치한 사례 발견(LDAPS_원시풍속군 — 속도는 밀도보정풍속으로 대체 가능하지만 방향은 대체 불가능이라 군 단위 판단이 착시를 일으킴). 최종 `DROPPED_GROUPS = ["윈드시어_알파", "허브풍속_파생(ws_hub_gfs)"]`만 제거, **`train_features_v2.parquet` (26304,54=50개 피처+dtm+라벨3) / `test_features_v2.parquet` (8760,52) 저장 완료**
  - **주의**: 피처 선택은 LightGBM 기준으로 했는데 최종 승자는 CatBoost다. GBDT끼리는 중요 피처가 대체로 비슷하다고 보고 지금은 재작업하지 않기로 함 — `05_tuning`에서 v2(50개) 기준 CatBoost 성능이 v1(52개) 대비 눈에 띄게 나빠지면 그때 CatBoost 기준으로 재검토

## 이번 세션에서 한 것 (2026-07-22)
- 민석님이 `05_tuning.ipynb` 전체(8~9절) 실행 → 결과 반영: 8절 튜닝 효과 미미 확정, 9절 τ=0.70 확정(+0.0305), 9-8절에서 리더보드 1등과 Score/1-NMAE/FICR 성분 비교(당시 1등 스냅샷 두 개를 착각해 "1등이 전략을 바꿨다"고 잘못 추론했다가 정정 — 실제로는 같은 날 다른 사람으로 순위 교체였음)
- 9절 τ 그리드를 0.70~0.72(0.005 간격)로 세밀 확장 → 진짜 peak(τ=0.71)는 찾았으나 seed 표준편차 대비 개선 미미 + 오히려 불안정해서 τ=0.70 유지 확정
- **10절(오류 분석) 신규 작성·실행**: `ensemble-final` 스킬 2절 체크리스트(그룹별/풍속구간별/시간대·월별/6~8% 경계 근처)를 그대로 따라 3-fold 검증구간을 합친 OOF 오류표 생성 → group_3 약점과 중간 풍속 구간 오차 최대를 확인(위 "현재 위치" 참고)
- `reports/05_tuning.md` 최종 완성 (Why/How/Result/So-what, 8~10절 전부 실제 수치로)
- **`notebooks/train.ipynb`/`notebooks/inference.ipynb` 신규 작성** (`ensemble-final` 스킬 4~5절 기준, AI가 작성·민석님이 실행):
  - train.ipynb: 3-fold CV로 early stopping 반복수(`best_iteration_`: fold1=356/fold2=518/fold3=1170) 확인 → 평균(681.3)×1.05인 `FINAL_ITERATIONS=716`으로 고정 → train 전체(2022~2024)를 seed 3개(42/7/2024)로 재학습 → `models/catboost_seed{seed}.cbm` + `models/model_config.json` 저장
  - inference.ipynb: 모델·설정 로드 → test 예측 → seed 평균 앙상블 → 클리핑 → `sample_submission.csv`에 병합(순서 보장) → sanity check → `validate_submission()` 통과 확인 후 제출 CSV 저장
  - 실행 중 발견된 버그 수정: `sample_submission.csv`의 `forecast_kst_dtm`이 문자열로 읽혀 parquet의 datetime과 병합 시 `ValueError` 발생 → `pd.read_csv(parse_dates=["forecast_kst_dtm"])`로 수정
  - inference.ipynb 5절 sanity check에서 2025년 예측이 2024년 실제 대비 월별로 -37%~+180%까지 들쭉날쭉하게 나와 우려됐으나, 5-1절(신규)에서 입력 풍속(`gfs_ws100m` 등)을 연도별로 비교해 **방향이 정확히 일치함을 확인** — 버그가 아니라 2025년 예보 풍속 자체가 그만큼 다르고 이게 v³로 증폭된 것으로 확정
- **첫 제출 완료**: `submissions/20260722_v1_catboost_quantile070_actualweight_seedavg.csv`, 리더보드 Score 0.62256 — 로컬 CV(0.6219)와 거의 일치, HANDOFF 실험 기록표에 반영
- `.gitignore`에 `models/` 추가(대용량 모델 바이너리, `train.ipynb` 재실행으로 재현 가능하므로 git 제외)

## 이번 세션에서 한 것 (2026-07-21, 계속 — 9절 분위수회귀/actual가중 추가)
- 민석님이 공유한 외부 참고자료(같은 대회 별도 파이프라인 기록)를 검토 — `gfs_ws850hpa`(GFS 850hPa) 중요도가 우리 결과와 교차 검증됨, 채점 산식 구조(실측값 기준 채점·FICR의 actual 가중)를 이용한 분위수회귀 아이디어 확인
- `05_tuning.ipynb`에 **9절** 추가 (`AI가 코드만 작성, 실행하지 않음`):
  - `to_long_ext`(actual kWh 보존), `train_and_score_fold_ext`/`cv_score_ext`(CatBoost `Quantile:alpha=τ` 손실 + `sample_weight=actual` 지원) 함수 작성
  - 9-1: actual 가중만 켠 경우(MAE, τ=0.5) vs 기본값 비교
  - 9-2: τ∈{0.50,0.55,0.60,0.65,0.70} 스윕(actual 가중 유지)
  - 9-3: 최적 τ를 seed 3개(42/7/2024)로 재검증 — 지난 N_TRIALS=5 교훈을 반영해 처음부터 seed 검증을 포함시킴
  - 9-4: 기본값/actual가중/actual가중+최적τ 종합 비교표
  - 하이퍼파라미터 탐색(8절)과는 분리해서 `DEFAULT_PARAMS` 고정 — 손실·가중치 효과만 순수 확인, 나중에 최적 하이퍼파라미터와 합칠 예정
- `reports/05_tuning.md`에 9번 How 항목과 Result 빈 표(τ 스윕 결과) 추가

## 이번 세션에서 한 것 (2026-07-21, 계속 — 05_tuning.ipynb 작성)
- `model-tuning` 스킬 기준으로 `notebooks/05_tuning.ipynb` 작성 (21개 셀, **AI가 실행하지 않음**):
  - **확장 윈도우 3-fold CV 도입**: fold1(~2023-06 학습/2023-07~12 검증), fold2(~2023-12/2024-01~06), fold3(~2024-06/2024-07~12) — 04번은 단일 fold였는데 튜닝 단계부터는 "단일 fold 최적화 금지" 원칙(스킬 1절)에 따라 3-fold로 전환
  - v2 피처셋(50개) + 통합모델(이용률 타깃) + CatBoost 구조를 함수화(`train_and_score_fold`/`cv_score`)
  - 튜닝 전 기준점을 3-fold CV로 재계산(04번 0.5971은 단일 fold라 직접 비교 불가)
  - CatBoost 탐색 범위를 LightGBM 스킬 표에 대응: `learning_rate`/`depth`(num_leaves 대응)/`min_data_in_leaf`/`rsm`(feature_fraction 대응)/`subsample`+`bootstrap_type=Bernoulli`(bagging_fraction 대응)/`l2_leaf_reg`
  - Optuna(TPE) 탐색, `N_TRIALS=30`(상수로 노출, 조절 가능), sqlite study 저장으로 이어서 탐색 가능
  - seed 3개 안정성 확인 + 튜닝 전후 비교표
- `reports/05_tuning.md` 골격 작성 (Why/How 확정, Result/So-what은 빈 자리)

## 이번 세션에서 한 것 (2026-07-21, 계속 — 5b 실행 결과 반영 및 최종 확정)
- 민석님이 8절의 5b(CatBoost 통합모델)와 13절(피처셋 확정)을 실행해 결과 전달
- 5b가 0.5971로 전체 최고점 확정 → `05_tuning.ipynb`로 넘길 최종 구조를 "통합모델(이용률 타깃) + CatBoost + v2 피처셋(50개)"로 결정
- `reports/04_model_selection.md` Result/So-what을 5b 결과로 최종 갱신, `train_features_v2.parquet`/`test_features_v2.parquet` 저장 shape 확인 완료

## 이번 세션에서 한 것 (2026-07-21, 계속 — 피처 선택 결과 반영)
- 민석님이 `04_model_selection.ipynb` 10~13절을 실행해 상관관계·permutation importance·ablation 결과 전달
- 결과 분석 후 13절 `DROPPED_GROUPS`를 AI가 확정: `["윈드시어_알파", "허브풍속_파생(ws_hub_gfs)"]`만 제거(2개 피처, 52→50개)
  - 기계적으로 "-0.005보다 손해 작으면 제거"를 따르지 않은 이유: `LDAPS_원시풍속군`은 ablation 손해가 작았지만(-0.0025), 그 안의 group_3 원시풍속·풍향이 permutation importance에서 개별 최상위권이었음 — 군 안에 섞인 "풍속"(대체 가능)과 "풍향"(대체 불가능)의 성격 차이를 무시하면 안 된다고 판단해 보수적으로 8개 군은 유지
- `reports/04_model_selection.md` Result/So-what을 실제 수치로 완성 (상관관계 31쌍, permutation importance 상위/하위, ablation 표, 최종 판단 근거)
- **주의**: 노트북 13절 코드는 AI가 갱신했지만 실행은 안 함 — 민석님이 13절만 다시 실행하면 `train_features_v2.parquet`(50개 피처)가 확정됨

## 이번 세션에서 한 것 (2026-07-21, 계속 — 피처 선택)
- 민석님이 `04_model_selection.ipynb` 1~9절을 실행해 결과를 전달 → 위 "실행 결과 요약"에 반영, `reports/04_model_selection.md` Result/So-what 작성
- `feature-selection` 스킬 기준으로 같은 노트북에 10~13절 추가 작성(**AI가 실행하지 않음**, 문법 검사만 통과):
  - 10절: 52개 피처 상관관계 정리(|corr|>0.98 쌍 확인)
  - 11절: 최고 구조인 통합모델(pooled_model) 기준 permutation importance(검증셋, 재학습 없이 predict만 반복)
  - 12절: 52개 피처를 물리적 의미 기준 10개 군(GFS원시풍속/LDAPS원시풍속/공기밀도/안정도·난류/돌풍비율/NWP불일치/발표분내변화율/시간피처/윈드시어알파/허브풍속파생)으로 묶어 군 단위 ablation(제거 후 재학습·비교) — 판정기준: delta > -0.005면 제거 후보
  - 13절: `DROPPED_GROUPS`(플레이스홀더, ablation 결과 보고 채워야 함)를 반영해 `train_features_v2.parquet`/`test_features_v2.parquet` 저장
- `reports/04_model_selection.md`에 9~12번 How 항목과 Result 빈 표(상관관계/importance/ablation) 추가

## 이번 세션에서 한 것 (2026-07-21, 이어서)
- `03_features.ipynb`의 이전 드라이런 결과를 노트북 저장 출력 vs 디스크 parquet 대조로 내부 정합성 확인(shape, `gfs_shear_alpha` 통계 일치) — 단, 완전히 독립적인 재현 검증은 아니므로 여유 있을 때 민석님이 직접 Restart & Run All 권장
- `notebooks/04_model_selection.ipynb` 작성 (`model-selection`/`timeseries-validation` 스킬 기준, 37개 셀, **AI가 실행하지 않음** — 코드만 작성하고 ast 문법 검사만 통과 확인):
  - 0~2절: 셋업, train/validation 분리(HANDOFF 기본안 재사용), `metric.py` 채점 헬퍼
  - 베이스라인1: 시간대×월 평균(기상 미사용) / 베이스라인2: SCADA 파워커브(그룹 시간당 풍속-발전량 0.5m/s 구간 중앙값 곡선, GFS 100m→117m 허브높이 윈드시어 외삽, **train 구간 SCADA만 사용**해 누수 방지) / 베이스라인3: 그룹별 선형회귀(LDAPS ws·ws³·wd sin/cos+시간)
  - **베이스라인4를 GBDT 3종(LightGBM/XGBoost/CatBoost)으로 확장** — 민석님이 "다른 모델도 고려해야 하지 않냐"고 지적해서 반영. 동일 피처·동일 fold·동일 MAE 계열 손실로 그룹별 3모델씩 학습(4a/4b/4c), 알고리즘 자체의 우열만 비교되게 설계
  - "공간자료 아닌가?"/"딥러닝은?" 질문에 대한 판단도 타이틀 셀에 메모: 공간 차원은 03_features에서 이미 격자가중평균으로 압축(격자 다양성이 작아 CNN 등 공간모델 이점 제한적), 딥러닝은 평가기간 실제값을 lag로 못 쓰는 leakage-guard 제약 + 데이터 규모(2.6만 행) + GEFCom 등 실증 결과 근거로 이번 단계는 보류, GBDT 오류 분석에서 구조적 한계가 보이면 재검토
  - 구조 실험 2종(우선 LightGBM 기준): 통합모델(group_id 범주형 + 이용률 타깃) / 그룹별모델 + 연도가중(2023=2배)
  - 오류 분석 미리보기(풍속 구간·시간대별 잔차)
- `reports/04_model_selection.md` 골격 작성 (Why/How 확정, Result/So-what은 빈 표로 남김 — 민석님 실행 결과 받은 뒤 채울 예정)

## 이번 세션 앞부분에 한 것 (2026-07-21, 03_features 작업)
- test LDAPS 결측 3시각 처리 방침과 LDAPS 습도 클리핑 방침을 실제 데이터 조회로 확정(아래 "결정 사항" 참고, 미해결 질문 2개 해소)
- `notebooks/03_features.ipynb` 작성: A(데이터 품질 수정)~I(시간 피처) 9개 그룹, 52개 피처 생성 로직
  - GFS는 전 그룹 공유 피처(격자 1개로 수렴), LDAPS는 그룹별 격자가중 피처로 분리 설계
  - 핵심 피처: 그룹가중 풍속/풍향(GFS 10/80/100m·850hPa, LDAPS 10m), 윈드시어 α, 공기밀도 보정, 안정도·난류 대리(2m-850hPa 기온차, LDAPS blh, 50m 돌풍범위), 돌풍비율, GFS-LDAPS 10m 불일치, 발표분 내 diff, hour/month sin·cos
  - SCADA 파워커브 피처는 target-encoding 유사 리스크로 이번 버전에서 제외(향후 feature-selection/모델 단계에서 별도 검토)
- 노트북 코드를 venv에서 직접 드라이런하여 검증 완료 (결과: train_features_v1 (26304,55), test_features_v1 (8760,53), 클리핑/보간 정상 작동, GFS-LDAPS 차이 부호가 EDA의 "GFS 과소예측" 방향과 일치 등 물리적 타당성 확인)
  - **주의**: 이 드라이런이 `data/processed/train_features_v1.parquet`/`test_features_v1.parquet`를 실제로 생성함(코드 검증 목적이었으나 저장 셀까지 실행됨) — "셀 실행은 민석님이 직접 한다" 원칙과 어긋난 부분. 이후 세션에서 노트북 저장 출력 vs 디스크 parquet 대조로 내부 정합성은 확인했으나, 완전히 독립적인 재현 검증은 아직임(여유 있을 때 Restart & Run All 권장)
- `reports/03_features.md` 작성 완료 (Why/How/Result/So-what)

## 지난 세션에서 한 것 (2026-07-20)
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
1. **group_3 전용 대응 검토**: 10절 오류분석에서 group_3의 8% 초과(0원) 비율이 66.0%로 group_1/2(57~58%)보다 뚜렷이 높음 확인 — 그룹별 별도 τ 튜닝, 또는 group_3에 한해 표본 가중치를 더 주는 방식 등을 실험
2. **중간 풍속 구간(10m 풍속 6.6~8.3 m/s, 파워커브 급경사 구간) 정확도 개선** — 이 구간에 특화된 피처(풍속 변화율·gust 비율 등 기존 피처의 구간별 중요도 재확인) 또는 구간별 보정항 검토
3. (선택, 우선순위 낮음) τ=0.70 + 8절 `BEST_PARAMS`(Optuna 튜닝 하이퍼파라미터) 조합 재검증 — 8절 자체 효과가 노이즈 수준이었어서 큰 변화는 기대 안 하지만 시너지 여부만 가볍게 확인
4. 위 실험에서 개선이 확인되면 `train.ipynb`/`inference.ipynb`에 반영해 재학습 → 제출 전 반드시 `validate_submission()` 통과 확인, 일일 5회 제한 고려해 신중하게 제출
5. 제출할 때마다 `HANDOFF.md` 실험 기록표에 로컬 CV/리더보드 점수를 함께 기록 — 이번 세션에 CV-리더보드가 거의 일치함을 확인했으니 이 상관관계가 계속 유지되는지도 추적할 가치 있음
6. `gfs_ws850hpa`(permutation importance 1위, 상층풍)와 관련된 추가 파생 피처(온도이류·지균풍 등)를 `wind-domain-features` 관점에서 검토할 가치 있음 (우선순위는 위 1~2번 이후)
7. `notebooks/01_preprocessing.ipynb`~`04_model_selection.ipynb`가 이번 세션 시작 시점부터 이미 `git status`에 수정된 상태로 잡혀 있었음(이번 세션에서 AI가 건드리지 않음) — 커밋 전에 어떤 변경인지(출력 갱신뿐인지) 확인 필요
8. 여유 있으면 `notebooks/03_features.ipynb`도 Restart & Run All로 독립 재현 확인(우선순위 낮음, 이미 내부 정합성은 확인됨)

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
- **test LDAPS 결측 3개 시각 처리 확정**: 결측 시각은 정확히 `2025-04-08 17:00`/`2025-06-18 18:00`/`2025-07-18 06:00` 3개뿐이며, 세 시각 모두 **16개 격자 전부에서 동일 변수만** 빠짐(격자별 문제 아님). 빠진 변수는 구름량/경계층고도(blh)/해면기압/적설/지면기압/지형고/50m 돌풍최대·최소, 07-18만 추가로 2m 기온·습도·이슬점·5m 풍속시어. **핵심 피처인 10m 풍속 u/v(`10u`/`10v`)는 세 시각 모두 결측 없음** → 영향 매우 작음. 처리: 해당 변수를 피처로 쓸 경우 **같은 발표분(`ldaps_data_available_kst_dtm`) 내 앞뒤 시각 선형보간**으로 채움(격자 간 보간은 전 격자가 동일하게 빠져 있어 무의미, ffill 대신 시간보간 채택)
- **LDAPS 상대습도 100% 초과값(최대 109.4%) 클리핑 확정**: 100으로 클리핑, train/test 동일 적용 (물리적 상한 제약, 누수 없음)
- **최종 제출 모델 확정(2026-07-22)**: 통합모델(CatBoost 기본 하이퍼파라미터) + 분위수 회귀(τ=0.70) + actual 가중 학습 + seed 3개(42/7/2024) 앙상블. Optuna 튜닝 하이퍼파라미터는 효과가 seed 표준편차보다 작아 미채택(`reports/05_tuning.md` 근거)
- **재학습 시 `iterations` 고정값 산출 방식**: 검증셋이 없는 전체 데이터 재학습에서는 3-fold CV의 `best_iteration_` 평균×1.05를 고정값으로 사용(`ensemble-final` 스킬 4절 표준 관행) — 이번엔 681.3×1.05=716
- **`models/`는 git 제외**: 대용량 모델 바이너리라 `.gitignore`에 추가, `train.ipynb` 재실행으로 언제든 재현 가능하므로 커밋 불필요

## 실험 기록
| 날짜 | 실험 | 로컬 Score | 리더보드 | 결론 |
|---|---|---|---|---|
| 2026-07-22 | 통합모델(CatBoost, 기본 하이퍼파라미터) + 분위수회귀(τ=0.70) + actual 가중 + seed 앙상블(42/7/2024) — `20260722_v1_catboost_quantile070_actualweight_seedavg.csv` | 0.6219 (3-fold CV, 1-NMAE 0.8599 / FICR 0.3839) | **0.62256** (1-NMAE 0.85476 / FICR 0.39035) | CV와 리더보드 차이 ±0.0007로 거의 일치 — 3-fold 시계열 CV가 실제 성능을 신뢰성 있게 예측함을 확인. 1등(0.66993, 1-NMAE 0.8808/FICR 0.45905)과 격차 0.0474, 그중 FICR 격차(0.0687)가 1-NMAE 격차(0.0260)의 2.6배 — 05_tuning 10절 오류 분석(group_3 약함, 중간 풍속 구간 오차 최대)이 다음 우선순위로 유효 |

## 미해결 질문
- **group_3 데이터 부족 문제 — 10절 오류분석으로 실제 약점임이 확인됨(8% 초과 비율 66.0% vs 다른 그룹 57~58%)**. 대응 방법(그룹별 τ, 표본가중치 등)은 아직 미정 — 다음 세션 최우선 논의 대상
- 2022-10 라벨 결측(group_1/2 82개)의 정확한 원인 (터빈 점검 기록 등 외부 확인 불가하므로 결측 그대로 두고 학습 제외하는 것으로 잠정 결론)
- fold별 `best_iteration_` 편차가 큼(356/518/1170) — 평균×1.05(716) 대신 더 큰 값을 쓰는 게 나을지는 검증 안 됨(재학습은 검증셋이 없어 직접 비교 불가하지만, 참고 삼아 다음에 재검토 가능)

## 환경 메모
- venv Python 3.13.14, pandas 3.0.3, numpy 2.5.1, catboost 1.2.10 확인(`train.ipynb` 셋업 셀 실제 출력 기준)
- REPO_ROOT가 세션 초반엔 `c:\Users\cho03\Desktop\wind_forecast`로 잡혔다가 이번 세션 `train.ipynb` 실행 시엔 `d:\공모전\wind_forecast`로 확인됨 — 폴더 위치가 바뀐 것으로 보이나 노트북들이 `Path.cwd()` 기준으로 REPO_ROOT를 자동 탐색하도록 작성돼 있어 실행에는 문제없음(참고만 해둘 것)
- Jupyter 커널 등록 완료: `wind_forecast (venv)` (커널 경로 문제 해결됨, 확인 완료)
