# HANDOFF — 마지막 갱신: 2026-07-23 (장소: 집)

## 이번 세션 마지막 — 외부 파이프라인과 격차 재확인, 다음 실험 보류
- 2차 제출(0.625596) 후 민석님이 "여전히 외부 파이프라인(exp017, Public 0.63886)이 더 높다"고 재확인 → 격차 재분석
- **격차의 유력 후보를 특정함**: 외부 파이프라인은 `phase4_tuning.md` §2-4에서 **그룹별로 다른 τ**(group_1=0.70/group_2=0.50/group_3=0.65)를 썼다 — group_2는 이용률이 셋 중 가장 높아(0.328) 10% 임계값 미만 확률이 낮으므로 위쪽 편향(τ>0.5)이 덜 필요했다는 근거. **우리는 지금까지 CatBoost에 항상 전 그룹 공통 τ=0.70만 썼다** — 11-4에서 group_3만 다른 τ를 시도한 적은 있지만 group_1·group_2는 항상 0.70 고정이었음. 이 부분이 시도된 적이 없다는 게 이번에 새로 특정된 사실
- **다음 실험 제안(보류)**: 05_tuning.ipynb 15절로 "그룹별 τ 재검토"(그룹마다 독립적으로 0.50~0.75 그리드 탐색 → 최적 조합 seed 재검증 → CatBoost+MLP 블렌드 가중치 재탐색)를 제안했으나, **민석님이 "학교 가서 하겠다"고 보류** — 코드는 아직 작성 안 함(제안만 한 상태)
- **다음 세션 시작 시 최우선 행동**: 15절(그룹별 τ 재검토) 코드 작성부터 시작. 기존 인프라(11-2의 `train_fold_model`/`predict_group`, 14-8의 `single_group_score`)를 그대로 재사용 가능해 새로 만들 코드는 많지 않음
- (참고, 아직 미착수) 더 큰 잠재 레버로 피처셋 크기 차이도 있음 — 외부 파이프라인은 179개 피처를 거의 그대로 썼고 어떤 ablation도 도움이 안 됐다고 기록(`phase6_nn_metric_loss.md` §1-1 "raw 800컬럼만 써도 -0.0116" 등 오히려 피처를 줄이면 손해). 반면 우리는 04번에서 50개로 줄임(LightGBM 기준 선택, 최종 승자는 CatBoost). 이 불일치는 아직 검증 안 했고, 그룹별 τ 실험 이후 후순위로 검토 가치 있음

## 이번 세션에서 한 것 (2026-07-22, 계속 — train.ipynb/inference.ipynb에 MLP+블렌드 반영)
- `src/nn.py`에 재사용 함수 2개 추가: `build_features`(group_id 원-핫 + 표준화, 05_tuning 14절 로컬 함수와 동일 로직을 공용화), `train_mlp_full`(검증셋 없이 고정 epoch로 전체 데이터 학습 — CatBoost의 `FINAL_ITERATIONS` 방식과 대응)
- **`train.ipynb` 수정** (AI가 코드 작성, 실행은 안 함):
  - 셋업에 `torch`/`from src import nn as mlp_nn` 추가
  - 2절에 MLP 설정 추가: `MLP_T_SOFT=0.003`, `MLP_HIDDEN=(256,256)`, `MLP_DROPOUT=0.15`, `BLEND_WEIGHTS={g1:0.4, g2:0.5, g3:1.0}`
  - **3b절 신규**: MLP도 CatBoost와 같은 3-fold CV로 조기 종료 시점(`best_epoch`) 확인 → 평균×1.05로 `MLP_FINAL_EPOCHS` 결정
  - **4b절 신규**: MLP를 train 전체로 seed 3개(42/7/2024) 재학습, `models/mlp_seed{seed}.pt` + `models/mlp_scaler.npz` 저장
  - 5절(model_config.json)에 `mlp`(하이퍼파라미터·입력차원·torch버전) + `blend_weights` 추가
  - 6절(sanity check)을 CatBoost 단독 vs 최종 블렌드 비교로 확장 — group_3만 뚜렷이 달라야 정상(블렌드 가중치 1.0이므로)이라는 검증 포인트 추가
- **`inference.ipynb` 수정**: MLP 모델·스케일러·블렌드 가중치 로드 → 그룹별로 CatBoost·MLP 각각 seed 평균 예측 → 그룹별 가중치로 블렌드(클리핑은 블렌드 후 마지막에) → 이하 기존 파이프라인(제출 조립·sanity·저장·검증) 그대로. 파일명 설명을 `catboost_mlp_groupblend_quantile070`(v2)로 변경
- **두 노트북 로직을 축소된 설정(iterations=100, epochs=5, seed 2개)으로 실제 데이터에 대해 end-to-end 스모크 테스트 완료** — train 파이프라인 전체(3절~6절) 및 inference 파이프라인 전체(로드~`validate_submission()` 통과)가 에러 없이 동작함을 확인. 테스트 산출물은 정리 완료(실제 `models/`, `submissions/`에는 영향 없음)
- **민석님이 `train.ipynb`/`inference.ipynb` 실제 실행 완료, 전부 에러 없이 성공**:
  - `FINAL_ITERATIONS=716`(CatBoost, 첫 제출과 동일 — 3-fold 구조가 그대로라 당연), `MLP_FINAL_EPOCHS=60`(fold별 best_epoch 37/56/77, 평균 56.7×1.05)
  - CatBoost·MLP·MLP_scaler 전부 `models/`에 저장 완료
  - 6절 sanity check로 블렌드가 의도대로 적용됨을 확인: group_3(가중치 1.0)이 group_1(0.4)보다 CatBoost 대비 차이가 월별로 일관되게 더 크게 남
  - `inference.ipynb`에서도 group_3 블렌드 범위가 MLP 범위와 정확히 일치(가중치 1.0이므로 당연 — 버그 없음 확인)
  - **제출 파일 생성 완료**: `submissions/20260723_v2_catboost_mlp_groupblend_quantile070.csv`, `validate_submission()` 통과. 2024→2025 계절 패턴(겨울↑여름↓)도 첫 제출 때와 동일하게 유지, 클리핑 비율 0%(첫 제출 0.2~1.5%보다 오히려 개선)
- **2차 제출 완료, 리더보드 결과 확인**: Score **0.625596**, 1-NMAE **0.857136**, FICR **0.394057**
  - **1차 제출(CatBoost 단독, 0.62256) 대비 +0.00304 — 실제 리더보드에서도 개선이 재현됨**(1-NMAE +0.00238, FICR +0.00371 모두 개선)
  - 다만 로컬 CV가 예측한 개선폭(+0.0082)의 약 37%만 실현됨 — 로컬 CV는 2022~2024 기준, 실제 test는 2025년이라 두 기간의 분포 차이(`inference.ipynb` 5-1절에서 이미 확인한 "2025년 풍속 자체가 2024년과 다름") 때문으로 추정. 방향은 정확히 일치하므로 CV의 "어느 쪽이 나은지" 판단력은 여전히 유효하고, "개선폭의 절대 크기"만 낙관적으로 나오는 경향이 있다는 뜻으로 해석
  - **다음 세션 참고**: 앞으로 로컬 CV 개선폭을 리더보드에 그대로 기대하지 말고, 방향(+/-)과 상대적 크기 비교 용도로만 신뢰할 것. 실험 기록표(아래)에 반영 완료

## 이번 세션에서 한 것 (2026-07-22, 계속 — 14절 실행 결과: 진짜 개선 확인, 최종안 변경)
- torch 설치 후 `build_mlp_features`의 NaN 버그(`*_diff_prev`류 피처 결측을 0으로 채우지 않아 발생), 14-8의 `make_pred_df` KeyError(단일 그룹만 채점하려는데 3개 그룹 컬럼을 요구하는 함수를 잘못 씀 — `single_group_score`로 직접 복제해서 해결) 2건 수정
- 민석님이 14절(14-1~14-9) 전체 실행 → **결과**:
  - MLP 단독(T_soft=0.003, seed 평균): CV 0.6256(표준편차 0.0006), CatBoost 최종안(0.6213) 대비 **+0.0043**
  - CatBoost+MLP 전역 블렌드(w=0.5): CV 0.6293 — MLP 단독보다도 좋음(진짜 앙상블 효과)
  - **그룹별 블렌드(g1=0.4/g2=0.5/g3=1.0) + seed 3개 재검증(최종): CV 0.6295(표준편차 0.0013), CatBoost 대비 +0.0082(표준편차의 6.2배)**
  - **group_3(11절부터 계속 약점이던 그룹)은 CatBoost를 완전히 배제(MLP 100%)하는 게 최적** — 데이터가 적은 상황에서 MLP가 트리보다 덜 과적합했을 가능성
- **11~13절과 달리 이번 개선은 seed를 바꿔도 거의 그대로 유지됨 — 진짜 개선으로 확정**
- `05_tuning.ipynb` 14-10에 종합 해석 작성, `reports/05_tuning.md`에 15번(노트북 14절) Result/So-what 완성, 전체 노트북 결론을 "CatBoost 단독"에서 "CatBoost+MLP 그룹별 블렌드"로 갱신
- **최종 채택안 변경**: CatBoost(τ=0.70+actual가중) 단독 → **CatBoost + MLP(T_soft=0.003) 그룹별 블렌드(g1=0.4/g2=0.5/g3=1.0)**
- **다음 세션(또는 이어서) 할 일**: `train.ipynb`/`inference.ipynb`에 MLP 학습(`src/nn.py` 재사용) + 블렌드 로직 반영 → 전체 데이터(2022~2024)로 재학습 → 재제출. 일일 5회 제출 제한 고려해 신중하게 진행

## 이번 세션에서 한 것 (2026-07-22, 계속 — 14절 MLP+미분가능 FICR손실 신규 작성, 딥러닝 방향 전환)
- AI가 11~13절 실패 이후 "개선 탐색 마무리, 최종 제출 준비로 전환"을 제안했으나, 민석님이 **딥러닝을 실제로 시도해본 별도 파이프라인 기록(외부 문서, 우리 저장소엔 없음)을 공유하며 강하게 반박** — 그 파이프라인은 LightGBM 단독(로컬 0.6308) 대비 MLP 블렌드로 로컬 0.6458, **실제 리더보드에서도 +0.0160 개선**(exp016 Public 0.62284 → exp017 Public 0.63886)을 실측으로 검증함
  - **AI가 입장을 수정**: 이전 "딥러닝 비추천" 판단은 "데이터가 작아 신경망이 GBDT를 일반적으로 못 이긴다"는 표준론이었는데, 저 사례는 그게 아니었음 — 핵심은 신경망 자체가 아니라 **채점 산식의 계단 함수(FICR)를 시그모이드로 미분 가능하게 근사해 backprop으로 직접 최적화**했다는 것. CatBoost/LightGBM은 커스텀 손실에 2차 미분(hessian) 근사가 필요해 이런 손실에서 불안정(그 외부 문서에서도 커스텀 LightGBM 목적함수는 CV 0.5641로 폭락하며 실패, 결국 신경망으로 전환해 성공). 우리가 11~13절에서 시도한 건 전부 quantile/MAE 손실의 간접 우회였지 FICR 계단 구조를 직접 겨냥한 적이 없었음
- **`src/nn.py` 신규 작성**: `MLP`(입력→256→256→1, sigmoid 출력), `soft_metric_loss`(미분 가능한 FICR 근사), `true_score`(조기종료용, 진짜 계단함수), `train_mlp`(전체배치 학습)
- `05_tuning.ipynb`에 **14절 신규 작성**(20개 셀 추가, 총 107개, AI가 코드만 작성·미실행):
  - 14-1~14-3: torch/src.nn import, 피처 행렬 헬퍼(group_id 원-핫 + 표준화), 학습·예측·CV 함수
  - 14-4: MLP 단독 baseline CV, CatBoost 기준점과 비교
  - 14-5: `T_soft`(계단 근사 온도) 스윕
  - 14-6: seed 3개 안정성 확인
  - 14-7: CatBoost+MLP 예측 캐시 후 전역 블렌드 가중치 스캔
  - 14-8: 그룹별 블렌드 가중치 최적화(11절 `compute_group_metrics`/`combined_score` 재사용)
  - 14-9: 최종 블렌드 seed 3개 재검증
  - 14-10: 종합 해석은 빈 자리
- `requirements.txt`에 `torch` 추가, **venv에 CPU 버전 설치 진행**(백그라운드, 완료 확인 필요)
- `reports/05_tuning.md`에 15번(노트북 14절) Why/How 추가, Result/So-what은 실행 결과 대기
- **다음 세션(또는 이어서) 할 일**: torch 설치 확인 → 민석님이 14-1~14-9 실행(다른 절보다 오래 걸릴 수 있음, T_soft 스윕 5개+seed 3개+블렌드 재학습이 전부 3-fold 학습을 요구) → 결과 전달하면 14-10 해석과 리포트 Result/So-what 채움. 개선 확인되면 `train.ipynb`/`inference.ipynb`에 MLP+블렌드 구조 반영해 재학습·재제출

## 이번 세션에서 한 것 (2026-07-22, 계속 — 13절 실행 결과: 개선 없음, 개선 탐색 마무리 제안)
- 민석님이 13절(13-1~13-4) 실행 → **blws_mag/blws_ratio/둘다 3개 조합 전부 baseline(0.6219)보다 손해**(-0.0008~-0.0016) — 10m 풍속과 상관 0.73인 진짜 새로운 원시변수였는데도 실패. seed 재검증까지 갈 필요 없이 종료
- `05_tuning.ipynb` 13-5에 **11·12·13절 종합 해석** 작성: 손실함수/가중치(11절)·저비용 피처(12절)·신규 원시변수(13절) 세 가지 성격이 다른 레버가 모두 실패 → 10절 진단("채점대상 60.5%가 오차율 8% 초과, 원인은 예보 자체 오차")이 세 번 독립적으로 재확인됨
- **AI 제안(민석님 승인 대기): 여기서 개선 탐색을 마무리하고 최종 제출 준비로 전환.** 모델·피처·하이퍼파라미터 변경 없이 기존 확정안(CatBoost + 50개 피처 + τ=0.70 + actual 가중) 유지 — `train.ipynb`/`inference.ipynb` 재실행 불필요, 제출도 그대로(0.62256)
- `reports/05_tuning.md`에 14번(노트북 13절) Result/So-what 완성, 전체 노트북 결론 갱신
- **다음 세션 시작 시**: 민석님과 "탐색 마무리 vs 계속 시도" 결정 확인 필요. 마무리하기로 하면 로드맵상 남은 건 최종 검증·앙상블 안정성 재점검 정도(`ensemble-final` 스킬 9~10절은 이미 완료된 상태이므로, 사실상 대회 종료까지 대기하며 리더보드 변화만 관찰하는 국면일 가능성)

## 이번 세션에서 한 것 (2026-07-22, 계속 — 모델 아키텍처 질문 답변 + 13절 미사용 원시변수 신규 작성)
- 민석님이 "CatBoost 대신 Prophet, 딥러닝(MLP/RNN/CNN/Attention) 고려해볼까?" 질문 → 비추천으로 답변: 11~12절에서 확인된 정체가 "모델이 패턴을 못 배워서"가 아니라 "입력값(NWP 예보) 자체의 정보량 한계"라 모델 교체로는 해결 안 됨. Prophet은 계절성보다 외생변수(풍속)에 좌우되는 이 문제 구조와 안 맞고 그룹 풀링 효과도 잃음. 딥러닝은 데이터 규모(풀링해도 3.9만 행)가 tabular GBDT 우위 영역이고 leakage 제약상 자기회귀 불가, 공간정보도 이미 03_features에서 압축됨 — 기존 HANDOFF 판단(딥러닝은 오류분석에서 구조적 한계가 보이면 재검토) 재확인, 오히려 이번 결과는 "재검토 불필요" 쪽 증거로 해석
- 11·12절 실패의 공통점(이미 모델이 아는 정보의 재포장)을 반성해 `data_description.md`를 다시 훑음 → 지금까지 어떤 피처에도 안 쓰인 `heightAboveGround_5_XBLWS`/`YBLWS`(LDAPS 5m 경계층 바람) 발견. 원시데이터로 직접 확인: 크기 ~0.15 m/s(10m 풍속의 1/40 수준), 10m 풍속과 상관 0.73(관련은 있으나 완전 중복 아님), test에 1개 발표분 16셀 결측
- `05_tuning.ipynb`에 **13절 신규 작성**(10개 셀 추가, 총 87개, AI가 코드만 작성·미실행):
  - 13-1: `train_base_wide.parquet`에서 `03_features.ipynb`와 동일한 `GROUP_GRID_WEIGHTS`/`group_weighted_uv` 로직을 05_tuning에 재정의해 `{group}_ldaps_blws_mag`/`blws_ratio`(=mag/ws10m, gust_ratio와 같은 방식) 계산, test 결측은 같은 발표분 내 시간보간으로 처리
  - 13-2: baseline(50)/+mag(53)/+ratio(53)/+둘다(56) 4개 피처셋 3-fold CV 비교
  - 13-3: 이긴 조합만 seed 3개 재검증(11·12절과 같은 원칙)
  - 13-4: 승자로 중간풍속 구간(6.6~8.3 m/s) 오차 재확인 — 12-4와 같이 self-contained(0~4절+9-2/9-3만 있으면 단독 실행 가능)
  - 13-5: 종합 해석은 빈 자리, 실행 결과 대기
- `reports/05_tuning.md`에 14번(노트북 13절) Why/How 추가, Result/So-what은 실행 결과 대기
- **다음 세션(또는 이어서) 할 일**: 민석님이 13-1~13-4 실행 → 결과 전달하면 13-5 해석·리포트 Result/So-what 채움. 개선 확인되면 `03_features.ipynb`에 정식 그룹(J절)으로 추가해 v3 parquet 생성 후 재학습·재제출

## 이번 세션에서 한 것 (2026-07-22, 계속 — 12절 실행 결과: 개선 없음)
- 민석님이 12절(12-1~12-4) 실행 → **4개 피처셋(baseline/+상호작용/+shear_alpha/+둘다) 중 baseline(0.6219)이 최고, 나머지는 전부 -0.0016~-0.0019로 손해** — seed 재검증까지 갈 필요 없이 개선 없음으로 종료
- 원인: 풍향×풍속 상호작용은 이미 있는 개별 피처(`ws`, `sin(wd)`, `cos(wd)`)와 공선이라 새 정보가 아니었고, `gfs_shear_alpha`는 `04_model_selection`의 제거 근거(원시 풍속으로 대체 가능, 야간 무풍구간 불안정)가 최종 모델에서도 재확인됨
- `05_tuning.ipynb` 12-5, `reports/05_tuning.md`에 결과 반영 완료
- **11절(group_3)·12절(피처보강) 둘 다 실패 — 손실함수·가중치·저비용 피처 레버는 사실상 소진.** 10절 진단("채점대상 60.5%가 오차율 8% 초과")대로 문제의 핵심이 NWP 예보 자체의 정확도 한계일 가능성이 높음. 다음은 (a) 더 비용이 큰 피처(LDAPS gust_ratio 등, 원시데이터 재처리 필요)를 시도할지 (b) 여기서 탐색을 마무리하고 최종 제출 준비로 넘어갈지 민석님과 논의 필요

## 이번 세션에서 한 것 (2026-07-22, 계속 — 12절 중간 풍속 구간 피처 보강 신규 작성)
- 11절(group_3 전용 대응)이 개선 없음으로 끝난 뒤, "다음 할 일" 2순위인 중간 풍속 구간(6.6~8.3 m/s) 개선으로 이동
- v1/v2 피처 대조로 실제 드랍된 피처가 `gfs_shear_alpha` 하나뿐임을 확인(`허브풍속_파생(ws_hub_gfs)`은 `gfs_ws100m`과 상관 0.9999로 완전 중복이라 애초에 재검토 불필요, 04번 기록 재확인)
- 저비용 후보 2가지로 `05_tuning.ipynb`에 **12절 신규 작성**(`wind-domain-features`/`leakage-guard`/`model-tuning` 스킬 기준, AI가 코드만 작성, 10개 셀 추가로 총 77개, **미실행**):
  - 12-1: v1 parquet에서 `gfs_shear_alpha`를 시각 기준 병합 + 그룹별 LDAPS·공유 GFS 풍향×풍속 상호작용 8개 컬럼 생성(parquet 재저장 없이 노트북 내 메모리에서만, 실험 단계이므로)
  - 12-2: baseline(50)/+상호작용(58)/+shear(51)/+둘다(59) 4개 피처셋을 3-fold CV(최종 확정 설정: τ=0.70+actual가중)로 비교
  - 12-3: 최고 조합이 baseline을 이겼을 때만 seed 3개(42/7/2024) 재검증(11절과 같은 원칙 — 단일 CV 개선은 안 믿는다)
  - 12-4: 승자 피처셋으로 10-2와 같은 방식의 풍속 5분위 구간별 오차표 재생성 — baseline과 나란히 비교해 문제의 6.6~8.3 m/s 구간에서 실제로 줄었는지 확인
  - 12-5: 종합 해석은 빈 자리, 실행 결과 받은 뒤 채울 예정
- **재실행 시간 단축**: 12-4가 10절 `scored_df`를 재사용하던 걸 self-contained(자체 baseline/승자 OOF 재계산)로 수정 — 이제 12절만 돌리려면 **0~4절 + 9-2/9-3(τ 스윕·seed 확인)만 있으면 충분**, 5~8절(Optuna)·9-1/9-4~9-8·10절·11절은 전부 건너뛰어도 됨(민석님이 "매번 실행하니까 오래 걸려"라고 지적해서 반영)
- `reports/05_tuning.md`에 13번(노트북 12절) Why/How 추가, Result/So-what은 실행 결과 대기
- **다음 세션(또는 이어서) 할 일**: 민석님이 12-1~12-4 실행 → 결과를 AI에게 전달하면 12-5 해석과 리포트 Result/So-what 채움. 개선 확인되면 채택된 피처를 `03_features.ipynb`/`04_model_selection.ipynb`에 정식 반영해 v3 parquet 생성 후 `train.ipynb`/`inference.ipynb` 재실행·재제출

## 이번 세션에서 한 것 (2026-07-22, 계속 — 11절 group_3 전용 대응 실험 작성·실행·결론)
- 첫 제출(0.62256) 이후 "다음 할 일" 1순위였던 group_3 대응을 `05_tuning.ipynb`에 **11절로 신규 작성**(`model-tuning`/`leakage-guard` 스킬 기준, AI가 코드 작성, 12개 셀 추가로 총 67개) → **민석님이 즉시 실행 완료**
  - 11-1: `metric()`을 그룹별 NMAE/FICR 분해 단계까지 복제한 헬퍼(`compute_group_metrics`/`combined_score`) + `metric()`과 재조합 결과 일치 검산(0.603306 vs 0.603306, 소수점 6자리까지 일치 — `src/metric.py`는 손대지 않음)
  - 11-2: 학습·예측 분리 헬퍼(`train_fold_model`/`predict_group`) — group_1/2는 baseline 모델, group_3만 다른 모델로 섞어 예측할 수 있게 설계
  - 11-3: group_3 표본 가중치 배수 실험(1.0~5.0) — **baseline(w3=1.0) 0.6219 → 최고(w3=2.0) 0.6225, +0.0006.** w3=3.0/5.0에서는 다시 하락(U자)
  - 11-4: group_3 전용 τ 실험(0.60~0.85, 풀링 구조 유지 — 04번 풀링 효과 0.5971 vs 0.5868을 잃지 않도록 그룹별 완전분리 대신 예측만 교체) — **baseline(τ3=0.70) 0.6219 → 최고(τ3=0.75) 0.6224, +0.0005.** τ3가 오를수록 group_3 NMAE는 계속 악화(0.145→0.192), FICR은 0.75까지만 개선 후 하락(전형적 trade-off)
  - **11-5(핵심): 더 나았던 w3=2.0을 seed 3개(42/7/2024)로 재검증 → 평균 0.6213 = baseline(tau_seed_mean 0.6213)과 완전히 동일, 개선폭 +0.0000(표준편차 0.0013의 0배).** 단일 seed에서 보였던 +0.0006은 순수 seed 노이즈였음이 확인됨(8절 Optuna 튜닝의 +0.000045 "효과 미미" 패턴과 동일)
  - 11-6 종합 해석 작성 완료: **group_3 전용 대응(표본 가중치·전용 τ 둘 다)으로는 유의미한 개선을 찾지 못함** — 10절 오류분석 결론("채점대상 60.5%가 오차율 8% 초과라 미세조정으로는 부족, 예보 자체 오차를 줄여야 함")과 정확히 일치
- `reports/05_tuning.md`에 12번(노트북 11절) Why/How/Result/So-what 전부 실제 수치로 완성
- **결론: 최종 확정 모델·설정은 변경 없음** (통합모델 CatBoost + τ=0.70 + actual 가중, 표본 가중치 없음) — `train.ipynb`/`inference.ipynb` 재실행·재제출 불필요
- **다음 우선순위는 "다음 할 일" 2번(중간 풍속 구간 6.6~8.3 m/s 특화 피처 보강)으로 이동** — τ·가중치 조정 여지는 9절+11절로 사실상 소진됨을 확인했으니, `model-tuning` 스킬 4절 원칙대로 이제 피처/구조 쪽에 시간 투자

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
1. **[다음 최우선] 그룹별 τ 재검토(CatBoost)**: `05_tuning.ipynb` 15절 신규 작성 필요(민석님이 "학교 가서 하겠다"고 보류, 아직 코드 없음) — 외부 파이프라인은 group_1/2/3에 각각 다른 τ(0.70/0.50/0.65)를 썼는데 우리는 항상 공통 τ=0.70만 씀. 그룹별 τ 그리드(0.50~0.75) 독립 탐색 → seed 재검증 → CatBoost+MLP 블렌드 가중치 재탐색까지 이어서. 기존 함수(`train_fold_model`/`predict_group`/`single_group_score`) 재사용 가능
2. (후순위, 아직 미착수) 피처셋 크기(50개 vs 외부 179개) 차이가 격차의 또 다른 원인일 가능성 — 그룹별 τ 실험 이후 검토
3. ~~group_3 전용 대응~~ — **완료, 개선 없음으로 결론**(11절 seed 재검증 결과 +0.0000). 재시도할 필요 없음
4. ~~중간 풍속 구간 피처 보강~~ — **완료, 개선 없음으로 결론**(상호작용·shear_alpha 둘 다 baseline보다 손해). 재시도할 필요 없음
5. ~~미사용 원시변수(LDAPS 5m 경계층바람)~~ — **완료, 개선 없음으로 결론**(blws_mag/ratio 모두 baseline보다 손해). 재시도할 필요 없음
6. ~~개선 탐색 마무리 제안~~ — **철회됨**. 민석님이 딥러닝 실측 성공 사례(외부 문서)를 근거로 반박 → AI가 판단 수정, 14절(MLP+미분가능 FICR손실) 진행
7. ~~MLP+미분가능 FICR손실 실행~~ — **완료, 진짜 개선 확인**(CatBoost+MLP 그룹별 블렌드 CV 0.6295, +0.0082, 표준편차의 6.2배). 최종 채택안 변경됨
8. ~~`train.ipynb`/`inference.ipynb`에 반영~~ — **완료**. 실제 실행·제출·리더보드 확인까지 완료(Score 0.625596, 1차 대비 +0.00304)
9. (선택, 우선순위 낮음) τ=0.70 + 8절 `BEST_PARAMS`(Optuna 튜닝 하이퍼파라미터) 조합 재검증 — 8절 자체 효과가 노이즈 수준이었어서 큰 변화는 기대 안 하지만 시너지 여부만 가볍게 확인
10. 제출할 때마다 `HANDOFF.md` 실험 기록표에 로컬 CV/리더보드 점수를 함께 기록
11. `gfs_ws850hpa`(permutation importance 1위, 상층풍)와 관련된 추가 파생 피처(온도이류·지균풍 등)를 `wind-domain-features` 관점에서 검토할 가치 있음 (우선순위는 위 1~8번 이후)
12. `notebooks/01_preprocessing.ipynb`~`04_model_selection.ipynb`가 이번 세션 시작 시점부터 이미 `git status`에 수정된 상태로 잡혀 있었음(이번 세션에서 AI가 건드리지 않음) — 커밋 전에 어떤 변경인지(출력 갱신뿐인지) 확인 필요
13. 여유 있으면 `notebooks/03_features.ipynb`도 Restart & Run All로 독립 재현 확인(우선순위 낮음, 이미 내부 정합성은 확인됨)

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
| 2026-07-23 | CatBoost(위와 동일) + MLP(`soft_metric_loss`, T_soft=0.003, seed 앙상블) 그룹별 블렌드(g1=0.4/g2=0.5/g3=1.0, MLP만 씀) — `20260723_v2_catboost_mlp_groupblend_quantile070.csv` | 0.6295 (3-fold CV, 표준편차 0.0013, 1차 대비 +0.0082) | **0.625596** (1-NMAE 0.857136 / FICR 0.394057) | **1차 제출 대비 +0.00304 — 실제 리더보드에서도 개선 재현**(1-NMAE +0.00238, FICR +0.00371 모두 개선). 단 CV가 예측한 개선폭(+0.0082)의 약 37%만 실현 — 로컬 CV(2022~2024)와 실제 test(2025)의 분포 차이 때문으로 추정(`inference.ipynb` 5-1절: 2025년 풍속이 2024년과 다름을 이미 확인). CV는 방향·상대비교는 신뢰할 수 있으나 개선폭의 절대 크기는 낙관적으로 나올 수 있음 — 향후 판단 시 참고 |

## 미해결 질문
- **group_3 데이터 부족 문제 — 10절 오류분석으로 실제 약점임이 확인됨(8% 초과 비율 66.0% vs 다른 그룹 57~58%). 11절에서 표본가중치·전용 τ 둘 다 시도했으나 seed 재검증에서 개선 없음(+0.0000) — 손실함수·가중치 조정으로는 해결 불가로 잠정 결론.** 남은 대응 방향은 피처 보강이나 group_3 데이터 자체를 늘리는 방법(현재로선 미제공)뿐, 후자는 손댈 수 없어 우선순위 낮춤
- 2022-10 라벨 결측(group_1/2 82개)의 정확한 원인 (터빈 점검 기록 등 외부 확인 불가하므로 결측 그대로 두고 학습 제외하는 것으로 잠정 결론)
- fold별 `best_iteration_` 편차가 큼(356/518/1170) — 평균×1.05(716) 대신 더 큰 값을 쓰는 게 나을지는 검증 안 됨(재학습은 검증셋이 없어 직접 비교 불가하지만, 참고 삼아 다음에 재검토 가능)

## 환경 메모
- venv Python 3.13.14, pandas 3.0.3, numpy 2.5.1, catboost 1.2.10 확인(`train.ipynb` 셋업 셀 실제 출력 기준)
- REPO_ROOT가 세션 초반엔 `c:\Users\cho03\Desktop\wind_forecast`로 잡혔다가 이번 세션 `train.ipynb` 실행 시엔 `d:\공모전\wind_forecast`로 확인됨 — 폴더 위치가 바뀐 것으로 보이나 노트북들이 `Path.cwd()` 기준으로 REPO_ROOT를 자동 탐색하도록 작성돼 있어 실행에는 문제없음(참고만 해둘 것)
- Jupyter 커널 등록 완료: `wind_forecast (venv)` (커널 경로 문제 해결됨, 확인 완료)
