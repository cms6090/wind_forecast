"""
대회 산식을 직접 최적화하는 신경망 (MLP).

`train.ipynb`(학습)와 `inference.ipynb`(추론)가 **똑같은 구조·똑같은 전처리**를 쓰도록
공통 로직을 이 모듈에 모았다 (CLAUDE.md 12-3: "공통 로직은 src/의 함수로 두고 양쪽에서 import").
두 노트북에 신경망 정의를 복사해 두면 한쪽만 고치는 사고가 난다.

---

## 왜 신경망인가 (reports/phase6_nn_metric_loss.md에 자세히)

Phase 3~5에서 LightGBM으로 할 수 있는 것을 다 했고, 이후 12개 가설(피처 추가, 원시 격자,
앙상블, 트리 수, 결정이론적 점예측 등)이 전부 교차검증에서 기각됐다.
이유는 하나였다: **예보 피처의 결정론적 함수를 아무리 추가해도 트리에게는 새 정보가 아니다.**

남은 것은 "정보를 늘리는 것"이 아니라 **"목표를 바꾸는 것"** 이었다.

## 산식을 손실함수로 (핵심 아이디어)

대회 점수(그룹 하나, 이용률 단위):

    score = 0.5*(1 - mean|ŷ - y|) + 0.5 * Σ y·p(e) / (4 Σ y),   e = |ŷ - y|
    p(e) = 4 (e<=0.06), 3 (e<=0.08), 0 (그 밖)      <- 계단 함수라 미분 불가

계단 p를 시그모이드 두 개로 매끄럽게 바꾼다:

    p_soft(e) = 3·σ((0.08 - e)/T) + σ((0.06 - e)/T)

    e ≪ 0.06  -> 3 + 1 = 4  ✔
    0.06<e<0.08 -> 3 + 0 = 3  ✔
    e > 0.08  -> 0 + 0 = 0  ✔

이제 `-score`를 손실로 쓰면 경사하강법이 **대회 점수 자체를 최대화**한다.

**이 손실이 하는 일**: 어떤 샘플이 6% 밴드에 들어갈 가망이 있으면 그쪽으로 강하게 당기고,
가망이 없으면 그냥 L1처럼 다룬다. 시각마다 다르게 행동한다.
분위수 τ를 올려 **모든 시각을 똑같이** 위로 미는 방식과 근본적으로 다르다.

**왜 GBDT로는 안 되는가**: 부스팅은 잎 값을 `-Σg/Σh`로 정한다.
비볼록한 밴드 보너스가 만든 기울기를 그 규칙이 제대로 처리하지 못한다
(LightGBM 내장 L1은 잎 값을 잔차의 중앙값으로 다시 계산하는 특수 보정이 있지만,
커스텀 목적함수에는 그 보정이 없다). 실제로 시도했다가 CV가 0.61 -> 0.56으로 무너졌다.
경사하강법에는 그런 제약이 없다.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 상수 (근거는 reports/phase6_nn_metric_loss.md)
# ---------------------------------------------------------------------------
# T_SOFT — 계단을 부드럽게 하는 폭. **이 값은 "기본값"일 뿐이고 실제 학습에서는 호출부가 정한다.**
#
# `train_mlp()`가 `t_soft` 인자를 받으므로 `train.ipynb`는 자기 CONFIGS의 값을 명시적으로 넘긴다.
# 여기 상수를 그대로 두는 이유는 두 가지다.
#   (1) 아래 0.006은 **exp017(Public 0.6388632247) 확정 모델이 실제로 쓴 값**이다.
#       `06_feature_ablation.ipynb`의 칸 A가 이 상수를 import해 그 모델을 재현하므로,
#       값을 바꾸면 과거 실험의 재현이 조용히 깨진다.
#   (2) 추론(`inference.ipynb`)은 학습을 하지 않아 이 값을 쓰지 않는다.
#
# 격자 탐색 이력 (2024 홀드아웃 하네스, `reports/06_feature_ablation.md`):
#   1차(구v5)  0.004 / 0.006 / 0.010     -> 0.006
#   2차(06 §5) 0.003 / 0.006 / 0.012 / 0.025 -> 0.003 (**격자 최솟값 = 경계에서 멈춤**)
#   3차(06 §12) 0.0005 / 0.001 / 0.002 / 0.003 / 0.004 -> **0.001에서 봉우리가 갇힘**
#       (0.0005에서 다시 내려간다 — T가 작아질수록 기울기를 받는 샘플이 밴드 경계 ±T 창
#        안으로만 좁아지기 때문. 아래 soft_price 주석 참고)
#   단, 내부검증 개선(+0.0045)의 홀드아웃 실현율은 11%에 그쳤다 -> A'(0.003)와 A''(0.001)를
#   둘 다 제출해 리더보드로 가르는 중이다.
T_SOFT = 0.006
HIDDEN = 256            # 은닉층 폭
DROPOUT = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 400
EVAL_EVERY = 5          # 몇 에폭마다 검증 점수를 볼지
PATIENCE = 60           # 이만큼 개선이 없으면 조기 종료

# FICR 밴드 경계 (src/metric.py의 산식과 동일한 값)
BAND_FULL = 0.06        # 이 이내면 단가 4 (만점)
BAND_PART = 0.08        # 이 이내면 단가 3


def set_seed(seed: int) -> None:
    """
    파이썬·numpy·torch의 난수를 모두 고정한다 (CLAUDE.md 12-3: 재현성).

    `use_deterministic_algorithms(True)`는 비결정적인 CUDA/CPU 커널 사용을 막는다.
    이 프로젝트는 CPU만 쓰므로(모델이 작다) 이것만으로 비트 단위 재현이 된다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    """
    n_in개 피처 -> [hidden] x n_layers -> 1. 출력은 sigmoid로 [0, 1](이용률)에 가둔다.

    출력을 sigmoid로 가두는 이유: 발전량은 0 미만이나 설비용량 초과가 물리적으로 불가능하다.
    모델이 애초에 그 범위 밖을 예측하지 못하게 하면 학습이 쉬워진다.
    (LightGBM은 이런 제약을 못 걸어서 나중에 clip으로 잘라내야 한다.)

    입력: (배치, n_in) float32 텐서. **반드시 표준화된 값**이어야 한다 (standardize 참조).
    출력: (배치,) 이용률 예측 [0, 1]

    **하위 호환**: `n_layers=2`(기본값)이면 층 구성이 예전과 완전히 같아
    `state_dict` 키(`net.0.*`, `net.4.*`, `net.8.*`)까지 동일하다.
    즉 이 변경으로 기존에 저장된 가중치를 그대로 읽을 수 있다.
    학습 데이터 규모가 달라지면 적정 폭·깊이도 달라지므로(예: g3 통합 학습은 표본이 5.5배)
    호출부가 인자로 넘겨 탐색할 수 있게 열어 둔 것이다.
    """

    def __init__(self, n_in: int, hidden=HIDDEN, p_drop: float = DROPOUT,
                 n_layers: int = 2):
        """
        hidden : 정수면 `n_layers`개 층이 모두 그 폭이 된다 (예: 256, n_layers=2 -> [256, 256]).
                 **시퀀스면 층별 폭을 그대로 쓰고 `n_layers`는 무시한다** (예: (512, 256) -> 512 -> 256).
                 좁아지는 구조를 표현하려면 시퀀스로 넘긴다.
        """
        super().__init__()
        widths = [int(hidden)] * n_layers if isinstance(hidden, (int, np.integer)) else [int(w) for w in hidden]
        assert len(widths) >= 1, "은닉층이 최소 하나는 있어야 합니다"
        layers, d = [], n_in
        for w in widths:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(p_drop)]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.widths = tuple(widths)          # config.json에 기록해 추론 때 재구성하는 데 쓴다

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def soft_price(e: torch.Tensor, t_soft: float = T_SOFT) -> torch.Tensor:
    """
    대회 단가 계단함수(4 / 3 / 0)의 미분 가능한 근사.

    입력: e — 오차율 |ŷ - y| (이용률 단위이므로 설비용량으로 나눈 값과 같다)
    출력: 단가 근사값 (약 0 ~ 4)
    공식: 3·σ((0.08 - e)/T) + σ((0.06 - e)/T)
    T가 작을수록 실제 계단에 가깝지만 기울기가 날카로워져 학습이 불안정해진다.
    """
    return 3.0 * torch.sigmoid((BAND_PART - e) / t_soft) + torch.sigmoid((BAND_FULL - e) / t_soft)


def metric_loss(pred: torch.Tensor, y: torch.Tensor, t_soft: float = T_SOFT) -> torch.Tensor:
    """
    대회 산식을 그대로 옮긴 손실함수 (최소화 대상 = -score).

    입력:
        pred : (n,) 예측 이용률 [0,1]
        y    : (n,) 실제 이용률 [0,1]. **채점 대상 행(이용률 >= 10%)만** 넣는다.
    출력: 스칼라 텐서

    L = 0.5 * mean(|ŷ - y|)  -  0.5 * Σ y·p_soft(e) / (4 Σ y)
        └─ NMAE 항 (작을수록 좋음)      └─ FICR 항 (클수록 좋으므로 빼 준다)

    주의: FICR 항은 '전체 합의 비율'이라 미니배치가 작으면 추정이 흔들린다.
          그래서 학습은 전체 배치(full-batch)로 한다. 17,000행 x 179열이면 CPU로 충분하다.
    """
    e = torch.abs(pred - y)
    nmae = e.mean()
    ficr = (y * soft_price(e, t_soft)).sum() / (4.0 * y.sum() + 1e-8)
    return 0.5 * nmae - 0.5 * ficr


def fit_standardizer(X: np.ndarray):
    """
    표준화 통계(평균/표준편차)를 구한다. **반드시 학습 데이터에서만 호출한다** (CLAUDE.md 4번).

    입력: X — (n, d) float 배열
    출력: (mu, sd) 각각 (d,) 배열. sd에는 0으로 나누는 것을 막기 위해 1e-6을 더한다.
    """
    return X.mean(axis=0), X.std(axis=0) + 1e-6


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> torch.Tensor:
    """학습에서 구한 (mu, sd)로 표준화해 텐서로 만든다. test에는 transform만 적용된다."""
    return torch.tensor(((X - mu) / sd).astype(np.float32))


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray, seed: int, n_epochs: int,
              t_soft: float = T_SOFT, eval_fn=None,
              hidden=HIDDEN, p_drop: float = DROPOUT, n_layers: int = 2,
              lr: float = LR, weight_decay: float = WEIGHT_DECAY):
    """
    산식 손실로 MLP를 학습한다.

    입력:
        X_tr    : (n, d) 표준화된 학습 피처 (numpy)
        y_tr    : (n,) 학습 타깃 = 이용률 [0,1]. 채점 대상 행만.
        seed    : 난수 시드
        n_epochs: 학습 에폭 수
        t_soft  : 계단 근사 폭
        eval_fn : (model) -> float. 주어지면 EVAL_EVERY 에폭마다 호출해
                  값이 가장 큰 시점의 가중치를 되돌린다 (조기 종료).
                  None이면 조기 종료 없이 n_epochs를 끝까지 학습한다.
        hidden, p_drop, n_layers, lr, weight_decay :
                  신경망 구조와 최적화 설정. **기본값은 모듈 상수와 같아 하위 호환된다.**
                  `hidden`은 정수(모든 층 같은 폭) 또는 시퀀스(층별 폭, 예: (512, 256))다.
                  학습 데이터 규모가 바뀌면 이 값들의 적정치도 바뀌므로 인자로 열어 뒀다.
                  특히 **`lr`이 중요하다** — 아래처럼 full-batch로 학습하므로
                  배치 크기가 곧 데이터 크기이고, 표본이 몇 배가 되면 기울기의 성격이 달라진다.
    출력:
        (model, best_epoch). eval_fn이 None이면 best_epoch = n_epochs.

    학습 방식: full-batch AdamW + 코사인 스케줄 + 기울기 클리핑.
    full-batch인 이유는 metric_loss의 FICR 항이 '전체 합의 비율'이기 때문이다.
    """
    set_seed(seed)
    Xt = torch.tensor(X_tr.astype(np.float32))
    yt = torch.tensor(y_tr.astype(np.float32))

    model = MLP(X_tr.shape[1], hidden, p_drop, n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_score, best_state, best_epoch, bad = -np.inf, None, n_epochs, 0
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        metric_loss(model(Xt), yt, t_soft).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if eval_fn is not None and (ep % EVAL_EVERY == 0 or ep == n_epochs - 1):
            model.eval()
            s = eval_fn(model)
            if s > best_score:
                best_score, bad, best_epoch = s, 0, ep + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += EVAL_EVERY
                if bad >= PATIENCE:
                    break

    if eval_fn is not None and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def predict_mlp(model: MLP, X: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                capacity: float) -> np.ndarray:
    """
    표준화 -> 예측 -> 이용률을 kWh로 환산한다.

    입력: X (n, d) 원본 피처, (mu, sd) 학습에서 구한 표준화 통계, capacity 설비용량[kWh]
    출력: (n,) 발전량 예측 [kWh], [0, capacity] 범위
    """
    model.eval()
    p = model(standardize(X, mu, sd)).numpy()
    return np.clip(p, 0.0, 1.0) * capacity
