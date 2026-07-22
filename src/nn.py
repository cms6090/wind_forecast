"""MLP + 미분 가능한 FICR 손실 (05_tuning.ipynb 14절에서 사용).

CatBoost/LightGBM의 quantile·MAE 손실로는 FICR의 계단 구조(오차율 6%/8% 경계에서
단가가 4원→3원→0원으로 뚝뚝 떨어지는 것)를 직접 겨냥할 수 없다. 이 모듈은 그 계단
함수를 시그모이드로 부드럽게 근사해, 경사하강으로 대회 채점 산식 자체를 직접
최적화하는 MLP를 제공한다. GBDT 커스텀 목적함수는 2차 미분(hessian) 근사가 필요해
이런 손실에서 불안정하지만(리프값 보정이 없으면 무너짐), 신경망은 backprop이라
그 제약이 없다.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 256), dropout: float = 0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 이용률(발전량/설비용량)은 물리적으로 [0, 1] 근방을 벗어날 수 없다.
        # clip이 아니라 sigmoid로 아예 그 범위를 못 벗어나게 한다.
        return torch.sigmoid(self.net(x).squeeze(-1))


def fit_standardizer(x: np.ndarray):
    """train 파트에서만 fit (leakage-guard 원칙)."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def build_features(df: pd.DataFrame, feature_cols, mu=None, sd=None, n_groups: int = 3):
    """group_id를 원-핫으로, 수치 피처는 표준화(mu/sd 없으면 이 데이터로 새로 fit)해 이어붙인다.

    *_diff_prev류 피처는 발표분 첫 시각에 이전 값이 없어 NaN이다(하루 1096개, train.ipynb/
    05_tuning.ipynb 14절에서 확인). CatBoost는 결측을 스스로 처리하지만 MLP는 NaN이 그대로
    전파되므로, "이전 시각 대비 변화를 알 수 없음"을 중립값 0(변화 없음)으로 채운다.
    """
    group_onehot = pd.get_dummies(df["group_id"].astype(int), prefix="grp").reindex(
        columns=[f"grp_{i}" for i in range(n_groups)], fill_value=0
    ).to_numpy(dtype=np.float32)

    num_x = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    if mu is None:
        mu, sd = fit_standardizer(num_x)
    num_x = apply_standardizer(num_x, mu, sd).astype(np.float32)

    x = np.concatenate([num_x, group_onehot], axis=1)
    return x, mu, sd


def soft_metric_loss(pred_util, actual_util, actual_kwh, is_scored, T: float = 0.006):
    """score = 0.5*(1-NMAE) + 0.5*FICR 를 시그모이드로 미분 가능하게 근사한 것의 음수(손실).

    price_soft(e)는 e=0 근처에서 4, e=0.06~0.08 사이에서 3, e>0.08에서 0에 가까워지는
    계단 함수를 시그모이드 두 개의 합으로 부드럽게 흉내낸다. T가 작을수록 실제 계단에
    가깝지만 기울기가 날카로워져 학습이 불안정해진다(T_soft 스윕으로 절충점을 찾는다).
    """
    pred_s = pred_util[is_scored]
    actual_s = actual_util[is_scored]
    kwh_s = actual_kwh[is_scored]

    e = (pred_s - actual_s).abs()
    nmae_term = e.mean()

    price_soft = 3.0 * torch.sigmoid((0.08 - e) / T) + torch.sigmoid((0.06 - e) / T)
    ficr_soft = (kwh_s * price_soft).sum() / (4.0 * kwh_s.sum())

    score = 0.5 * (1.0 - nmae_term) + 0.5 * ficr_soft
    return -score


def true_score(pred_util, actual_util, actual_kwh, is_scored) -> float:
    """미분 불가능한 실제 채점 방식(진짜 계단 함수) — 조기 종료·모니터링 전용."""
    pred_s = pred_util[is_scored].detach().cpu().numpy()
    actual_s = actual_util[is_scored].detach().cpu().numpy()
    kwh_s = actual_kwh[is_scored].detach().cpu().numpy()

    e = np.abs(pred_s - actual_s)
    nmae = float(e.mean())
    price = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], default=0.0)
    ficr = float(np.sum(kwh_s * price) / (4.0 * np.sum(kwh_s)))
    return 0.5 * (1 - nmae) + 0.5 * ficr


def train_mlp(
    fit_X, fit_util, fit_kwh, fit_scored,
    early_X, early_util, early_kwh, early_scored,
    input_dim: int, seed: int = 42, T_soft: float = 0.006,
    hidden=(256, 256), dropout: float = 0.15,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    max_epochs: int = 300, patience: int = 30, verbose: bool = False,
):
    """전체 배치(full-batch)로 학습한다. FICR 항이 "전체 합 대비 비율"이라 미니배치로
    쪼개면 그 비율 추정이 흔들리기 때문이다. 조기 종료는 손실(soft)이 아니라 실제
    채점 방식(true_score)의 검증 성능으로 판단한다 — 최적화 목표와 감시 지표를 분리."""
    set_seed(seed)
    device = torch.device("cpu")

    model = MLP(input_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    fit_X_t = torch.tensor(fit_X, dtype=torch.float32, device=device)
    fit_util_t = torch.tensor(fit_util, dtype=torch.float32, device=device)
    fit_kwh_t = torch.tensor(fit_kwh, dtype=torch.float32, device=device)
    fit_scored_t = torch.tensor(fit_scored, dtype=torch.bool, device=device)

    early_X_t = torch.tensor(early_X, dtype=torch.float32, device=device)
    early_util_t = torch.tensor(early_util, dtype=torch.float32, device=device)
    early_kwh_t = torch.tensor(early_kwh, dtype=torch.float32, device=device)
    early_scored_t = torch.tensor(early_scored, dtype=torch.bool, device=device)

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(fit_X_t)
        loss = soft_metric_loss(pred, fit_util_t, fit_kwh_t, fit_scored_t, T=T_soft)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            early_pred = model(early_X_t)
            val_score = true_score(early_pred, early_util_t, early_kwh_t, early_scored_t)

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch}: val_score={val_score:.4f} (best {best_score:.4f} @ {best_epoch})")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_score, best_epoch


def train_mlp_full(
    fit_X, fit_util, fit_kwh, fit_scored,
    input_dim: int, epochs: int, seed: int = 42, T_soft: float = 0.006,
    hidden=(256, 256), dropout: float = 0.15,
    lr: float = 1e-3, weight_decay: float = 1e-4,
):
    """검증셋 없이 고정된 epoch 수만큼 전체 데이터로 학습한다(early stopping 없음).

    train.ipynb에서 seed 앙상블 최종 모델을 만들 때 쓴다. epoch 수는 CatBoost의
    `FINAL_ITERATIONS`와 같은 방식 — 3-fold CV에서 관측된 `best_epoch`(train_mlp가
    반환하는 값)의 평균×1.05를 미리 정해서 이 함수에 넘긴다.
    """
    set_seed(seed)
    device = torch.device("cpu")

    model = MLP(input_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.tensor(fit_X, dtype=torch.float32, device=device)
    util_t = torch.tensor(fit_util, dtype=torch.float32, device=device)
    kwh_t = torch.tensor(fit_kwh, dtype=torch.float32, device=device)
    scored_t = torch.tensor(fit_scored, dtype=torch.bool, device=device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = soft_metric_loss(pred, util_t, kwh_t, scored_t, T=T_soft)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    return model


def predict_mlp(model, x: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        pred = model(x_t).cpu().numpy()
    return pred
