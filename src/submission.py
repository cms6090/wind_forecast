"""
제출 파일 생성 + 검증.

CLAUDE.md 7번 섹션의 "제출 파일 규칙 체크리스트"를 코드로 강제하기 위한 모듈이다.
노트북(주로 inference.ipynb)에서는 이 모듈의 build_submission()과
validate_submission()을 import해서 쓰고, 같은 로직을 셀에 복사하지 않는다
(CLAUDE.md 8번 규칙).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.metric import CAPACITY_KWH, TARGET_COLS

# sample_submission.csv의 정해진 컬럼 순서. 이 순서를 벗어나면 채점 시스템이
# 값을 잘못 읽을 수 있으므로 build_submission()의 출력은 항상 이 순서를 따른다.
SUBMISSION_COLS = ["forecast_id", "forecast_kst_dtm"] + TARGET_COLS

# 대회에서 요구하는 제출 파일의 행 수 = 2025년 1년치 시간 수 (365일 x 24시간)
N_SUBMISSION_ROWS = 8760

# CLAUDE.md 7번 체크리스트가 요구하는 시각 문자열 형식: "YYYY-MM-DD HH:MM:SS"
DTM_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def build_submission(pred_df: pd.DataFrame, dtm_col: str, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    예측 결과를 sample_submission 형식에 맞춰 병합한다.

    입력:
        pred_df  : 예측값 DataFrame. dtm_col(시각) 컬럼과 TARGET_COLS
                   (kpx_group_1/2/3, kWh) 컬럼을 포함해야 한다. 행 순서는 상관없다.
        dtm_col  : pred_df에서 시각을 담은 컬럼 이름 (예: "forecast_kst_dtm").
        sample_df: data/sample_submission.csv를 그대로 읽은 DataFrame
                   (forecast_id, forecast_kst_dtm 원본 문자열을 그대로 보존하기 위해
                   경로가 아니라 이미 로딩된 DataFrame을 받는다 — 경로 하드코딩 방지).

    출력:
        sample_df와 같은 8,760행 순서를 유지한 DataFrame.
        forecast_id/forecast_kst_dtm은 sample_df의 원본 문자열 그대로이고,
        kpx_group_1/2/3만 pred_df에서 가져온 값으로 채워진다.

    주의 (CLAUDE.md 7번 규칙):
        - 시각 매칭은 문자열이 아니라 datetime으로 파싱해 merge한다
          (행 순서가 같다고 가정하고 그냥 이어 붙이면, pred_df가 다른 순서로
          정렬돼 있을 때 값이 엉뚱한 시각에 붙는 사고가 난다).
        - 병합 후에는 반드시 clip(0, capacity)으로 음수·설비용량 초과를 제거한다
          (평가 산식은 물리적으로 불가능한 예측을 걸러주지 않는다).
    """
    sample = sample_df.copy()
    pred = pred_df.copy()

    # 원본 문자열은 그대로 보존하고, 병합 전용으로 datetime 파싱한 컬럼을 따로 만든다
    # (CLAUDE.md: "저장 시 원본 문자열을 그대로 보존하는 게 가장 안전")
    sample["_dtm_parsed"] = pd.to_datetime(sample["forecast_kst_dtm"])
    pred["_dtm_parsed"] = pd.to_datetime(pred[dtm_col])

    merged = sample.merge(
        pred[["_dtm_parsed"] + TARGET_COLS],
        on="_dtm_parsed",
        how="left",
        suffixes=("_orig", ""),
    )

    missing = merged[TARGET_COLS].isna().any(axis=1).sum()
    if missing > 0:
        raise ValueError(
            f"병합 후 예측값이 비어 있는 시각이 {missing}개 있습니다. "
            "pred_df의 시각 범위가 2025-01-01 01:00 ~ 2026-01-01 00:00을 "
            "모두 커버하는지 확인하세요."
        )

    for col in TARGET_COLS:
        # 물리적으로 불가능한 값(음수, 설비용량 초과) 제거 — CLAUDE.md 5번 후처리 기본기
        merged[col] = merged[col].clip(lower=0, upper=CAPACITY_KWH[col])

    return merged[SUBMISSION_COLS]


def save_submission(submission_df: pd.DataFrame, out_path: str) -> None:
    """
    제출 파일을 저장한다. utf-8-sig 인코딩 + index 없이 저장 (CLAUDE.md 7번 규칙).

    입력:
        submission_df: SUBMISSION_COLS 순서의 DataFrame (build_submission 출력).
        out_path     : 저장 경로 (예: "submissions/submission_exp001.csv").
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def validate_submission(path: str, sample_df: pd.DataFrame = None, raise_on_error: bool = True):
    """
    저장된 제출 파일이 CLAUDE.md 7번 체크리스트를 통과하는지 검사한다.
    모든 제출 파일은 생성 직후 이 함수로 검증해야 한다.

    입력:
        path          : 검사할 제출 파일 경로.
        sample_df     : (선택) sample_submission.csv를 읽은 DataFrame. 주어지면
                         forecast_id/forecast_kst_dtm이 원본과 완전히 같은지도 검사한다.
        raise_on_error: True면 문제가 있을 때 AssertionError를 발생시켜 노트북
                        실행을 즉시 중단시킨다 (기본값, 실수를 놓치지 않기 위함).
                        False면 중단하지 않고 문제 목록(list[str])만 반환한다.

    출력:
        issues: list[str]. 비어 있으면 모든 검사를 통과했다는 뜻.

    검사 항목 (CLAUDE.md 7번):
        1) 컬럼 순서가 정확히 forecast_id, forecast_kst_dtm, kpx_group_1/2/3인가
        2) 행 수가 정확히 8,760인가
        3) 시각 문자열이 "YYYY-MM-DD HH:MM:SS" 형식인가 (Excel 재저장 등으로
           형식이 깨지면 이 정규식에서 걸린다)
        4) forecast_id/forecast_kst_dtm이 sample_submission과 완전히 동일한가
           (sample_df가 주어졌을 때만)
        5) 예측값 3개 컬럼에 결측(NaN)이 없는가
        6) 예측값에 음수가 없는가
        7) 예측값이 그룹별 설비용량(21,600 / 21,600 / 21,000 kWh)을 넘지 않는가
    """
    issues = []
    df = pd.read_csv(path, encoding="utf-8-sig")

    if df.columns.tolist() != SUBMISSION_COLS:
        issues.append(f"컬럼 순서/이름이 다릅니다: {df.columns.tolist()} (기대값: {SUBMISSION_COLS})")

    if len(df) != N_SUBMISSION_ROWS:
        issues.append(f"행 수가 {len(df)}개입니다 (기대값: {N_SUBMISSION_ROWS}개)")

    if "forecast_kst_dtm" in df.columns:
        bad_fmt = ~df["forecast_kst_dtm"].astype(str).str.match(DTM_PATTERN)
        if bad_fmt.any():
            issues.append(
                f"시각 형식이 'YYYY-MM-DD HH:MM:SS'가 아닌 행이 {bad_fmt.sum()}개 있습니다 "
                "(Excel로 열어 재저장하지 않았는지 확인하세요)"
            )

    if sample_df is not None:
        for id_col in ["forecast_id", "forecast_kst_dtm"]:
            if id_col in df.columns and id_col in sample_df.columns:
                if not df[id_col].reset_index(drop=True).equals(sample_df[id_col].reset_index(drop=True)):
                    issues.append(f"'{id_col}' 컬럼이 sample_submission.csv 원본과 다릅니다")

    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        if np.isnan(values).any():
            issues.append(f"'{col}'에 결측(NaN)이 {np.isnan(values).sum()}개 있습니다")
        if (values < 0).any():
            issues.append(f"'{col}'에 음수 값이 {(values < 0).sum()}개 있습니다")
        capacity = CAPACITY_KWH[col]
        if (values > capacity).any():
            issues.append(f"'{col}'에 설비용량({capacity} kWh) 초과 값이 {(values > capacity).sum()}개 있습니다")

    if raise_on_error and issues:
        raise AssertionError("제출 파일 검증 실패:\n- " + "\n- ".join(issues))

    return issues
