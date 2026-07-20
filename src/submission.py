"""제출 파일(submissions/*.csv) 검증 유틸리티.

노트북에서 사용법:
    from src.submission import validate_submission
    validate_submission("submissions/20260101_v1_baseline.csv")

CLI 사용법:
    python src/submission.py submissions/20260101_v1_baseline.csv
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUBMISSION_PATH = REPO_ROOT / "data" / "sample_submission.csv"

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
ID_COLS = ["forecast_id", "forecast_kst_dtm"]
EXPECTED_COLS = ID_COLS + TARGET_COLS
EXPECTED_ROWS = 8760

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


class SubmissionValidationError(ValueError):
    pass


def validate_submission(filepath, sample_path=SAMPLE_SUBMISSION_PATH, verbose=True):
    """제출 파일이 대회 규격을 만족하는지 검사한다.

    실패 시 SubmissionValidationError를 발생시키며, 어떤 항목이
    왜 실패했는지 메시지에 구체적으로 담는다. 통과 시 True를 반환한다.
    """
    filepath = Path(filepath)

    # UTF-8(-sig)로 읽히는지 확인 (Excel로 열었다 저장하면 인코딩이 깨질 수 있음)
    try:
        sub = pd.read_csv(filepath, encoding="utf-8-sig", dtype={"forecast_id": str})
    except UnicodeDecodeError as e:
        raise SubmissionValidationError(
            f"UTF-8(-sig) 인코딩으로 읽을 수 없습니다: {filepath} ({e})"
        )

    sample = pd.read_csv(sample_path, encoding="utf-8-sig", dtype={"forecast_id": str})

    # 1. 컬럼 구성 확인
    if list(sub.columns) != EXPECTED_COLS:
        raise SubmissionValidationError(
            f"컬럼이 규격과 다릅니다. 기대: {EXPECTED_COLS}, 실제: {list(sub.columns)}"
        )

    # 2. 행 수 확인
    if len(sub) != EXPECTED_ROWS:
        raise SubmissionValidationError(
            f"행 수가 {EXPECTED_ROWS}이어야 하는데 {len(sub)}행입니다."
        )

    # 3. forecast_id / forecast_kst_dtm 불변 확인 (순서 포함)
    if not sub["forecast_id"].equals(sample["forecast_id"]):
        raise SubmissionValidationError(
            "forecast_id가 sample_submission.csv와 다릅니다 (값 또는 순서 변경 금지)."
        )
    if not sub["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise SubmissionValidationError(
            "forecast_kst_dtm이 sample_submission.csv와 다릅니다 (값 또는 순서 변경 금지)."
        )

    # 4. 결측치 확인
    na_counts = sub[TARGET_COLS].isna().sum()
    if na_counts.sum() > 0:
        raise SubmissionValidationError(f"예측값에 결측치가 있습니다:\n{na_counts}")

    # 5. 음수 확인
    negative_counts = (sub[TARGET_COLS] < 0).sum()
    if negative_counts.sum() > 0:
        raise SubmissionValidationError(f"예측값에 음수가 있습니다:\n{negative_counts}")

    # 6. 설비용량 상한 확인 (제출 전 클리핑 누락 방지)
    over_capacity = {
        col: int((sub[col] > CAPACITY_KWH[col]).sum()) for col in TARGET_COLS
    }
    if any(v > 0 for v in over_capacity.values()):
        raise SubmissionValidationError(
            f"예측값이 그룹 설비용량을 초과합니다 (0~설비용량으로 클리핑 필요): {over_capacity}"
        )

    if verbose:
        print(f"[PASS] {filepath.name}: 컬럼/행 수/ID 불변/결측·음수·상한 초과 없음 확인 완료 "
              f"({len(sub)}행)")

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python src/submission.py <제출파일경로>")
        sys.exit(1)
    validate_submission(sys.argv[1])
