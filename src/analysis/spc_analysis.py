"""LOT 단위 SPC 통계 계산.

LOT 하나가 측정값 하나인 데이터이므로 I-MR(Individual-Moving Range)
관리도를 사용합니다.

- 관리한계(UCL/LCL)와 Cp/Cpk: 군내 표준편차 σ_within = MR̄ / d2
  (d2 = 1.128, 이동범위 크기 2)
- Pp/Ppk: 전체 표준편차 σ_overall (표본 표준편차, ddof=1)

관리한계는 공정이 스스로 내는 변동으로 정하고, 규격한계(LSL/USL)는
고객·설계 요구로 정합니다. 두 기준은 의미가 다르므로 섞지 않습니다.
값은 시간 순서대로 들어와야 합니다. 이동범위가 순서에 따라 달라집니다.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

D2_MOVING_RANGE_2: Final[float] = 1.128
SIGMA_LIMIT: Final[float] = 3.0


def _grade(cpk: float) -> str:
    if cpk >= 1.67:
        return "Excellent"
    if cpk >= 1.33:
        return "Capable"
    if cpk >= 1.0:
        return "Marginal"
    return "Improvement Required"


def _capability(
    mean_value: float,
    sigma: float,
    lsl: float,
    usl: float,
) -> tuple[float, float]:
    if sigma <= 0:
        return 0.0, 0.0
    spread = (usl - lsl) / (6 * sigma)
    upper = (usl - mean_value) / (3 * sigma)
    lower = (mean_value - lsl) / (3 * sigma)
    return float(spread), float(min(upper, lower))


def calculate_spc_statistics(
    values: pd.Series,
    lsl: float,
    usl: float,
) -> dict[str, float | int | str]:
    """시간 순서 측정값으로 I-MR 관리한계와 공정능력을 계산합니다.

    `standard_deviation`은 화면 호환을 위해 σ_overall을 그대로 둡니다.
    관리한계와 Cp/Cpk는 σ_within을 씁니다.
    """

    if lsl >= usl:
        raise ValueError("LSL은 USL보다 작아야 합니다.")

    numeric = (
        pd.to_numeric(values, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )

    if len(numeric) < 2:
        raise ValueError(
            "I-MR 관리도는 분석 가능한 LOT가 2개 이상 필요합니다."
        )

    mean_value = float(numeric.mean())
    sigma_overall = float(numeric.std(ddof=1))
    mr_bar = float(numeric.diff().abs().dropna().mean())
    sigma_within = mr_bar / D2_MOVING_RANGE_2

    ucl = mean_value + SIGMA_LIMIT * sigma_within
    lcl = mean_value - SIGMA_LIMIT * sigma_within

    cp, cpk = _capability(mean_value, sigma_within, lsl, usl)
    pp, ppk = _capability(mean_value, sigma_overall, lsl, usl)

    specification_out_count = int(
        ((numeric < lsl) | (numeric > usl)).sum()
    )
    control_out_count = int(
        ((numeric < lcl) | (numeric > ucl)).sum()
    )

    return {
        "mean": mean_value,
        "standard_deviation": sigma_overall,
        "sigma_overall": sigma_overall,
        "sigma_within": sigma_within,
        "mr_bar": mr_bar,
        "ucl": ucl,
        "lcl": lcl,
        "cp": cp,
        "cpk": cpk,
        "pp": pp,
        "ppk": ppk,
        "specification_out_count": specification_out_count,
        "control_out_count": control_out_count,
        "process_grade": _grade(cpk),
        "data_count": len(numeric),
    }
