from __future__ import annotations

import math

import pandas as pd
import pytest

from src.analysis.spc_analysis import calculate_spc_statistics


def test_imr_limits_use_moving_range() -> None:
    # MR = 1, 1, 1, 1 → MR̄ = 1, σ_within = 1/1.128
    stats = calculate_spc_statistics(pd.Series([1, 2, 3, 4, 5]), lsl=0, usl=6)
    sigma_within = 1 / 1.128
    assert stats["mr_bar"] == pytest.approx(1.0)
    assert stats["sigma_within"] == pytest.approx(sigma_within)
    assert stats["ucl"] == pytest.approx(3 + 3 * sigma_within)
    assert stats["lcl"] == pytest.approx(3 - 3 * sigma_within)
    assert stats["cp"] == pytest.approx(6 / (6 * sigma_within))
    assert stats["cpk"] == pytest.approx(3 / (3 * sigma_within))


def test_pp_ppk_use_overall_sigma() -> None:
    stats = calculate_spc_statistics(pd.Series([1, 2, 3, 4, 5]), lsl=0, usl=6)
    sigma_overall = math.sqrt(2.5)
    assert stats["sigma_overall"] == pytest.approx(sigma_overall)
    assert stats["standard_deviation"] == pytest.approx(sigma_overall)
    assert stats["pp"] == pytest.approx(6 / (6 * sigma_overall))
    assert stats["ppk"] == pytest.approx(3 / (3 * sigma_overall))


def test_trend_widens_overall_sigma_not_control_limits() -> None:
    # 꾸준한 상승 추세: 전체 σ는 커지지만 이동범위는 작습니다.
    values = pd.Series([float(i) for i in range(20)])
    stats = calculate_spc_statistics(values, lsl=-10, usl=30)
    assert stats["sigma_within"] < stats["sigma_overall"]
    assert stats["cpk"] > stats["ppk"]
    assert stats["control_out_count"] > 0


def test_cpk_uses_nearer_specification() -> None:
    stats = calculate_spc_statistics(pd.Series([8, 9, 8, 9, 8]), lsl=0, usl=10)
    assert stats["cpk"] == pytest.approx(
        (10 - stats["mean"]) / (3 * stats["sigma_within"])
    )


def test_counts_specification_and_control_violations_separately() -> None:
    values = pd.Series([5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 9.0])
    stats = calculate_spc_statistics(values, lsl=4.0, usl=8.0)
    assert stats["specification_out_count"] == 1
    assert stats["control_out_count"] >= 1


def test_ignores_non_numeric_values() -> None:
    stats = calculate_spc_statistics(
        pd.Series([1, None, "x", 2, 3]), lsl=0, usl=4
    )
    assert stats["data_count"] == 3


def test_constant_values_return_zero_capability() -> None:
    stats = calculate_spc_statistics(pd.Series([2.0, 2.0, 2.0]), lsl=1, usl=3)
    assert stats["cp"] == 0.0
    assert stats["ppk"] == 0.0


@pytest.mark.parametrize(
    "values, lsl, usl",
    [
        (pd.Series([1.0]), 0, 2),
        (pd.Series([None, "x"]), 0, 2),
        (pd.Series([1.0, 2.0]), 3, 3),
    ],
)
def test_rejects_invalid_input(values: pd.Series, lsl: float, usl: float) -> None:
    with pytest.raises(ValueError):
        calculate_spc_statistics(values, lsl=lsl, usl=usl)
