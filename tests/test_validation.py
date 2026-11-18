from __future__ import annotations

import pytest

from src.analysis.validation import _independent_grade, run
from src.detection.rule_detector import classify_range, classify_value

PUMP = {
    "normal_min": 1.95,
    "normal_max": 2.25,
    "warning_min": 1.80,
    "warning_max": 2.40,
    "critical_min": 1.65,
    "critical_max": 2.55,
}


@pytest.mark.parametrize(
    "value, expected",
    [
        (2.10, "NORMAL"),
        (2.30, "WARNING"),  # 경고 범위 안
        (2.45, "WARNING"),  # 경고 한계와 위험 한계 사이 (이전에는 WATCH)
        (1.70, "WARNING"),
        (2.60, "CRITICAL"),
        (1.60, "CRITICAL"),
    ],
)
def test_range_grade_never_improves_away_from_normal(value: float, expected: str) -> None:
    assert classify_range(value, PUMP) == expected


def test_range_grade_is_monotonic() -> None:
    order = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3}
    upward = [2.25 + step * 0.01 for step in range(40)]
    grades = [order[classify_range(value, PUMP)] for value in upward]
    assert grades == sorted(grades)


@pytest.mark.parametrize(
    "limits",
    [
        {"direction": "lower_is_better", "normal_max": 1.2, "watch_max": 1.49,
         "warning_max": 1.99, "critical_min": 2.0},
        {"direction": "higher_is_better", "normal_min": 17.0, "watch_min": 16.5,
         "warning_min": 15.5, "critical_max": 15.49},
        {"direction": "higher_is_better", "normal_min": 96.0, "warning_min": 94.0,
         "critical_min": 92.0},
        PUMP,
    ],
)
def test_independent_grade_matches_detector(limits: dict) -> None:
    bounds = [v for k, v in limits.items() if k != "direction"]
    low, high = min(bounds) - 1.0, max(bounds) + 1.0
    for step in range(1001):
        value = round(low + (high - low) * step / 1000, 4)
        assert _independent_grade(value, limits) == classify_value(value, limits)


def test_all_independent_checks_pass_on_committed_data() -> None:
    checks, model = run()
    failed = [check for check in checks if not check.passed]
    assert failed == []
    assert 0.0 <= model["majority_accuracy"] <= 1.0
