"""핵심 계산을 다른 방식으로 다시 계산해 대조합니다.

1. SPC: I-MR 관리한계·Cp/Cpk/Pp/Ppk를 NumPy로 따로 계산해 비교
2. 규칙 판정: process_limits.yaml의 구간표로 등급을 다시 매겨
   DB의 anomaly_result와 LOT·항목·등급 단위로 비교
3. 예지보전 특징량·라벨: NumPy 슬라이딩 윈도와 scipy 회귀로
   다시 계산해 저장된 학습 데이터와 비교
4. 예지보전 모델 평가의 한계: 다수 클래스 기준선, 라벨 절단 건수

1~3은 통과/실패, 4는 수치만 보고합니다(통과 기준이 아닌 공개 항목).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from src.analysis.spc_analysis import calculate_spc_statistics

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATABASE_PATH: Final[Path] = PROJECT_ROOT / "database" / "wet_cleaning.db"
LIMITS_PATH: Final[Path] = PROJECT_ROOT / "config" / "process_limits.yaml"
PM_DATASET_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "predictive_maintenance_dataset.csv"
)
PM_METRICS_PATH: Final[Path] = (
    PROJECT_ROOT / "models" / "predictive_maintenance" / "model_metrics.json"
)

TOLERANCE: Final[float] = 1e-9
SPC_PARAMETERS: Final[list[str]] = [
    "yield_percent",
    "particle_count",
    "filter_differential_pressure",
    "flow_rate",
]
# SPC 대조에는 규격값 자체가 아니라 계산식이 맞는지가 중요하므로
# 각 변수의 관측 범위를 감싸는 임의 규격을 씁니다.
PM_SENSORS: Final[list[str]] = [
    "filter_differential_pressure",
    "flow_rate",
    "motor_current",
    "vibration",
    "particle_count",
    "yield_percent",
]
PM_WINDOW: Final[int] = 5
PM_HORIZON: Final[int] = 10


@dataclass
class Check:
    group: str
    name: str
    compared: int
    mismatched: int
    max_abs_error: float | None
    passed: bool
    note: str = ""


# ---------------------------------------------------------------- SPC
def _independent_imr(values: np.ndarray, lsl: float, usl: float) -> dict[str, float]:
    x = values.astype(float)
    n = x.size
    mean = x.sum() / n
    moving_ranges = np.abs(x[1:] - x[:-1])
    sigma_within = moving_ranges.sum() / (n - 1) / 1.128
    sigma_overall = np.sqrt(((x - mean) ** 2).sum() / (n - 1))
    return {
        "mean": mean,
        "ucl": mean + 3 * sigma_within,
        "lcl": mean - 3 * sigma_within,
        "cp": (usl - lsl) / (6 * sigma_within),
        "cpk": min(usl - mean, mean - lsl) / (3 * sigma_within),
        "pp": (usl - lsl) / (6 * sigma_overall),
        "ppk": min(usl - mean, mean - lsl) / (3 * sigma_overall),
    }


def check_spc(lot_data: pd.DataFrame) -> Check:
    compared = 0
    mismatched = 0
    max_error = 0.0
    for equipment_id, group in lot_data.groupby("equipment_id"):
        group = group.sort_values("start_time")
        for parameter in SPC_PARAMETERS:
            values = group[parameter].to_numpy(dtype=float)
            span = values.max() - values.min()
            lsl = values.min() - 0.5 * span
            usl = values.max() + 0.25 * span
            ours = calculate_spc_statistics(group[parameter], lsl, usl)
            reference = _independent_imr(values, lsl, usl)
            for key, expected in reference.items():
                error = abs(float(ours[key]) - expected)
                max_error = max(max_error, error)
                compared += 1
                if error > TOLERANCE * max(1.0, abs(expected)):
                    mismatched += 1
    return Check(
        "SPC",
        "I-MR 관리한계·Cp/Cpk/Pp/Ppk (설비 3 × 변수 4 × 지표 7)",
        compared,
        mismatched,
        max_error,
        mismatched == 0,
    )


# ---------------------------------------------------------- 규칙 판정
def _independent_grade(value: float, limits: dict[str, Any]) -> str:
    """구간표 방식 재판정. 경계값 포함 여부는 설정 문서 기준을 따릅니다."""

    direction = limits.get("direction")
    if direction == "lower_is_better":
        bounds = [
            (limits.get("normal_max"), "NORMAL"),
            (limits.get("watch_max"), "WATCH"),
            (limits.get("warning_max"), "WARNING"),
        ]
        for upper, grade in bounds:
            if upper is not None and value <= upper:
                return grade
        return "CRITICAL"
    if direction == "higher_is_better":
        critical = limits.get("critical_max", limits.get("critical_min"))
        if critical is not None and value < critical:
            return "CRITICAL"
        if value >= limits["normal_min"]:
            return "NORMAL"
        # 관찰 구간 하한: watch_min, 없으면 warning_min
        watch_floor = limits.get("watch_min", limits.get("warning_min"))
        if watch_floor is not None and value >= watch_floor:
            return "WATCH"
        return "WARNING"
    if limits["normal_min"] <= value <= limits["normal_max"]:
        return "NORMAL"
    if value < limits.get("critical_min", -np.inf) or value > limits.get(
        "critical_max", np.inf
    ):
        return "CRITICAL"
    return "WARNING"


def _parameter_values(lot_data: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    concentration = (
        (lot_data["chemical_concentration"] - lot_data["target_concentration"]).abs()
        / lot_data["target_concentration"].replace(0, np.nan)
        * 100
    ).fillna(0.0)
    temperature = (
        lot_data["bath_temperature"] - lot_data["target_bath_temperature"]
    ).abs()
    groups = {
        "quality": [
            "yield_percent",
            "particle_count",
            "metal_contamination_ppb",
            "water_mark_count",
            "organic_residue_count",
            "pattern_collapse_count",
        ],
        "equipment": [
            "pump_pressure",
            "flow_rate",
            "filter_differential_pressure",
            "nozzle_pressure",
            "motor_current",
            "vibration",
            "exhaust_pressure",
        ],
        "process": ["bath_age_hours", "diw_resistivity"],
    }
    values = {
        (group, column): lot_data[column]
        for group, columns in groups.items()
        for column in columns
    }
    values[("process", "concentration_deviation_percent")] = concentration
    values[("process", "temperature_deviation")] = temperature
    return values


def check_rule_grades(lot_data: pd.DataFrame, connection: sqlite3.Connection) -> Check:
    limits = yaml.safe_load(LIMITS_PATH.read_text(encoding="utf-8"))
    expected: dict[tuple[str, str], str] = {}
    compared = 0
    for (group, parameter), series in _parameter_values(lot_data).items():
        for lot_id, value in zip(lot_data["lot_id"], series):
            compared += 1
            grade = _independent_grade(float(value), limits[group][parameter])
            if grade != "NORMAL":
                expected[(str(lot_id), parameter)] = grade
    stored = {
        (str(lot_id), parameter): severity
        for lot_id, parameter, severity in connection.execute(
            "SELECT lot_id, parameter_name, severity FROM anomaly_result "
            "WHERE detection_method = 'RULE_BASED'"
        )
    }
    keys = set(expected) | set(stored)
    mismatched = sum(expected.get(key) != stored.get(key) for key in keys)
    return Check(
        "규칙 판정",
        f"LOT 120 × 항목 17 등급 재판정 vs DB anomaly_result ({len(stored)}건)",
        compared,
        mismatched,
        None,
        mismatched == 0,
    )


# ------------------------------------------------------- 예지보전
def check_pm_features(lot_data: pd.DataFrame) -> list[Check]:
    stored = pd.read_csv(PM_DATASET_PATH)
    compared = 0
    mismatched = 0
    max_error = 0.0
    label_compared = 0
    label_mismatched = 0
    censored = 0
    lot_data = lot_data.copy()
    lot_data["start_time"] = pd.to_datetime(lot_data["start_time"])
    for equipment_id, group in lot_data.groupby("equipment_id"):
        group = group.sort_values("start_time").reset_index(drop=True)
        position = {lot: index for index, lot in enumerate(group["lot_id"])}
        rows = stored[stored["equipment_id"] == equipment_id]
        for sensor in PM_SENSORS:
            values = group[sensor].to_numpy(dtype=float)
            windows = sliding_window_view(values, PM_WINDOW)
            x = np.arange(PM_WINDOW, dtype=float)
            reference = {
                "mean_5": windows.mean(axis=1),
                "std_5": windows.std(axis=1, ddof=1),
                "min_5": windows.min(axis=1),
                "max_5": windows.max(axis=1),
                "slope_5": np.array(
                    [stats.linregress(x, window).slope for window in windows]
                ),
            }
            for _, row in rows.iterrows():
                index = position[row["lot_id"]]
                window_index = index - (PM_WINDOW - 1)
                for suffix, array in reference.items():
                    error = abs(float(row[f"{sensor}_{suffix}"]) - array[window_index])
                    max_error = max(max_error, error)
                    compared += 1
                    if error > 1e-9 * max(1.0, abs(array[window_index])):
                        mismatched += 1
                change = values[index] - values[index - 1]
                error = abs(float(row[f"{sensor}_change_1"]) - change)
                max_error = max(max_error, error)
                compared += 1
                if error > 1e-9 * max(1.0, abs(change)):
                    mismatched += 1
        # 라벨: 이후 10 LOT 안에 WARNING/CRITICAL LOT가 있는가
        all_failure = group["lot_failure"].to_numpy(dtype=bool)
        for _, row in rows.iterrows():
            index = position[row["lot_id"]]
            future = all_failure[index + 1 : index + 1 + PM_HORIZON]
            label = int(future.any())
            label_compared += 1
            if label != int(row["failure_within_10_lots"]):
                label_mismatched += 1
            if label == 0 and len(future) < PM_HORIZON:
                censored += 1
    return [
        Check(
            "예지보전",
            "이동 평균·표준편차·최소·최대·기울기·변화량 (NumPy·scipy 재계산)",
            compared,
            mismatched,
            max_error,
            mismatched == 0,
        ),
        Check(
            "예지보전",
            "라벨 '10 LOT 안 이상 발생' 재계산",
            label_compared,
            label_mismatched,
            None,
            label_mismatched == 0,
            note=f"이 중 라벨 0이지만 이후 LOT가 10개 미만이라 확인 불가한 건: {censored}",
        ),
    ]


def model_disclosure() -> dict[str, Any]:
    stored = pd.read_csv(PM_DATASET_PATH)
    metrics = json.loads(PM_METRICS_PATH.read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]
    negatives = matrix[0][0] + matrix[0][1]
    positives = matrix[1][0] + matrix[1][1]
    return {
        "rows": int(len(stored)),
        "positive_rate": float(stored["failure_within_10_lots"].mean()),
        "test_rows": int(metrics["test_data_count"]),
        "test_negatives": int(negatives),
        "model_accuracy": float(metrics["accuracy"]),
        "majority_accuracy": float(positives / (positives + negatives)),
        "true_negatives": int(matrix[0][0]),
        "roc_auc": float(metrics["roc_auc"]),
    }


# ------------------------------------------------------------- 실행
def load_lot_data(connection: sqlite3.Connection) -> pd.DataFrame:
    from src.detection.rule_detector import load_lot_data as load

    lot_data = load(connection)
    worst = connection.execute(
        "SELECT lot_id FROM anomaly_result "
        "WHERE severity IN ('WARNING', 'CRITICAL') GROUP BY lot_id"
    ).fetchall()
    failing = {lot_id for (lot_id,) in worst}
    lot_data["lot_failure"] = lot_data["lot_id"].isin(failing)
    return lot_data


def run(database_path: Path = DATABASE_PATH) -> tuple[list[Check], dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    try:
        lot_data = load_lot_data(connection)
        checks = [
            check_spc(lot_data),
            check_rule_grades(lot_data, connection),
            *check_pm_features(lot_data),
        ]
    finally:
        connection.close()
    return checks, model_disclosure()
