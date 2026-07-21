from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_processing.analysis_repository import (  # noqa: E402
    load_lot_trend,
)


SENSOR_COLUMNS: Final[list[str]] = [
    "filter_differential_pressure",
    "flow_rate",
    "motor_current",
    "vibration",
    "particle_count",
    "yield_percent",
]

WINDOW_SIZE: Final[int] = 5
PREDICTION_HORIZON: Final[int] = 10


def calculate_slope(
    series: pd.Series,
) -> float:
    """선형회귀 기울기를 계산합니다."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return 0.0

    x_values = np.arange(
        len(values),
        dtype=float,
    )

    slope = np.polyfit(
        x_values,
        values.to_numpy(dtype=float),
        1,
    )[0]

    return float(slope)


def create_rolling_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """설비별 이동통계 및 추세 특징을 생성합니다."""

    result_frames: list[pd.DataFrame] = []

    for equipment_id, equipment_data in dataframe.groupby(
        "equipment_id"
    ):
        equipment_data = (
            equipment_data
            .sort_values("start_time")
            .reset_index(drop=True)
            .copy()
        )

        for column in SENSOR_COLUMNS:
            if column not in equipment_data.columns:
                continue

            numeric_values = pd.to_numeric(
                equipment_data[column],
                errors="coerce",
            )

            equipment_data[f"{column}_mean_5"] = (
                numeric_values
                .rolling(
                    window=WINDOW_SIZE,
                    min_periods=WINDOW_SIZE,
                )
                .mean()
            )

            equipment_data[f"{column}_std_5"] = (
                numeric_values
                .rolling(
                    window=WINDOW_SIZE,
                    min_periods=WINDOW_SIZE,
                )
                .std()
            )

            equipment_data[f"{column}_min_5"] = (
                numeric_values
                .rolling(
                    window=WINDOW_SIZE,
                    min_periods=WINDOW_SIZE,
                )
                .min()
            )

            equipment_data[f"{column}_max_5"] = (
                numeric_values
                .rolling(
                    window=WINDOW_SIZE,
                    min_periods=WINDOW_SIZE,
                )
                .max()
            )

            equipment_data[f"{column}_slope_5"] = (
                numeric_values
                .rolling(
                    window=WINDOW_SIZE,
                    min_periods=WINDOW_SIZE,
                )
                .apply(
                    calculate_slope,
                    raw=False,
                )
            )

            equipment_data[f"{column}_change_1"] = (
                numeric_values.diff()
            )

        equipment_data["equipment_id"] = equipment_id

        result_frames.append(equipment_data)

    if not result_frames:
        return pd.DataFrame()

    return pd.concat(
        result_frames,
        ignore_index=True,
    )


def create_failure_flag(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """현재 LOT의 이상 여부를 0 또는 1로 변환합니다."""

    severity = (
        dataframe["severity"]
        .fillna("NORMAL")
        .astype(str)
        .str.upper()
    )

    return severity.isin(
        [
            "WARNING",
            "CRITICAL",
        ]
    ).astype(int)


def create_future_failure_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """향후 일정 LOT 이내 이상 발생 여부를 생성합니다."""

    result_frames: list[pd.DataFrame] = []

    for _, equipment_data in dataframe.groupby(
        "equipment_id"
    ):
        equipment_data = (
            equipment_data
            .sort_values("start_time")
            .reset_index(drop=True)
            .copy()
        )

        failure_array = (
            equipment_data["current_failure"]
            .to_numpy(dtype=int)
        )

        future_failure: list[int] = []
        remaining_lots: list[float] = []

        for current_index in range(
            len(equipment_data)
        ):
            search_start = current_index + 1

            search_end = min(
                current_index
                + PREDICTION_HORIZON
                + 1,
                len(equipment_data),
            )

            future_window = failure_array[
                search_start:search_end
            ]

            failure_positions = np.where(
                future_window == 1
            )[0]

            if len(failure_positions) > 0:
                future_failure.append(1)

                remaining_lots.append(
                    float(
                        failure_positions[0] + 1
                    )
                )

            else:
                future_failure.append(0)
                remaining_lots.append(np.nan)

        equipment_data[
            "failure_within_10_lots"
        ] = future_failure

        equipment_data[
            "remaining_lots_to_failure"
        ] = remaining_lots

        result_frames.append(equipment_data)

    return pd.concat(
        result_frames,
        ignore_index=True,
    )


def build_predictive_maintenance_dataset(
    equipment_id: str | None = None,
) -> pd.DataFrame:
    """Predictive Maintenance 학습 데이터를 생성합니다."""

    dataframe = load_lot_trend(
        equipment_id=equipment_id
    )

    if dataframe.empty:
        raise ValueError(
            "Predictive Maintenance용 LOT 데이터가 없습니다."
        )

    dataframe = dataframe.copy()

    dataframe["start_time"] = pd.to_datetime(
        dataframe["start_time"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "equipment_id",
            "lot_id",
            "start_time",
        ]
    )

    dataframe["current_failure"] = (
        create_failure_flag(dataframe)
    )

    feature_dataframe = create_rolling_features(
        dataframe
    )

    feature_dataframe = create_future_failure_target(
        feature_dataframe
    )

    required_feature_columns = [
        column
        for column in feature_dataframe.columns
        if column.endswith(
            (
                "_mean_5",
                "_std_5",
                "_min_5",
                "_max_5",
                "_slope_5",
                "_change_1",
            )
        )
    ]

    feature_dataframe = feature_dataframe.dropna(
        subset=required_feature_columns,
    ).reset_index(drop=True)

    return feature_dataframe


def save_predictive_maintenance_dataset(
    output_path: Path | None = None,
) -> Path:
    """생성한 학습 데이터를 CSV로 저장합니다."""

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "predictive_maintenance_dataset.csv"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = (
        build_predictive_maintenance_dataset()
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


def print_dataset_summary(
    dataframe: pd.DataFrame,
) -> None:
    """학습 데이터 요약을 출력합니다."""

    print("\nPredictive Maintenance Dataset")
    print("-" * 50)

    print(
        f"전체 데이터 수: {len(dataframe):,}"
    )

    print(
        "\n향후 10 LOT 이상 발생 여부:"
    )

    print(
        dataframe[
            "failure_within_10_lots"
        ]
        .value_counts()
        .sort_index()
    )

    failure_rows = dataframe[
        dataframe["failure_within_10_lots"] == 1
    ]

    if not failure_rows.empty:
        average_remaining_lots = (
            failure_rows[
                "remaining_lots_to_failure"
            ].mean()
        )

        print(
            "\n이상 발생까지 평균 잔여 LOT:"
            f" {average_remaining_lots:.2f}"
        )

    print("\n설비별 데이터 수:")

    print(
        dataframe[
            "equipment_id"
        ].value_counts()
    )


def main() -> None:
    """학습 데이터 생성 모듈을 실행합니다."""

    dataset = (
        build_predictive_maintenance_dataset()
    )

    output_path = (
        save_predictive_maintenance_dataset()
    )

    print_dataset_summary(dataset)

    print(
        "\n저장 완료:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()