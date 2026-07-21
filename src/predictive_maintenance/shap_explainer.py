from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predictive_maintenance.feature_engineering import (  # noqa: E402
    build_predictive_maintenance_dataset,
)


MODEL_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "models"
    / "predictive_maintenance"
)

MODEL_PATH: Final[Path] = (
    MODEL_DIRECTORY
    / "random_forest_model.joblib"
)

FEATURE_COLUMNS_PATH: Final[Path] = (
    MODEL_DIRECTORY
    / "feature_columns.json"
)

OUTPUT_PATH: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictive_maintenance_shap.csv"
)


def load_model() -> RandomForestClassifier:
    """학습된 Random Forest 모델을 불러옵니다."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Predictive Maintenance 모델이 없습니다.\n"
            "먼저 train_model.py를 실행하세요.\n"
            f"경로: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not isinstance(
        model,
        RandomForestClassifier,
    ):
        raise TypeError(
            "저장된 모델이 RandomForestClassifier가 아닙니다."
        )

    return model


def load_feature_columns() -> list[str]:
    """모델 학습에 사용된 Feature 목록을 불러옵니다."""

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Feature 목록 파일이 없습니다.\n"
            f"경로: {FEATURE_COLUMNS_PATH}"
        )

    with FEATURE_COLUMNS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    if not isinstance(feature_columns, list):
        raise TypeError(
            "Feature 목록 파일 형식이 올바르지 않습니다."
        )

    return [
        str(column)
        for column in feature_columns
    ]


def extract_positive_class_shap_values(
    shap_values: object,
) -> np.ndarray:
    """이상 Class 1에 해당하는 SHAP 값을 추출합니다."""

    if isinstance(shap_values, list):
        if len(shap_values) < 2:
            raise ValueError(
                "이상 Class의 SHAP 값을 찾을 수 없습니다."
            )

        return np.asarray(
            shap_values[1],
            dtype=float,
        )

    shap_array = np.asarray(
        shap_values,
        dtype=float,
    )

    if shap_array.ndim == 3:
        if shap_array.shape[2] < 2:
            raise ValueError(
                "이상 Class의 SHAP 값을 찾을 수 없습니다."
            )

        return shap_array[:, :, 1]

    if shap_array.ndim == 2:
        return shap_array

    raise ValueError(
        "지원하지 않는 SHAP 결과 형식입니다.\n"
        f"SHAP shape: {shap_array.shape}"
    )


def classify_contribution_direction(
    shap_value: float,
) -> str:
    """SHAP 값 방향을 위험 증가 또는 감소로 분류합니다."""

    if shap_value > 0:
        return "위험 증가"

    if shap_value < 0:
        return "위험 감소"

    return "영향 없음"


def convert_feature_name(
    feature_name: str,
) -> str:
    """Feature 이름을 읽기 쉬운 한글 이름으로 변환합니다."""

    sensor_name_map = {
        "filter_differential_pressure": "Filter 차압",
        "flow_rate": "유량",
        "motor_current": "Motor Current",
        "vibration": "진동",
        "particle_count": "Particle",
        "yield_percent": "수율",
    }

    statistic_name_map = {
        "mean_5": "최근 5 LOT 평균",
        "std_5": "최근 5 LOT 표준편차",
        "min_5": "최근 5 LOT 최솟값",
        "max_5": "최근 5 LOT 최댓값",
        "slope_5": "최근 5 LOT 기울기",
        "change_1": "직전 LOT 대비 변화",
    }

    converted_name = feature_name

    for original_name, display_name in sensor_name_map.items():
        if converted_name.startswith(original_name):
            converted_name = converted_name.replace(
                original_name,
                display_name,
                1,
            )
            break

    for original_name, display_name in statistic_name_map.items():
        if converted_name.endswith(original_name):
            converted_name = converted_name.replace(
                original_name,
                display_name,
            )
            break

    return converted_name.replace("_", " ")


def create_engineer_interpretation(
    feature_name: str,
    feature_value: float,
    shap_value: float,
) -> str:
    """Feature별 SHAP 분석 문장을 생성합니다."""

    display_name = convert_feature_name(
        feature_name
    )

    if shap_value > 0:
        return (
            f"{display_name} 값 {feature_value:.3f}이 "
            f"이상 발생 위험을 증가시키는 방향으로 "
            f"영향을 주었습니다."
        )

    if shap_value < 0:
        return (
            f"{display_name} 값 {feature_value:.3f}이 "
            f"이상 발생 위험을 낮추는 방향으로 "
            f"영향을 주었습니다."
        )

    return (
        f"{display_name} 값 {feature_value:.3f}은 "
        "현재 예측에 미치는 영향이 낮습니다."
    )


def calculate_latest_equipment_shap() -> pd.DataFrame:
    """설비별 최신 LOT의 SHAP 값을 계산합니다."""

    model = load_model()

    feature_columns = load_feature_columns()

    feature_dataframe = (
        build_predictive_maintenance_dataset()
    )

    if feature_dataframe.empty:
        raise ValueError(
            "SHAP 분석에 사용할 데이터가 없습니다."
        )

    missing_columns = [
        column
        for column in feature_columns
        if column not in feature_dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "SHAP 분석에 필요한 Feature가 없습니다:\n"
            + "\n".join(missing_columns)
        )

    feature_dataframe["start_time"] = pd.to_datetime(
        feature_dataframe["start_time"],
        errors="coerce",
    )

    latest_dataframe = (
        feature_dataframe
        .sort_values(
            by=[
                "equipment_id",
                "start_time",
            ]
        )
        .groupby(
            "equipment_id",
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    model_features = latest_dataframe[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if model_features.isna().any().any():
        invalid_columns = (
            model_features.columns[
                model_features.isna().any()
            ].tolist()
        )

        raise ValueError(
            "SHAP 분석 Feature에 결측값이 있습니다:\n"
            + "\n".join(invalid_columns)
        )

    explainer = shap.TreeExplainer(
        model
    )

    raw_shap_values = explainer.shap_values(
        model_features
    )

    positive_shap_values = (
        extract_positive_class_shap_values(
            raw_shap_values
        )
    )

    failure_probabilities = model.predict_proba(
        model_features
    )[:, 1]

    result_rows: list[dict[str, object]] = []

    for row_index, latest_row in (
        latest_dataframe.iterrows()
    ):
        equipment_id = str(
            latest_row["equipment_id"]
        )

        lot_id = str(
            latest_row["lot_id"]
        )

        start_time = latest_row["start_time"]

        failure_probability = float(
            failure_probabilities[row_index]
        )

        for feature_index, feature_name in enumerate(
            feature_columns
        ):
            feature_value = float(
                model_features.iloc[
                    row_index,
                    feature_index,
                ]
            )

            shap_value = float(
                positive_shap_values[
                    row_index,
                    feature_index,
                ]
            )

            result_rows.append(
                {
                    "equipment_id": equipment_id,
                    "lot_id": lot_id,
                    "start_time": start_time,
                    "failure_probability": (
                        failure_probability
                    ),
                    "failure_probability_percent": (
                        failure_probability * 100
                    ),
                    "feature": feature_name,
                    "feature_label": convert_feature_name(
                        feature_name
                    ),
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "absolute_shap_value": abs(
                        shap_value
                    ),
                    "contribution_direction": (
                        classify_contribution_direction(
                            shap_value
                        )
                    ),
                    "engineer_interpretation": (
                        create_engineer_interpretation(
                            feature_name=feature_name,
                            feature_value=feature_value,
                            shap_value=shap_value,
                        )
                    ),
                }
            )

    result_dataframe = pd.DataFrame(
        result_rows
    )

    result_dataframe["importance_rank"] = (
        result_dataframe
        .groupby("equipment_id")[
            "absolute_shap_value"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    result_dataframe = result_dataframe.sort_values(
        by=[
            "equipment_id",
            "importance_rank",
        ]
    ).reset_index(drop=True)

    return result_dataframe


def save_shap_results(
    dataframe: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """SHAP 분석 결과를 CSV로 저장합니다."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


def print_shap_summary(
    dataframe: pd.DataFrame,
) -> None:
    """설비별 주요 SHAP 원인을 출력합니다."""

    print("\nPredictive Maintenance SHAP Analysis")
    print("-" * 75)

    equipment_ids = (
        dataframe["equipment_id"]
        .drop_duplicates()
        .tolist()
    )

    for equipment_id in equipment_ids:
        equipment_dataframe = dataframe[
            dataframe["equipment_id"]
            == equipment_id
        ].copy()

        top_features = equipment_dataframe.head(5)

        probability = float(
            equipment_dataframe[
                "failure_probability_percent"
            ].iloc[0]
        )

        lot_id = str(
            equipment_dataframe["lot_id"].iloc[0]
        )

        print(
            f"\n[{equipment_id}] "
            f"기준 LOT: {lot_id}"
        )

        print(
            f"향후 10 LOT 이상 발생 확률: "
            f"{probability:.1f}%"
        )

        for row in top_features.itertuples(
            index=False
        ):
            print(
                f"{row.importance_rank}. "
                f"{row.feature_label}"
                f" | 값: {row.feature_value:.3f}"
                f" | SHAP: {row.shap_value:+.4f}"
                f" | {row.contribution_direction}"
            )


def main() -> None:
    """SHAP 분석 모듈을 실행합니다."""

    shap_dataframe = (
        calculate_latest_equipment_shap()
    )

    output_path = save_shap_results(
        dataframe=shap_dataframe
    )

    print_shap_summary(
        dataframe=shap_dataframe
    )

    print(
        "\n저장 완료:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()