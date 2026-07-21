from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
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
    / "predictive_maintenance_predictions.csv"
)


def load_model() -> RandomForestClassifier:
    """학습된 Predictive Maintenance 모델을 불러옵니다."""

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
            "불러온 모델이 RandomForestClassifier가 아닙니다."
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
            "Feature 목록의 형식이 올바르지 않습니다."
        )

    return [
        str(column)
        for column in feature_columns
    ]


def validate_prediction_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """예측 데이터에 필요한 Feature가 있는지 확인합니다."""

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "예측에 필요한 Feature가 없습니다:\n"
            + "\n".join(missing_columns)
        )


def classify_risk_level(
    probability: float,
) -> str:
    """고장 확률을 위험 등급으로 변환합니다."""

    if probability >= 0.8:
        return "CRITICAL"

    if probability >= 0.6:
        return "HIGH"

    if probability >= 0.4:
        return "MEDIUM"

    return "LOW"


def create_prediction_message(
    equipment_id: str,
    probability: float,
    risk_level: str,
) -> str:
    """설비별 Predictive Maintenance 메시지를 생성합니다."""

    probability_percent = probability * 100

    if risk_level == "CRITICAL":
        return (
            f"{equipment_id}는 향후 10 LOT 이내 이상 발생 확률이 "
            f"{probability_percent:.1f}%입니다. "
            "즉시 설비 점검과 예방보전을 권장합니다."
        )

    if risk_level == "HIGH":
        return (
            f"{equipment_id}는 향후 10 LOT 이내 이상 발생 확률이 "
            f"{probability_percent:.1f}%입니다. "
            "생산 전 주요 센서와 소모품 상태를 점검해야 합니다."
        )

    if risk_level == "MEDIUM":
        return (
            f"{equipment_id}는 향후 10 LOT 이내 이상 발생 확률이 "
            f"{probability_percent:.1f}%입니다. "
            "센서 추세를 지속적으로 관찰하는 것이 필요합니다."
        )

    return (
        f"{equipment_id}는 향후 10 LOT 이내 이상 발생 확률이 "
        f"{probability_percent:.1f}%로 현재 위험도가 낮습니다."
    )


def predict_equipment_risk() -> pd.DataFrame:
    """설비별 최신 LOT를 기준으로 고장 위험을 예측합니다."""

    model = load_model()

    feature_columns = load_feature_columns()

    feature_dataframe = (
        build_predictive_maintenance_dataset()
    )

    if feature_dataframe.empty:
        raise ValueError(
            "예측에 사용할 Feature 데이터가 없습니다."
        )

    validate_prediction_features(
        dataframe=feature_dataframe,
        feature_columns=feature_columns,
    )

    feature_dataframe["start_time"] = pd.to_datetime(
        feature_dataframe["start_time"],
        errors="coerce",
    )

    latest_dataframe = (
        feature_dataframe
        .sort_values(
            [
                "equipment_id",
                "start_time",
            ]
        )
        .groupby(
            "equipment_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    model_features = latest_dataframe[
        feature_columns
    ].copy()

    model_features = model_features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if model_features.isna().any().any():
        invalid_columns = (
            model_features
            .columns[
                model_features.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "예측 Feature에 결측값이 있습니다:\n"
            + "\n".join(invalid_columns)
        )

    failure_probability = model.predict_proba(
        model_features
    )[:, 1]

    prediction_result = latest_dataframe[
        [
            "equipment_id",
            "lot_id",
            "start_time",
            "severity",
            "filter_differential_pressure",
            "flow_rate",
            "motor_current",
            "vibration",
            "particle_count",
            "yield_percent",
        ]
    ].copy()

    prediction_result[
        "failure_probability"
    ] = failure_probability

    prediction_result[
        "failure_probability_percent"
    ] = (
        prediction_result[
            "failure_probability"
        ]
        * 100
    )

    prediction_result[
        "risk_level"
    ] = prediction_result[
        "failure_probability"
    ].apply(
        classify_risk_level
    )

    prediction_result[
        "recommended_action"
    ] = prediction_result.apply(
        lambda row: create_prediction_message(
            equipment_id=str(
                row["equipment_id"]
            ),
            probability=float(
                row["failure_probability"]
            ),
            risk_level=str(
                row["risk_level"]
            ),
        ),
        axis=1,
    )

    risk_order = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    prediction_result[
        "risk_score"
    ] = prediction_result[
        "risk_level"
    ].map(risk_order)

    prediction_result = (
        prediction_result
        .sort_values(
            by=[
                "risk_score",
                "failure_probability",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .drop(
            columns=[
                "risk_score",
            ]
        )
        .reset_index(drop=True)
    )

    return prediction_result


def save_predictions(
    prediction_dataframe: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """예측 결과를 CSV로 저장합니다."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


def print_prediction_summary(
    prediction_dataframe: pd.DataFrame,
) -> None:
    """설비별 예측 결과를 터미널에 출력합니다."""

    print("\nPredictive Maintenance Prediction")
    print("-" * 70)

    display_columns = [
        "equipment_id",
        "lot_id",
        "failure_probability_percent",
        "risk_level",
    ]

    display_dataframe = prediction_dataframe[
        display_columns
    ].copy()

    display_dataframe[
        "failure_probability_percent"
    ] = display_dataframe[
        "failure_probability_percent"
    ].map(
        lambda value: f"{value:.1f}%"
    )

    print(
        display_dataframe.to_string(
            index=False
        )
    )

    print("\nEngineer Recommendation")
    print("-" * 70)

    for row in prediction_dataframe.itertuples(
        index=False
    ):
        print(
            f"[{row.risk_level}] "
            f"{row.recommended_action}"
        )


def main() -> None:
    """Predictive Maintenance 예측을 실행합니다."""

    prediction_dataframe = (
        predict_equipment_risk()
    )

    output_path = save_predictions(
        prediction_dataframe
    )

    print_prediction_summary(
        prediction_dataframe
    )

    print(
        "\n저장 완료:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()