from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_PATH: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictive_maintenance_dataset.csv"
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

METRICS_PATH: Final[Path] = (
    MODEL_DIRECTORY
    / "model_metrics.json"
)

FEATURE_IMPORTANCE_PATH: Final[Path] = (
    MODEL_DIRECTORY
    / "feature_importance.csv"
)

TARGET_COLUMN: Final[str] = (
    "failure_within_10_lots"
)

RANDOM_STATE: Final[int] = 42


def load_training_dataset() -> pd.DataFrame:
    """Predictive Maintenance 학습 데이터를 불러옵니다."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "학습 데이터 파일이 없습니다.\n"
            "먼저 feature_engineering.py를 실행하세요.\n"
            f"경로: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "학습 데이터가 비어 있습니다."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(
            f"Target Column이 없습니다: {TARGET_COLUMN}"
        )

    return dataframe


def select_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """모델 학습에 사용할 숫자형 Feature를 선택합니다."""

    allowed_suffixes = (
        "_mean_5",
        "_std_5",
        "_min_5",
        "_max_5",
        "_slope_5",
        "_change_1",
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column.endswith(allowed_suffixes)
    ]

    if not feature_columns:
        raise ValueError(
            "학습에 사용할 Feature Column이 없습니다."
        )

    return feature_columns


def prepare_training_data(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """학습 데이터와 평가 데이터를 분리합니다."""

    feature_dataframe = dataframe[
        feature_columns
    ].copy()

    target_series = (
        dataframe[TARGET_COLUMN]
        .astype(int)
        .copy()
    )

    feature_dataframe = feature_dataframe.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    valid_mask = (
        feature_dataframe.notna().all(axis=1)
        & target_series.notna()
    )

    feature_dataframe = (
        feature_dataframe.loc[valid_mask]
        .reset_index(drop=True)
    )

    target_series = (
        target_series.loc[valid_mask]
        .reset_index(drop=True)
    )

    if target_series.nunique() < 2:
        raise ValueError(
            "Target 값이 하나의 Class로만 구성되어 있습니다.\n"
            "정상과 이상 데이터가 모두 필요합니다."
        )

    return train_test_split(
        feature_dataframe,
        target_series,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target_series,
    )


def create_model() -> RandomForestClassifier:
    """Random Forest 분류 모델을 생성합니다."""

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model(
    model: RandomForestClassifier,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, object]:
    """학습된 모델의 성능을 평가합니다."""

    predicted_target = model.predict(
        test_features
    )

    predicted_probability = model.predict_proba(
        test_features
    )[:, 1]

    accuracy = accuracy_score(
        test_target,
        predicted_target,
    )

    precision = precision_score(
        test_target,
        predicted_target,
        zero_division=0,
    )

    recall = recall_score(
        test_target,
        predicted_target,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        test_target,
        predicted_probability,
    )

    confusion = confusion_matrix(
        test_target,
        predicted_target,
    )

    report = classification_report(
        test_target,
        predicted_target,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "test_data_count": int(
            len(test_target)
        ),
    }


def create_feature_importance(
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Feature Importance 결과를 생성합니다."""

    importance_dataframe = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    return importance_dataframe.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)


def save_model_artifacts(
    model: RandomForestClassifier,
    feature_columns: list[str],
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
) -> None:
    """모델과 평가 결과를 파일로 저장합니다."""

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    with FEATURE_COLUMNS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_columns,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def print_model_summary(
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
) -> None:
    """모델 성능 결과를 터미널에 출력합니다."""

    print("\nPredictive Maintenance Model")
    print("-" * 55)

    print(
        f"Accuracy : "
        f"{float(metrics['accuracy']):.4f}"
    )

    print(
        f"Precision: "
        f"{float(metrics['precision']):.4f}"
    )

    print(
        f"Recall   : "
        f"{float(metrics['recall']):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{float(metrics['roc_auc']):.4f}"
    )

    print("\nConfusion Matrix")

    confusion = metrics["confusion_matrix"]

    print(
        f"TN: {confusion[0][0]}"
        f" | FP: {confusion[0][1]}"
    )

    print(
        f"FN: {confusion[1][0]}"
        f" | TP: {confusion[1][1]}"
    )

    print("\nTop 10 Feature Importance")

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    print("\n저장 파일")

    print(f"Model   : {MODEL_PATH}")
    print(f"Features: {FEATURE_COLUMNS_PATH}")
    print(f"Metrics : {METRICS_PATH}")
    print(
        f"Importance: {FEATURE_IMPORTANCE_PATH}"
    )


def train_predictive_maintenance_model(
) -> tuple[
    RandomForestClassifier,
    dict[str, object],
    pd.DataFrame,
]:
    """Predictive Maintenance 모델을 학습합니다."""

    dataframe = load_training_dataset()

    feature_columns = select_feature_columns(
        dataframe
    )

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = prepare_training_data(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    model = create_model()

    model.fit(
        train_features,
        train_target,
    )

    metrics = evaluate_model(
        model=model,
        test_features=test_features,
        test_target=test_target,
    )

    feature_importance = (
        create_feature_importance(
            model=model,
            feature_columns=feature_columns,
        )
    )

    save_model_artifacts(
        model=model,
        feature_columns=feature_columns,
        metrics=metrics,
        feature_importance=feature_importance,
    )

    return (
        model,
        metrics,
        feature_importance,
    )


def main() -> None:
    """모델 학습 프로그램을 실행합니다."""

    (
        _,
        metrics,
        feature_importance,
    ) = train_predictive_maintenance_model()

    print_model_summary(
        metrics=metrics,
        feature_importance=feature_importance,
    )


if __name__ == "__main__":
    main()