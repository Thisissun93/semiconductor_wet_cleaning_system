from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predictive_maintenance.prediction_engine import (  # noqa: E402
    predict_equipment_risk,
    save_predictions,
)
from src.predictive_maintenance.shap_explainer import (  # noqa: E402
    calculate_latest_equipment_shap,
    save_shap_results,
)

from src.reporting.predictive_maintenance_report import (  # noqa: E402
    generate_predictive_maintenance_report,
)


MODEL_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "models"
    / "predictive_maintenance"
)

METRICS_PATH: Final[Path] = (
    MODEL_DIRECTORY
    / "model_metrics.json"
)

FEATURE_IMPORTANCE_PATH: Final[Path] = (
    MODEL_DIRECTORY
    / "feature_importance.csv"
)


RISK_ORDER: Final[dict[str, int]] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


RISK_DESCRIPTION: Final[dict[str, str]] = {
    "CRITICAL": "즉시 설비 점검 및 예방보전 필요",
    "HIGH": "생산 전 주요 센서와 소모품 점검 필요",
    "MEDIUM": "센서 추세 집중 모니터링 필요",
    "LOW": "현재 위험 수준이 낮음",
}


FEATURE_NAME_MAP: Final[dict[str, str]] = {
    "filter_differential_pressure": "Filter 차압",
    "flow_rate": "유량",
    "motor_current": "Motor Current",
    "vibration": "진동",
    "particle_count": "Particle",
    "yield_percent": "수율",
    "mean_5": "최근 5 LOT 평균",
    "std_5": "최근 5 LOT 표준편차",
    "min_5": "최근 5 LOT 최솟값",
    "max_5": "최근 5 LOT 최댓값",
    "slope_5": "최근 5 LOT 기울기",
    "change_1": "직전 LOT 대비 변화",
}


def configure_page() -> None:
    """Streamlit 페이지 기본 설정을 적용합니다."""

    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="🛠️",
        layout="wide",
    )


def apply_custom_css() -> None:
    """Predictive Maintenance 페이지 전용 스타일을 적용합니다."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .pm-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 1.7rem;
            margin-bottom: 0.4rem;
        }

        .pm-section-description {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .risk-card {
            background-color: white;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 12px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        }

        .risk-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .risk-description {
            color: #475467;
            line-height: 1.7;
        }

        .analysis-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 10px;
        }

        .analysis-card-title {
            font-weight: 700;
            margin-bottom: 6px;
        }

        .analysis-card-description {
            color: #475467;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def load_prediction_data() -> pd.DataFrame:
    """설비별 최신 Predictive Maintenance 예측 결과를 생성합니다."""

    prediction_dataframe = predict_equipment_risk()

    save_predictions(
        prediction_dataframe=prediction_dataframe
    )

    return prediction_dataframe


@st.cache_data(ttl=60)
def load_model_metrics() -> dict[str, object]:
    """저장된 Random Forest 모델 평가 결과를 불러옵니다."""

    if not METRICS_PATH.exists():
        return {}

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        return {}

    return metrics


@st.cache_data(ttl=60)
def load_feature_importance() -> pd.DataFrame:
    """저장된 전체 모델 Feature Importance를 불러옵니다."""

    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    if dataframe.empty:
        return pd.DataFrame()

    required_columns = {
        "feature",
        "importance",
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return pd.DataFrame()

    dataframe["importance"] = pd.to_numeric(
        dataframe["importance"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["importance"]
    )

    return dataframe


@st.cache_data(ttl=60)
def load_shap_data() -> pd.DataFrame:
    """설비별 최신 LOT의 SHAP 분석 결과를 생성합니다."""

    shap_dataframe = (
        calculate_latest_equipment_shap()
    )

    save_shap_results(
        dataframe=shap_dataframe
    )

    return shap_dataframe


def convert_feature_name(
    feature_name: str,
) -> str:
    """Feature 이름을 화면용 한글 이름으로 변환합니다."""

    converted_name = str(feature_name)

    sensor_names = [
        "filter_differential_pressure",
        "flow_rate",
        "motor_current",
        "vibration",
        "particle_count",
        "yield_percent",
    ]

    statistic_names = [
        "mean_5",
        "std_5",
        "min_5",
        "max_5",
        "slope_5",
        "change_1",
    ]

    for sensor_name in sensor_names:
        if converted_name.startswith(sensor_name):
            converted_name = converted_name.replace(
                sensor_name,
                FEATURE_NAME_MAP[sensor_name],
                1,
            )
            break

    for statistic_name in statistic_names:
        if converted_name.endswith(statistic_name):
            converted_name = converted_name.replace(
                statistic_name,
                FEATURE_NAME_MAP[statistic_name],
            )
            break

    return converted_name.replace("_", " ")


def render_header() -> None:
    """페이지 제목과 목적을 표시합니다."""

    st.title("Predictive Maintenance Dashboard")

    st.caption(
        "최근 공정 센서 추세와 Random Forest 모델을 이용하여 "
        "향후 10 LOT 이내 이상 발생 가능성을 예측하고, "
        "SHAP을 통해 예측 원인을 분석합니다."
    )


def render_sidebar(
    prediction_dataframe: pd.DataFrame,
) -> tuple[str, int, int]:
    """설비 필터와 차트 표시 설정을 제공합니다."""

    st.sidebar.title("Prediction Filter")

    equipment_ids = (
        prediction_dataframe["equipment_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    equipment_options = [
        "전체 설비",
        *equipment_ids,
    ]

    selected_equipment = st.sidebar.selectbox(
        "설비 선택",
        options=equipment_options,
    )

    importance_count = st.sidebar.slider(
        "전체 모델 주요 영향 인자 수",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    shap_count = st.sidebar.slider(
        "설비별 SHAP 영향 인자 수",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    if st.sidebar.button(
        "예측 결과 새로고침",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    st.sidebar.caption(
        "위험 등급 기준"
    )

    st.sidebar.markdown(
        """
        - **CRITICAL:** 80% 이상
        - **HIGH:** 60% 이상
        - **MEDIUM:** 40% 이상
        - **LOW:** 40% 미만
        """
    )

    return (
        selected_equipment,
        importance_count,
        shap_count,
    )


def filter_prediction_data(
    prediction_dataframe: pd.DataFrame,
    selected_equipment: str,
) -> pd.DataFrame:
    """선택된 설비를 기준으로 예측 결과를 필터링합니다."""

    if selected_equipment == "전체 설비":
        return prediction_dataframe.copy()

    return prediction_dataframe[
        prediction_dataframe["equipment_id"].astype(str)
        == selected_equipment
    ].copy()


def render_prediction_kpis(
    prediction_dataframe: pd.DataFrame,
) -> None:
    """예측 결과의 핵심 KPI를 표시합니다."""

    equipment_count = len(
        prediction_dataframe
    )

    highest_probability = float(
        prediction_dataframe[
            "failure_probability_percent"
        ].max()
    )

    average_probability = float(
        prediction_dataframe[
            "failure_probability_percent"
        ].mean()
    )

    critical_count = int(
        (
            prediction_dataframe["risk_level"]
            == "CRITICAL"
        ).sum()
    )

    high_risk_count = int(
        prediction_dataframe[
            "risk_level"
        ].isin(
            [
                "CRITICAL",
                "HIGH",
            ]
        ).sum()
    )

    column1, column2, column3, column4, column5 = (
        st.columns(5)
    )

    column1.metric(
        "분석 설비",
        f"{equipment_count}대",
    )

    column2.metric(
        "최고 이상 확률",
        f"{highest_probability:.1f}%",
    )

    column3.metric(
        "평균 이상 확률",
        f"{average_probability:.1f}%",
    )

    column4.metric(
        "Critical 설비",
        f"{critical_count}대",
    )

    column5.metric(
        "고위험 설비",
        f"{high_risk_count}대",
    )


def create_risk_probability_chart(
    prediction_dataframe: pd.DataFrame,
) -> go.Figure:
    """설비별 이상 발생 확률 막대그래프를 생성합니다."""

    chart_dataframe = (
        prediction_dataframe.copy()
    )

    chart_dataframe["risk_score"] = (
        chart_dataframe["risk_level"]
        .map(RISK_ORDER)
        .fillna(0)
    )

    chart_dataframe = chart_dataframe.sort_values(
        by=[
            "risk_score",
            "failure_probability_percent",
        ],
        ascending=[
            True,
            True,
        ],
    )

    figure = px.bar(
        chart_dataframe,
        x="failure_probability_percent",
        y="equipment_id",
        orientation="h",
        color="risk_level",
        text="failure_probability_percent",
        category_orders={
            "risk_level": [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ]
        },
        hover_data={
            "lot_id": True,
            "severity": True,
            "particle_count": ":.2f",
            "yield_percent": ":.2f",
            "failure_probability_percent": ":.1f",
            "risk_score": False,
        },
        labels={
            "failure_probability_percent": (
                "향후 10 LOT 이내 이상 발생 확률 (%)"
            ),
            "equipment_id": "설비",
            "risk_level": "위험 등급",
            "lot_id": "기준 LOT",
            "severity": "현재 이상 등급",
            "particle_count": "Particle",
            "yield_percent": "수율",
        },
        title="Equipment Failure Risk",
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    figure.add_vline(
        x=40,
        line_dash="dot",
        annotation_text="MEDIUM",
        annotation_position="top",
    )

    figure.add_vline(
        x=60,
        line_dash="dot",
        annotation_text="HIGH",
        annotation_position="top",
    )

    figure.add_vline(
        x=80,
        line_dash="dot",
        annotation_text="CRITICAL",
        annotation_position="top",
    )

    figure.update_xaxes(
        range=[
            0,
            105,
        ]
    )

    figure.update_layout(
        height=max(
            400,
            len(chart_dataframe) * 85,
        ),
        margin=dict(
            l=20,
            r=70,
            t=70,
            b=20,
        ),
    )

    return figure


def render_equipment_recommendations(
    prediction_dataframe: pd.DataFrame,
) -> None:
    """설비별 위험 등급과 권장 조치를 표시합니다."""

    st.markdown(
        '<div class="pm-section-title">'
        "Engineer Recommendation"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pm-section-description">
            이상 발생 확률과 위험 등급에 따라
            설비별 예방보전 우선순위를 제시합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sorted_dataframe = (
        prediction_dataframe.copy()
    )

    sorted_dataframe["risk_score"] = (
        sorted_dataframe["risk_level"]
        .map(RISK_ORDER)
        .fillna(0)
    )

    sorted_dataframe = sorted_dataframe.sort_values(
        by=[
            "risk_score",
            "failure_probability_percent",
        ],
        ascending=[
            False,
            False,
        ],
    )

    for row in sorted_dataframe.itertuples(
        index=False
    ):
        risk_level = str(
            row.risk_level
        )

        probability = float(
            row.failure_probability_percent
        )

        equipment_id = str(
            row.equipment_id
        )

        lot_id = str(
            row.lot_id
        )

        recommendation = str(
            row.recommended_action
        )

        description = RISK_DESCRIPTION.get(
            risk_level,
            "추가 분석 필요",
        )

        card_html = (
            '<div class="risk-card">'
            '<div class="risk-title">'
            f"{equipment_id} / {risk_level} / "
            f"{probability:.1f}%"
            "</div>"
            '<div class="risk-description">'
            f"기준 LOT: <b>{lot_id}</b><br>"
            f"판정: <b>{description}</b><br>"
            f"권장 의견: {recommendation}"
            "</div>"
            "</div>"
        )

        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )


def render_prediction_table(
    prediction_dataframe: pd.DataFrame,
) -> None:
    """설비별 Predictive Maintenance 상세 결과를 표시합니다."""

    st.markdown(
        '<div class="pm-section-title">'
        "설비별 예측 상세"
        "</div>",
        unsafe_allow_html=True,
    )

    display_columns = [
        "equipment_id",
        "lot_id",
        "start_time",
        "failure_probability_percent",
        "risk_level",
        "severity",
        "filter_differential_pressure",
        "flow_rate",
        "motor_current",
        "vibration",
        "particle_count",
        "yield_percent",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in prediction_dataframe.columns
    ]

    display_dataframe = prediction_dataframe[
        available_columns
    ].copy()

    if "start_time" in display_dataframe.columns:
        display_dataframe["start_time"] = pd.to_datetime(
            display_dataframe["start_time"],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    if (
        "failure_probability_percent"
        in display_dataframe.columns
    ):
        display_dataframe[
            "failure_probability_percent"
        ] = display_dataframe[
            "failure_probability_percent"
        ].map(
            lambda value: f"{float(value):.1f}%"
        )

    display_dataframe = display_dataframe.rename(
        columns={
            "equipment_id": "설비",
            "lot_id": "기준 LOT",
            "start_time": "생산 시간",
            "failure_probability_percent": "이상 확률",
            "risk_level": "위험 등급",
            "severity": "현재 이상 등급",
            "filter_differential_pressure": "Filter 차압",
            "flow_rate": "유량",
            "motor_current": "Motor Current",
            "vibration": "진동",
            "particle_count": "Particle",
            "yield_percent": "수율",
        }
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
        hide_index=True,
    )


def render_model_performance(
    metrics: dict[str, object],
) -> None:
    """모델 성능 지표와 혼동행렬을 표시합니다."""

    st.markdown(
        '<div class="pm-section-title">'
        "Model Performance"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pm-section-description">
            저장된 테스트 데이터 평가 결과를 이용하여
            Random Forest 모델 성능을 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not metrics:
        st.warning(
            "모델 평가 파일이 없습니다. "
            "train_model.py를 먼저 실행하세요."
        )
        return

    accuracy = float(
        metrics.get(
            "accuracy",
            0.0,
        )
    )

    precision = float(
        metrics.get(
            "precision",
            0.0,
        )
    )

    recall = float(
        metrics.get(
            "recall",
            0.0,
        )
    )

    roc_auc = float(
        metrics.get(
            "roc_auc",
            0.0,
        )
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Accuracy",
        f"{accuracy:.3f}",
    )

    column2.metric(
        "Precision",
        f"{precision:.3f}",
    )

    column3.metric(
        "Recall",
        f"{recall:.3f}",
    )

    column4.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}",
    )

    confusion_matrix = metrics.get(
        "confusion_matrix"
    )

    valid_confusion_matrix = (
        isinstance(confusion_matrix, list)
        and len(confusion_matrix) == 2
        and all(
            isinstance(row, list)
            and len(row) == 2
            for row in confusion_matrix
        )
    )

    if valid_confusion_matrix:
        confusion_dataframe = pd.DataFrame(
            confusion_matrix,
            index=[
                "Actual Normal",
                "Actual Failure",
            ],
            columns=[
                "Predicted Normal",
                "Predicted Failure",
            ],
        )

        figure = px.imshow(
            confusion_dataframe,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels={
                "x": "예측",
                "y": "실제",
                "color": "건수",
            },
        )

        figure.update_layout(
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    st.info(
        "예지보전 모델에서는 실제 이상을 정상으로 판단하는 "
        "False Negative를 줄이는 것이 중요합니다. "
        "따라서 Accuracy뿐 아니라 Recall을 함께 확인해야 합니다."
    )


def render_feature_importance(
    feature_importance: pd.DataFrame,
    importance_count: int,
) -> None:
    """전체 Random Forest 모델의 Feature Importance를 표시합니다."""

    st.markdown(
        '<div class="pm-section-title">'
        "전체 모델 주요 영향 인자"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pm-section-description">
            Feature Importance는 전체 데이터 기준으로
            모델이 자주 활용한 인자를 보여줍니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if feature_importance.empty:
        st.warning(
            "Feature Importance 파일이 없습니다. "
            "train_model.py를 먼저 실행하세요."
        )
        return

    chart_dataframe = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False,
        )
        .head(importance_count)
        .copy()
    )

    chart_dataframe["feature_label"] = (
        chart_dataframe["feature"]
        .astype(str)
        .apply(convert_feature_name)
    )

    chart_dataframe = chart_dataframe.sort_values(
        by="importance",
        ascending=True,
    )

    figure = px.bar(
        chart_dataframe,
        x="importance",
        y="feature_label",
        orientation="h",
        text="importance",
        labels={
            "importance": "Feature Importance",
            "feature_label": "영향 인자",
        },
        title="Random Forest Global Feature Importance",
    )

    figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    figure.update_layout(
        height=max(
            450,
            importance_count * 45,
        ),
        showlegend=False,
        margin=dict(
            l=20,
            r=70,
            t=70,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    top_feature_dataframe = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False,
        )
        .head(1)
    )

    if not top_feature_dataframe.empty:
        top_feature = (
            top_feature_dataframe.iloc[0]
        )

        top_feature_label = convert_feature_name(
            str(top_feature["feature"])
        )

        top_feature_importance = float(
            top_feature["importance"]
        )

        st.info(
            "현재 전체 모델에서 가장 영향도가 높은 인자는 "
            f"'{top_feature_label}'이며, "
            f"중요도는 {top_feature_importance:.3f}입니다."
        )


def create_local_shap_chart(
    shap_dataframe: pd.DataFrame,
    equipment_id: str,
    top_count: int,
) -> go.Figure:
    """선택 설비의 Local SHAP 막대그래프를 생성합니다."""

    chart_dataframe = shap_dataframe[
        shap_dataframe["equipment_id"].astype(str)
        == equipment_id
    ].copy()

    if chart_dataframe.empty:
        return go.Figure()

    chart_dataframe["shap_value"] = pd.to_numeric(
        chart_dataframe["shap_value"],
        errors="coerce",
    )

    chart_dataframe[
        "absolute_shap_value"
    ] = pd.to_numeric(
        chart_dataframe["absolute_shap_value"],
        errors="coerce",
    )

    chart_dataframe = chart_dataframe.dropna(
        subset=[
            "shap_value",
            "absolute_shap_value",
        ]
    )

    chart_dataframe = (
        chart_dataframe
        .sort_values(
            by="absolute_shap_value",
            ascending=False,
        )
        .head(top_count)
        .sort_values(
            by="shap_value",
            ascending=True,
        )
    )

    figure = px.bar(
        chart_dataframe,
        x="shap_value",
        y="feature_label",
        orientation="h",
        color="contribution_direction",
        text="shap_value",
        hover_data={
            "feature_value": ":.3f",
            "absolute_shap_value": ":.4f",
            "engineer_interpretation": True,
            "shap_value": ":.4f",
        },
        labels={
            "shap_value": "SHAP Value",
            "feature_label": "영향 인자",
            "contribution_direction": "영향 방향",
            "feature_value": "Feature 값",
            "absolute_shap_value": "절대 SHAP",
            "engineer_interpretation": "엔지니어 해석",
        },
        title=f"{equipment_id} Local SHAP Analysis",
    )

    figure.add_vline(
        x=0,
        line_dash="solid",
    )

    figure.update_traces(
        texttemplate="%{text:+.4f}",
        textposition="outside",
    )

    figure.update_layout(
        height=max(
            480,
            len(chart_dataframe) * 52,
        ),
        margin=dict(
            l=20,
            r=90,
            t=70,
            b=20,
        ),
    )

    return figure


def render_local_shap_analysis(
    shap_dataframe: pd.DataFrame,
    selected_equipment: str,
    shap_count: int,
) -> None:
    """설비별 SHAP 영향 인자와 엔지니어 해석을 표시합니다."""

    st.markdown(
        '<div class="pm-section-title">'
        "설비별 예측 원인 분석"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pm-section-description">
            SHAP 값이 양수이면 이상 발생 확률을 높이는 방향,
            음수이면 이상 발생 확률을 낮추는 방향으로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if shap_dataframe.empty:
        st.warning(
            "SHAP 분석 결과가 없습니다."
        )
        return

    available_equipment_ids = (
        shap_dataframe["equipment_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not available_equipment_ids:
        st.warning(
            "SHAP 분석이 가능한 설비가 없습니다."
        )
        return

    if (
        selected_equipment != "전체 설비"
        and selected_equipment
        in available_equipment_ids
    ):
        equipment_id = selected_equipment

    else:
        equipment_id = st.selectbox(
            "SHAP 분석 설비 선택",
            options=available_equipment_ids,
            key="shap_equipment_selector",
        )

    equipment_dataframe = shap_dataframe[
        shap_dataframe["equipment_id"].astype(str)
        == equipment_id
    ].copy()

    if equipment_dataframe.empty:
        st.warning(
            "선택 설비의 SHAP 분석 결과가 없습니다."
        )
        return

    probability = float(
        equipment_dataframe[
            "failure_probability_percent"
        ].iloc[0]
    )

    lot_id = str(
        equipment_dataframe[
            "lot_id"
        ].iloc[0]
    )

    start_time = pd.to_datetime(
        equipment_dataframe[
            "start_time"
        ].iloc[0],
        errors="coerce",
    )

    if pd.isna(start_time):
        start_time_text = "-"
    else:
        start_time_text = start_time.strftime(
            "%Y-%m-%d %H:%M"
        )

    positive_count = int(
        (
            equipment_dataframe[
                "contribution_direction"
            ]
            == "위험 증가"
        ).sum()
    )

    negative_count = int(
        (
            equipment_dataframe[
                "contribution_direction"
            ]
            == "위험 감소"
        ).sum()
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "분석 설비",
        equipment_id,
    )

    column2.metric(
        "향후 10 LOT 이상 확률",
        f"{probability:.1f}%",
    )

    column3.metric(
        "위험 증가 인자",
        f"{positive_count}개",
    )

    column4.metric(
        "위험 감소 인자",
        f"{negative_count}개",
    )

    st.caption(
        f"분석 기준 LOT: {lot_id} | "
        f"생산 시간: {start_time_text}"
    )

    shap_chart = create_local_shap_chart(
        shap_dataframe=shap_dataframe,
        equipment_id=equipment_id,
        top_count=shap_count,
    )

    st.plotly_chart(
        shap_chart,
        use_container_width=True,
    )

    top_risk_factors = (
        equipment_dataframe[
            equipment_dataframe[
                "contribution_direction"
            ]
            == "위험 증가"
        ]
        .sort_values(
            by="absolute_shap_value",
            ascending=False,
        )
        .head(3)
    )

    top_protective_factors = (
        equipment_dataframe[
            equipment_dataframe[
                "contribution_direction"
            ]
            == "위험 감소"
        ]
        .sort_values(
            by="absolute_shap_value",
            ascending=False,
        )
        .head(3)
    )

    risk_column, protective_column = st.columns(2)

    with risk_column:
        st.markdown(
            "#### 주요 위험 증가 인자"
        )

        if top_risk_factors.empty:
            st.success(
                "현재 이상 위험을 뚜렷하게 증가시키는 "
                "주요 인자가 없습니다."
            )

        else:
            for row in top_risk_factors.itertuples(
                index=False
            ):
                card_html = (
                    '<div class="analysis-card">'
                    '<div class="analysis-card-title">'
                    f"{row.importance_rank}. "
                    f"{row.feature_label}"
                    "</div>"
                    '<div class="analysis-card-description">'
                    f"현재 값: "
                    f"<b>{float(row.feature_value):.3f}</b><br>"
                    f"SHAP 영향: "
                    f"<b>{float(row.shap_value):+.4f}</b><br>"
                    f"{row.engineer_interpretation}"
                    "</div>"
                    "</div>"
                )

                st.markdown(
                    card_html,
                    unsafe_allow_html=True,
                )

    with protective_column:
        st.markdown(
            "#### 주요 위험 감소 인자"
        )

        if top_protective_factors.empty:
            st.warning(
                "현재 이상 위험을 낮추는 주요 인자가 "
                "뚜렷하지 않습니다."
            )

        else:
            for row in (
                top_protective_factors.itertuples(
                    index=False
                )
            ):
                card_html = (
                    '<div class="analysis-card">'
                    '<div class="analysis-card-title">'
                    f"{row.importance_rank}. "
                    f"{row.feature_label}"
                    "</div>"
                    '<div class="analysis-card-description">'
                    f"현재 값: "
                    f"<b>{float(row.feature_value):.3f}</b><br>"
                    f"SHAP 영향: "
                    f"<b>{float(row.shap_value):+.4f}</b><br>"
                    f"{row.engineer_interpretation}"
                    "</div>"
                    "</div>"
                )

                st.markdown(
                    card_html,
                    unsafe_allow_html=True,
                )

    render_shap_engineer_opinion(
        equipment_id=equipment_id,
        probability=probability,
        top_risk_factors=top_risk_factors,
        top_protective_factors=top_protective_factors,
    )

    render_shap_detail_table(
        equipment_dataframe=equipment_dataframe,
    )


def render_shap_engineer_opinion(
    equipment_id: str,
    probability: float,
    top_risk_factors: pd.DataFrame,
    top_protective_factors: pd.DataFrame,
) -> None:
    """SHAP 결과 기반 종합 엔지니어 의견을 생성합니다."""

    st.markdown(
        "#### AI Engineer Opinion"
    )

    if probability >= 80:
        risk_statement = (
            "현재 예측 위험도가 매우 높으므로 "
            "생산 지속 전 즉시 설비 점검이 필요합니다."
        )

    elif probability >= 60:
        risk_statement = (
            "현재 예측 위험도가 높은 수준이므로 "
            "다음 생산 LOT 투입 전에 주요 센서와 "
            "소모품 상태를 점검해야 합니다."
        )

    elif probability >= 40:
        risk_statement = (
            "현재 예측 위험도는 중간 수준으로, "
            "공정 중점 모니터링과 예방점검이 필요합니다."
        )

    else:
        risk_statement = (
            "현재 예측 위험도는 낮은 수준이지만 "
            "추세 변화 여부를 지속적으로 확인해야 합니다."
        )

    risk_feature_names = (
        top_risk_factors["feature_label"]
        .astype(str)
        .tolist()
        if not top_risk_factors.empty
        else []
    )

    protective_feature_names = (
        top_protective_factors["feature_label"]
        .astype(str)
        .tolist()
        if not top_protective_factors.empty
        else []
    )

    if risk_feature_names:
        risk_feature_text = ", ".join(
            risk_feature_names
        )

        cause_statement = (
            f"주요 위험 증가 요인은 "
            f"{risk_feature_text}입니다."
        )

    else:
        cause_statement = (
            "현재 SHAP 분석에서 단일 인자가 위험을 "
            "뚜렷하게 증가시키는 현상은 확인되지 않았습니다."
        )

    if protective_feature_names:
        protective_feature_text = ", ".join(
            protective_feature_names
        )

        protective_statement = (
            f"반면 {protective_feature_text}은 "
            "예측 위험을 낮추는 방향으로 작용했습니다."
        )

    else:
        protective_statement = (
            "위험도를 유의미하게 낮추는 보호 인자는 "
            "현재 명확하지 않습니다."
        )

    st.info(
        f"**{equipment_id} 분석 결과:** "
        f"{risk_statement} "
        f"{cause_statement} "
        f"{protective_statement} "
        "SHAP 결과는 인과관계의 확정이 아니라 모델 예측에 "
        "기여한 방향을 의미하므로, 실제 점검 시에는 센서 원시값, "
        "필터 교체 이력, Chemical 공급 상태 및 최근 알람 이력을 "
        "함께 확인해야 합니다."
    )


def render_shap_detail_table(
    equipment_dataframe: pd.DataFrame,
) -> None:
    """선택 설비의 SHAP 상세 결과를 표로 표시합니다."""

    st.markdown(
        "#### SHAP 상세 데이터"
    )

    detail_columns = [
        "importance_rank",
        "feature_label",
        "feature_value",
        "shap_value",
        "absolute_shap_value",
        "contribution_direction",
        "engineer_interpretation",
    ]

    available_columns = [
        column
        for column in detail_columns
        if column in equipment_dataframe.columns
    ]

    detail_dataframe = (
        equipment_dataframe[
            available_columns
        ]
        .sort_values(
            by="importance_rank",
            ascending=True,
        )
        .copy()
    )

    detail_dataframe = detail_dataframe.rename(
        columns={
            "importance_rank": "순위",
            "feature_label": "영향 인자",
            "feature_value": "현재 값",
            "shap_value": "SHAP 값",
            "absolute_shap_value": "절대 영향도",
            "contribution_direction": "영향 방향",
            "engineer_interpretation": "엔지니어 해석",
        }
    )

    st.dataframe(
        detail_dataframe,
        use_container_width=True,
        hide_index=True,
        column_config={
            "현재 값": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "SHAP 값": st.column_config.NumberColumn(
                format="%+.4f"
            ),
            "절대 영향도": st.column_config.NumberColumn(
                format="%.4f"
            ),
        },
    )



def render_predictive_maintenance_pdf(
    prediction_dataframe: pd.DataFrame,
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
    shap_dataframe: pd.DataFrame,
    selected_equipment: str,
) -> None:
    """현재 화면의 분석 결과를 Predictive Maintenance PDF로 제공합니다."""

    st.markdown(
        '<div class="pm-section-title">'
        "Predictive Maintenance PDF Report"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pm-section-description">
            현재 선택된 설비 범위의 예측 결과, 예방보전 권장 조치,
            모델 성능, Feature Importance 및 SHAP 분석을 PDF로 생성합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_key = (
        "predictive_maintenance_pdf_"
        f"{selected_equipment}"
    )

    if st.button(
        "PDF 보고서 생성",
        type="primary",
        use_container_width=True,
        key="generate_predictive_maintenance_pdf",
    ):
        try:
            with st.spinner(
                "Predictive Maintenance PDF 보고서를 생성하는 중입니다..."
            ):
                report_path = generate_predictive_maintenance_report(
                    prediction_dataframe=prediction_dataframe,
                    metrics=metrics,
                    feature_importance=feature_importance,
                    shap_dataframe=shap_dataframe,
                    selected_equipment=selected_equipment,
                )

                st.session_state[session_key] = {
                    "bytes": report_path.read_bytes(),
                    "name": report_path.name,
                }

            st.success(
                "Predictive Maintenance PDF 보고서가 생성되었습니다."
            )

        except Exception as error:
            st.error(
                "PDF 보고서 생성 중 오류가 발생했습니다."
            )
            st.exception(error)

    report_data = st.session_state.get(
        session_key
    )

    if report_data:
        st.download_button(
            label="PDF 보고서 다운로드",
            data=report_data["bytes"],
            file_name=report_data["name"],
            mime="application/pdf",
            use_container_width=True,
            key="download_predictive_maintenance_pdf",
        )

def main() -> None:
    """Predictive Maintenance Dashboard 전체 화면을 실행합니다."""

    configure_page()
    apply_custom_css()
    render_header()

    try:
        with st.spinner(
            "Predictive Maintenance 예측 결과를 불러오는 중입니다..."
        ):
            prediction_dataframe = (
                load_prediction_data()
            )

        if prediction_dataframe.empty:
            st.warning(
                "예측 결과 데이터가 없습니다."
            )
            return

        (
            selected_equipment,
            importance_count,
            shap_count,
        ) = render_sidebar(
            prediction_dataframe=prediction_dataframe
        )

        filtered_dataframe = filter_prediction_data(
            prediction_dataframe=prediction_dataframe,
            selected_equipment=selected_equipment,
        )

        if filtered_dataframe.empty:
            st.warning(
                "선택한 설비의 예측 결과가 없습니다."
            )
            return

        render_prediction_kpis(
            prediction_dataframe=filtered_dataframe
        )

        st.markdown(
            '<div class="pm-section-title">'
            "설비별 이상 발생 위험"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="pm-section-description">
                최신 LOT 시점에서 향후 10 LOT 이내
                이상 발생 가능성을 설비별로 비교합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        risk_chart = create_risk_probability_chart(
            prediction_dataframe=filtered_dataframe
        )

        st.plotly_chart(
            risk_chart,
            use_container_width=True,
        )

        render_equipment_recommendations(
            prediction_dataframe=filtered_dataframe
        )

        render_prediction_table(
            prediction_dataframe=filtered_dataframe
        )

        metrics = load_model_metrics()

        render_model_performance(
            metrics=metrics
        )

        feature_importance = (
            load_feature_importance()
        )

        render_feature_importance(
            feature_importance=feature_importance,
            importance_count=importance_count,
        )

        with st.spinner(
            "설비별 SHAP 예측 원인을 분석하는 중입니다..."
        ):
            shap_dataframe = load_shap_data()

        render_local_shap_analysis(
            shap_dataframe=shap_dataframe,
            selected_equipment=selected_equipment,
            shap_count=shap_count,
        )

        render_predictive_maintenance_pdf(
            prediction_dataframe=filtered_dataframe,
            metrics=metrics,
            feature_importance=feature_importance,
            shap_dataframe=shap_dataframe,
            selected_equipment=selected_equipment,
        )

    except FileNotFoundError as error:
        st.error(
            "필요한 모델 또는 데이터 파일을 찾을 수 없습니다."
        )

        st.code(
            str(error)
        )

        st.info(
            "아래 순서로 모듈을 실행한 뒤 "
            "페이지를 새로고침하세요."
        )

        st.code(
            "\n".join(
                [
                    (
                        r".\.venv\Scripts\python.exe "
                        r"-m src.predictive_maintenance."
                        r"feature_engineering"
                    ),
                    (
                        r".\.venv\Scripts\python.exe "
                        r"-m src.predictive_maintenance."
                        r"train_model"
                    ),
                    (
                        r".\.venv\Scripts\python.exe "
                        r"-m src.predictive_maintenance."
                        r"prediction_engine"
                    ),
                    (
                        r".\.venv\Scripts\python.exe "
                        r"-m src.predictive_maintenance."
                        r"shap_explainer"
                    ),
                ]
            ),
            language="powershell",
        )

    except Exception as error:
        st.error(
            "Predictive Maintenance Dashboard 실행 중 "
            "오류가 발생했습니다."
        )

        st.exception(error)


if __name__ == "__main__":
    main()