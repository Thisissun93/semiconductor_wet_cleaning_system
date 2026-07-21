from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_processing.analysis_repository import (  # noqa: E402
    load_anomaly_distribution,
    load_equipment_ids,
    load_equipment_summary,
    load_lot_trend,
    load_overall_kpis,
    load_recent_lot_summary,
    load_top_actions,
    load_top_root_causes,
)


STATUS_STYLE: Final[dict[str, dict[str, str]]] = {
    "NORMAL": {
        "label": "NORMAL",
        "background": "#E8F5E9",
        "border": "#2E7D32",
        "text": "#1B5E20",
        "message": "현재 Wet Cleaning 공정은 정상 상태입니다.",
    },
    "WATCH": {
        "label": "WATCH",
        "background": "#FFF8E1",
        "border": "#F9A825",
        "text": "#8D6E00",
        "message": (
            "일부 공정 또는 설비 지표가 정상 범위를 벗어났습니다. "
            "추세 관찰이 필요합니다."
        ),
    },
    "WARNING": {
        "label": "WARNING",
        "background": "#FFF3E0",
        "border": "#EF6C00",
        "text": "#E65100",
        "message": (
            "품질 또는 설비 이상이 확인됐습니다. "
            "원인 점검과 우선 조치가 필요합니다."
        ),
    },
    "CRITICAL": {
        "label": "CRITICAL",
        "background": "#FFEBEE",
        "border": "#C62828",
        "text": "#B71C1C",
        "message": (
            "위험 수준의 이상이 확인됐습니다. "
            "관련 설비와 LOT에 즉시 조치해야 합니다."
        ),
    },
}


SEVERITY_ORDER: Final[dict[str, int]] = {
    "CRITICAL": 4,
    "WARNING": 3,
    "WATCH": 2,
    "NORMAL": 1,
}


def configure_page() -> None:
    """Streamlit 페이지의 기본 설정을 적용합니다."""

    st.set_page_config(
        page_title="Wet Cleaning Monitoring System",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css() -> None:
    """Dashboard 가독성을 위한 공통 CSS를 적용합니다."""

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

        [data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }

        .section-description {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .insight-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 10px;
        }

        .insight-title {
            font-weight: 700;
            margin-bottom: 7px;
        }

        .small-text {
            color: #667085;
            font-size: 0.88rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def get_dashboard_data(
    equipment_id: str | None,
) -> dict[str, object]:
    """Dashboard에 필요한 데이터를 한 번에 불러옵니다."""

    return {
        "kpis": load_overall_kpis(),
        "equipment_summary": load_equipment_summary(),
        "lot_trend": load_lot_trend(
            equipment_id=equipment_id
        ),
        "recent_lots": load_recent_lot_summary(
            limit=12,
            equipment_id=equipment_id,
        ),
        "top_causes": load_top_root_causes(
            limit=5,
            equipment_id=equipment_id,
        ),
        "top_actions": load_top_actions(
            limit=5,
            equipment_id=equipment_id,
        ),
        "anomaly_distribution": (
            load_anomaly_distribution(
                equipment_id=equipment_id
            )
        ),
    }


def render_sidebar() -> tuple[str | None, int]:
    """설비 및 최근 LOT 개수 필터를 표시합니다."""

    st.sidebar.title("Monitoring Filter")

    equipment_ids = load_equipment_ids()

    equipment_options = ["전체 설비"] + equipment_ids

    selected_equipment = st.sidebar.selectbox(
        "설비 선택",
        options=equipment_options,
        index=0,
    )

    recent_lot_count = st.sidebar.slider(
        "추세 표시 LOT 수",
        min_value=12,
        max_value=120,
        value=40,
        step=4,
    )

    if selected_equipment == "전체 설비":
        equipment_id = None
    else:
        equipment_id = selected_equipment

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **분석 흐름**

        생산 데이터  
        → 이상 감지  
        → Root Cause  
        → Action Recommendation
        """
    )

    if st.sidebar.button(
        "데이터 새로고침",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    return equipment_id, recent_lot_count


def render_header(
    selected_equipment: str | None,
) -> None:
    """Dashboard 제목과 선택 설비 정보를 표시합니다."""

    st.title("Semiconductor Wet Cleaning Monitoring System")

    if selected_equipment is None:
        target_text = "전체 설비"
    else:
        target_text = selected_equipment

    st.caption(
        "Executive Summary 기반 공정·설비·품질 통합 Monitoring "
        f"| 분석 대상: {target_text}"
    )


def render_status_banner(
    status: str,
) -> None:
    """전체 공정 상태를 강조해 표시합니다."""

    style = STATUS_STYLE.get(
        status,
        STATUS_STYLE["WATCH"],
    )

    st.markdown(
        f"""
        <div style="
            background-color: {style['background']};
            border-left: 8px solid {style['border']};
            border-radius: 12px;
            padding: 18px 22px;
            margin: 8px 0 18px 0;
        ">
            <div style="
                color: {style['text']};
                font-size: 1.5rem;
                font-weight: 800;
                margin-bottom: 5px;
            ">
                공정 상태: {style['label']}
            </div>
            <div style="
                color: {style['text']};
                font-size: 1rem;
            ">
                {style['message']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(
    kpis: dict[str, float | int | str],
) -> None:
    """전체 핵심 KPI를 표시합니다."""

    st.markdown(
        '<div class="section-title">Executive KPI</div>',
        unsafe_allow_html=True,
    )

    column1, column2, column3, column4 = st.columns(4)

    column1.metric(
        label="전체 LOT",
        value=f"{int(kpis['total_lots'])}개",
    )

    column2.metric(
        label="평균 수율",
        value=f"{float(kpis['average_yield']):.2f}%",
    )

    column3.metric(
        label="평균 Particle",
        value=f"{float(kpis['average_particle']):.2f}",
    )

    column4.metric(
        label="이상 LOT",
        value=f"{int(kpis['abnormal_lots'])}개",
    )

    column5, column6, column7, column8 = st.columns(4)

    column5.metric(
        label="CRITICAL LOT",
        value=f"{int(kpis['critical_lots'])}개",
    )

    column6.metric(
        label="FAIL LOT",
        value=f"{int(kpis['fail_lots'])}개",
    )

    column7.metric(
        label="Open Action",
        value=f"{int(kpis['open_actions'])}건",
    )

    abnormal_rate = 0.0

    if int(kpis["total_lots"]) > 0:
        abnormal_rate = (
            int(kpis["abnormal_lots"])
            / int(kpis["total_lots"])
            * 100
        )

    column8.metric(
        label="이상 LOT 비율",
        value=f"{abnormal_rate:.1f}%",
    )


def render_equipment_summary(
    dataframe: pd.DataFrame,
) -> None:
    """설비별 품질과 이상 현황을 표로 표시합니다."""

    st.markdown(
        '<div class="section-title">설비별 위험 현황</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            각 설비의 평균 수율, Particle, Filter 차압 및 이상 건수를
            비교합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dataframe.empty:
        st.info("설비별 요약 데이터가 없습니다.")
        return

    display_dataframe = dataframe.rename(
        columns={
            "equipment_id": "설비",
            "lot_count": "LOT 수",
            "average_yield": "평균 수율(%)",
            "average_particle": "평균 Particle",
            "average_filter_dp": "평균 Filter 차압",
            "average_flow_rate": "평균 유량",
            "average_vibration": "평균 진동",
            "fail_lots": "FAIL LOT",
            "hold_lots": "HOLD LOT",
            "critical_lots": "CRITICAL LOT",
            "anomaly_count": "이상 건수",
        }
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
        hide_index=True,
        column_config={
            "평균 수율(%)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "평균 Particle": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "평균 Filter 차압": (
                st.column_config.NumberColumn(
                    format="%.3f"
                )
            ),
            "평균 유량": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "평균 진동": st.column_config.NumberColumn(
                format="%.3f"
            ),
        },
    )


def render_yield_particle_trend(
    dataframe: pd.DataFrame,
    recent_lot_count: int,
) -> None:
    """최근 LOT의 수율과 Particle 추세를 표시합니다."""

    st.markdown(
        '<div class="section-title">최근 LOT 품질 추세</div>',
        unsafe_allow_html=True,
    )

    if dataframe.empty:
        st.info("LOT 추세 데이터가 없습니다.")
        return

    trend_dataframe = (
        dataframe.sort_values("start_time")
        .tail(recent_lot_count)
        .copy()
    )

    chart_column1, chart_column2 = st.columns(2)

    yield_figure = px.line(
        trend_dataframe,
        x="start_time",
        y="yield_percent",
        color="equipment_id",
        markers=True,
        hover_data=[
            "lot_id",
            "severity",
            "particle_count",
        ],
        labels={
            "start_time": "생산 시간",
            "yield_percent": "수율(%)",
            "equipment_id": "설비",
        },
        title="수율 추세",
    )

    yield_figure.add_hline(
        y=96.0,
        line_dash="dash",
        annotation_text="정상 기준 96%",
    )

    yield_figure.update_layout(
        height=420,
        legend_title_text="설비",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    chart_column1.plotly_chart(
        yield_figure,
        use_container_width=True,
    )

    particle_figure = px.line(
        trend_dataframe,
        x="start_time",
        y="particle_count",
        color="equipment_id",
        markers=True,
        hover_data=[
            "lot_id",
            "severity",
            "yield_percent",
        ],
        labels={
            "start_time": "생산 시간",
            "particle_count": "Particle",
            "equipment_id": "설비",
        },
        title="Particle 추세",
    )

    particle_figure.add_hline(
        y=14,
        line_dash="dash",
        annotation_text="정상 상한 14",
    )

    particle_figure.add_hline(
        y=25,
        line_dash="dot",
        annotation_text="Critical 기준 25",
    )

    particle_figure.update_layout(
        height=420,
        legend_title_text="설비",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    chart_column2.plotly_chart(
        particle_figure,
        use_container_width=True,
    )


def render_filter_relationship(
    dataframe: pd.DataFrame,
    recent_lot_count: int,
) -> None:
    """Filter 차압과 Particle 관계를 표시합니다."""

    st.markdown(
        '<div class="section-title">설비 열화와 품질 영향</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Filter 차압 상승과 Particle 증가가 함께 발생하는지
            확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dataframe.empty:
        st.info("설비 열화 분석 데이터가 없습니다.")
        return

    chart_dataframe = (
        dataframe.sort_values("start_time")
        .tail(recent_lot_count)
        .copy()
    )

    column1, column2 = st.columns(2)

    filter_figure = px.line(
        chart_dataframe,
        x="start_time",
        y="filter_differential_pressure",
        color="equipment_id",
        markers=True,
        hover_data=[
            "lot_id",
            "flow_rate",
            "particle_count",
            "severity",
        ],
        labels={
            "start_time": "생산 시간",
            "filter_differential_pressure": (
                "Filter 차압(bar)"
            ),
            "equipment_id": "설비",
        },
        title="Filter 차압 추세",
    )

    filter_figure.add_hline(
        y=1.20,
        line_dash="dash",
        annotation_text="정상 상한 1.20 bar",
    )

    filter_figure.add_hline(
        y=2.00,
        line_dash="dot",
        annotation_text="Critical 2.00 bar",
    )

    filter_figure.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    column1.plotly_chart(
        filter_figure,
        use_container_width=True,
    )

    scatter_figure = px.scatter(
        chart_dataframe,
        x="filter_differential_pressure",
        y="particle_count",
        color="equipment_id",
        size="vibration",
        size_max=9,
        symbol="severity",
        hover_data=[
            "lot_id",
            "yield_percent",
            "flow_rate",
        ],
        labels={
            "filter_differential_pressure": (
                "Filter 차압(bar)"
            ),
            "particle_count": "Particle",
            "equipment_id": "설비",
            "severity": "등급",
        },
        title="Filter 차압과 Particle 관계",
    )

    scatter_figure.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    scatter_figure.update_traces(
    marker=dict(
        opacity=0.75,
        line=dict(width=0),
    )
)

    column2.plotly_chart(
        scatter_figure,
        use_container_width=True,
    )


def render_anomaly_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """이상 등급별 발생 건수를 표시합니다."""

    if dataframe.empty:
        st.info("이상 등급 데이터가 없습니다.")
        return

    severity_dataframe = dataframe.copy()

    severity_dataframe["order"] = (
        severity_dataframe["severity"]
        .map(SEVERITY_ORDER)
        .fillna(0)
    )

    severity_dataframe = severity_dataframe.sort_values(
        by="order",
        ascending=False,
    )

    figure = px.bar(
        severity_dataframe,
        x="severity",
        y="anomaly_count",
        text="anomaly_count",
        labels={
            "severity": "이상 등급",
            "anomaly_count": "발생 건수",
        },
        title="이상 등급별 발생 현황",
    )

    figure.update_traces(
        textposition="outside"
    )

    figure.update_layout(
        height=390,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def render_root_causes(
    dataframe: pd.DataFrame,
) -> None:
    """주요 Root Cause를 표시합니다."""

    if dataframe.empty:
        st.info("Root Cause 분석 결과가 없습니다.")
        return

    for index, row in dataframe.iterrows():
        rank = index + 1

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">
                    {rank}. {row['cause_name']}
                </div>
                <div class="small-text">
                    1순위 판정 {int(row['first_rank_count'])}건
                    · 전체 발생 {int(row['occurrence_count'])}건
                    · 평균 기여도
                    {float(row['average_contribution']):.2f}%
                    · HIGH 신뢰도
                    {int(row['high_confidence_count'])}건
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_actions(
    dataframe: pd.DataFrame,
) -> None:
    """추천 조치 우선순위를 표시합니다."""

    if dataframe.empty:
        st.info("추천 조치 결과가 없습니다.")
        return

    for index, row in dataframe.iterrows():
        rank = index + 1

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">
                    {rank}. {row['action_name']}
                </div>
                <div class="small-text">
                    대상: {row['target']}
                    · 담당: {row['responsible_department']}
                    · 1순위 추천
                    {int(row['first_rank_count'])}건
                    · URGENT
                    {int(row['urgent_count'])}건
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analysis_summary(
    top_causes: pd.DataFrame,
    top_actions: pd.DataFrame,
    anomaly_distribution: pd.DataFrame,
) -> None:
    """원인, 조치, 이상 등급을 한 화면에 배치합니다."""

    st.markdown(
        '<div class="section-title">Root Cause & Action</div>',
        unsafe_allow_html=True,
    )

    column1, column2, column3 = st.columns(
        [1.1, 1.1, 1.0]
    )

    with column1:
        st.subheader("주요 원인")
        render_root_causes(top_causes)

    with column2:
        st.subheader("우선 조치")
        render_actions(top_actions)

    with column3:
        st.subheader("이상 분포")
        render_anomaly_distribution(
            anomaly_distribution
        )


def render_recent_lots(
    dataframe: pd.DataFrame,
) -> None:
    """최근 LOT 분석 결과를 표로 표시합니다."""

    st.markdown(
        '<div class="section-title">최근 LOT 상세</div>',
        unsafe_allow_html=True,
    )

    if dataframe.empty:
        st.info("최근 LOT 데이터가 없습니다.")
        return

    display_dataframe = dataframe.copy()

    display_dataframe["start_time"] = (
        display_dataframe["start_time"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    display_dataframe = display_dataframe.rename(
        columns={
            "lot_id": "LOT",
            "equipment_id": "설비",
            "recipe_id": "Recipe",
            "start_time": "생산 시간",
            "lot_status": "LOT 상태",
            "yield_percent": "수율(%)",
            "particle_count": "Particle",
            "anomaly_count": "이상 건수",
            "severity": "최고 등급",
            "top_cause": "1순위 원인",
            "contribution_percent": "원인 기여도(%)",
            "confidence_level": "신뢰도",
            "top_action": "1순위 조치",
            "priority": "조치 우선순위",
            "responsible_department": "담당 부서",
            "action_status": "조치 상태",
        }
    )

    selected_columns = [
        "LOT",
        "설비",
        "생산 시간",
        "수율(%)",
        "Particle",
        "최고 등급",
        "1순위 원인",
        "원인 기여도(%)",
        "1순위 조치",
        "조치 우선순위",
        "담당 부서",
    ]

    st.dataframe(
        display_dataframe[selected_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "수율(%)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "원인 기여도(%)": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
        },
    )


def render_ai_engineer_opinion(
    recent_lots: pd.DataFrame,
    lot_trend: pd.DataFrame,
) -> None:
    """최근 공정 추세를 분석해 엔지니어 의견을 생성합니다."""

    st.markdown(
        '<div class="section-title">AI Engineer Opinion</div>',
        unsafe_allow_html=True,
    )

    if recent_lots.empty or lot_trend.empty:
        st.info("AI Engineer Opinion을 생성할 데이터가 없습니다.")
        return

    recent_lots = recent_lots.copy()
    lot_trend = lot_trend.copy()

    recent_lots["severity_score"] = (
        recent_lots["severity"]
        .map(SEVERITY_ORDER)
        .fillna(0)
    )

    target_row = recent_lots.sort_values(
        by=[
            "severity_score",
            "start_time",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    equipment_id = str(target_row["equipment_id"])
    lot_id = str(target_row["lot_id"])
    severity = str(target_row["severity"])
    yield_percent = float(target_row["yield_percent"])
    particle_count = int(target_row["particle_count"])

    top_cause = (
        str(target_row["top_cause"])
        if pd.notna(target_row["top_cause"])
        else "추가 분석 필요"
    )

    top_action = (
        str(target_row["top_action"])
        if pd.notna(target_row["top_action"])
        else "현장 점검 필요"
    )

    priority = (
        str(target_row["priority"])
        if pd.notna(target_row["priority"])
        else "미정"
    )

    equipment_trend = (
        lot_trend[
            lot_trend["equipment_id"] == equipment_id
        ]
        .sort_values("start_time")
        .copy()
    )

    if equipment_trend.empty:
        st.info("선택 설비의 추세 데이터가 없습니다.")
        return

    recent_count = min(
        10,
        len(equipment_trend),
    )

    recent_window = equipment_trend.tail(
        recent_count
    )

    previous_window = equipment_trend.iloc[
        :-recent_count
    ].tail(recent_count)

    recent_yield_average = float(
        recent_window["yield_percent"].mean()
    )

    recent_particle_average = float(
        recent_window["particle_count"].mean()
    )

    recent_filter_dp_average = float(
        recent_window[
            "filter_differential_pressure"
        ].mean()
    )

    if previous_window.empty:
        previous_yield_average = recent_yield_average
        previous_particle_average = recent_particle_average
        previous_filter_dp_average = (
            recent_filter_dp_average
        )

    else:
        previous_yield_average = float(
            previous_window["yield_percent"].mean()
        )

        previous_particle_average = float(
            previous_window["particle_count"].mean()
        )

        previous_filter_dp_average = float(
            previous_window[
                "filter_differential_pressure"
            ].mean()
        )

    yield_change = (
        recent_yield_average
        - previous_yield_average
    )

    particle_change_rate = 0.0

    if previous_particle_average != 0:
        particle_change_rate = (
            (
                recent_particle_average
                - previous_particle_average
            )
            / previous_particle_average
            * 100
        )

    filter_dp_change_rate = 0.0

    if previous_filter_dp_average != 0:
        filter_dp_change_rate = (
            (
                recent_filter_dp_average
                - previous_filter_dp_average
            )
            / previous_filter_dp_average
            * 100
        )

    first_filter_dp = float(
        recent_window[
            "filter_differential_pressure"
        ].iloc[0]
    )

    last_filter_dp = float(
        recent_window[
            "filter_differential_pressure"
        ].iloc[-1]
    )

    if last_filter_dp > first_filter_dp:
        filter_trend_text = "상승"
    elif last_filter_dp < first_filter_dp:
        filter_trend_text = "하락"
    else:
        filter_trend_text = "유지"

    opinion_html = (
        f'<div class="insight-card">'
        f'<div class="insight-title">'
        f'분석 대상: {equipment_id} / {lot_id}'
        f'</div>'

        f'<div style="line-height: 1.9;">'

        f'최근 주요 이상 등급은 '
        f'<b>{severity}</b>입니다.<br>'

        f'{equipment_id}의 최근 '
        f'<b>{recent_count}개 LOT</b>를 분석한 결과, '
        f'Filter 차압은 '
        f'<b>{filter_trend_text}</b> 추세를 보였습니다.<br>'

        f'이전 구간 대비 평균 Filter 차압은 '
        f'<b>{filter_dp_change_rate:+.1f}%</b>, '
        f'Particle은 '
        f'<b>{particle_change_rate:+.1f}%</b> 변화했습니다.<br>'

        f'평균 수율은 이전 구간 대비 '
        f'<b>{yield_change:+.2f}%p</b> 변화했으며, '
        f'대표 이상 LOT의 수율은 '
        f'<b>{yield_percent:.2f}%</b>, '
        f'Particle은 '
        f'<b>{particle_count}개</b>입니다.<br><br>'

        f'Rule-based Root Cause Analysis 결과, '
        f'가장 가능성이 높은 원인은 '
        f'<b>{top_cause}</b>로 판단됩니다.<br>'

        f'우선 권장 조치는 '
        f'<b>{top_action}</b>이며, '
        f'조치 우선순위는 '
        f'<b>{priority}</b>입니다.<br>'

        f'조치 완료 후에는 '
        f'<b>Filter 차압, 유량, Particle 및 수율</b>의 '
        f'정상 복귀 여부를 확인해야 합니다.'

        f'</div>'
        f'</div>'
    )

    st.markdown(
        opinion_html,
        unsafe_allow_html=True,
    )

def render_footer() -> None:
    """Dashboard 하단 안내를 표시합니다."""

    st.divider()

    st.caption(
        "본 Dashboard는 가상 Wet Cleaning 생산·센서·품질 데이터를 "
        "기반으로 한 포트폴리오용 Monitoring System입니다."
    )


def main() -> None:
    """Streamlit Dashboard를 실행합니다."""

    configure_page()
    apply_custom_css()

    try:
        equipment_id, recent_lot_count = render_sidebar()

        dashboard_data = get_dashboard_data(
            equipment_id=equipment_id
        )

        kpis = dashboard_data["kpis"]
        equipment_summary = dashboard_data[
            "equipment_summary"
        ]
        lot_trend = dashboard_data["lot_trend"]
        recent_lots = dashboard_data["recent_lots"]
        top_causes = dashboard_data["top_causes"]
        top_actions = dashboard_data["top_actions"]
        anomaly_distribution = dashboard_data[
            "anomaly_distribution"
        ]

        render_header(equipment_id)

        render_status_banner(
            status=str(kpis["overall_status"])
        )

        render_kpis(kpis)
        render_equipment_summary(equipment_summary)

        render_yield_particle_trend(
            dataframe=lot_trend,
            recent_lot_count=recent_lot_count,
        )

        render_filter_relationship(
            dataframe=lot_trend,
            recent_lot_count=recent_lot_count,
        )

        render_analysis_summary(
            top_causes=top_causes,
            top_actions=top_actions,
            anomaly_distribution=(
                anomaly_distribution
            ),
        )

        render_ai_engineer_opinion(
            recent_lots=recent_lots,
            lot_trend=lot_trend,
        )

        render_recent_lots(
            dataframe=recent_lots
        )

        render_footer()

    except Exception as error:
        st.error(
            "Dashboard 데이터를 불러오는 중 오류가 발생했습니다."
        )

        st.exception(error)

        st.info(
            "먼저 프로젝트 최상위 터미널에서 "
            "`python run_pipeline.py`를 실행해 주세요."
        )


if __name__ == "__main__":
    main()