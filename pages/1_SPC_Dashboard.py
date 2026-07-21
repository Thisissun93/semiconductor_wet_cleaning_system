from __future__ import annotations

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


from src.data_processing.analysis_repository import (  # noqa: E402
    load_equipment_ids,
    load_lot_trend,
)


SPECIFICATIONS: Final[dict[str, dict[str, float | str]]] = {
    "yield_percent": {
        "label": "수율",
        "unit": "%",
        "lsl": 96.0,
        "usl": 100.0,
    },
    "particle_count": {
        "label": "Particle",
        "unit": "개",
        "lsl": 0.0,
        "usl": 14.0,
    },
    "filter_differential_pressure": {
        "label": "Filter 차압",
        "unit": "bar",
        "lsl": 0.6,
        "usl": 1.2,
    },
    "flow_rate": {
        "label": "유량",
        "unit": "L/min",
        "lsl": 18.0,
        "usl": 24.0,
    },
    "motor_current": {
        "label": "Motor Current",
        "unit": "A",
        "lsl": 4.0,
        "usl": 7.0,
    },
    "vibration": {
        "label": "진동",
        "unit": "mm/s",
        "lsl": 0.0,
        "usl": 2.5,
    },
}


def configure_page() -> None:
    """SPC 페이지 기본 설정을 적용합니다."""

    st.set_page_config(
        page_title="Wet Cleaning SPC Dashboard",
        page_icon="📈",
        layout="wide",
    )


def apply_custom_css() -> None:
    """SPC Dashboard용 CSS를 적용합니다."""

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

        .spc-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }

        .spc-description {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def load_spc_data(
    equipment_id: str | None,
) -> pd.DataFrame:
    """SPC 분석에 사용할 LOT 데이터를 조회합니다."""

    return load_lot_trend(
        equipment_id=equipment_id
    )


def calculate_spc_statistics(
    dataframe: pd.DataFrame,
    parameter: str,
    lsl: float,
    usl: float,
) -> dict[str, float | int | str]:
    """선택 변수의 SPC 통계량을 계산합니다."""

    values = (
        pd.to_numeric(
            dataframe[parameter],
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        raise ValueError(
            "선택한 변수에 분석 가능한 데이터가 없습니다."
        )

    mean_value = float(values.mean())

    standard_deviation = float(
        values.std(ddof=1)
    )

    ucl = mean_value + 3 * standard_deviation
    lcl = mean_value - 3 * standard_deviation

    if standard_deviation == 0:
        cp = 0.0
        cpk = 0.0

    else:
        cp = (
            usl - lsl
        ) / (
            6 * standard_deviation
        )

        cpu = (
            usl - mean_value
        ) / (
            3 * standard_deviation
        )

        cpl = (
            mean_value - lsl
        ) / (
            3 * standard_deviation
        )

        cpk = min(cpu, cpl)

    specification_out_count = int(
        (
            (values < lsl)
            | (values > usl)
        ).sum()
    )

    control_out_count = int(
        (
            (values < lcl)
            | (values > ucl)
        ).sum()
    )

    if cpk >= 1.67:
        process_grade = "Excellent"

    elif cpk >= 1.33:
        process_grade = "Capable"

    elif cpk >= 1.0:
        process_grade = "Marginal"

    else:
        process_grade = "Improvement Required"

    return {
        "mean": mean_value,
        "standard_deviation": standard_deviation,
        "ucl": ucl,
        "lcl": lcl,
        "cp": float(cp),
        "cpk": float(cpk),
        "specification_out_count": specification_out_count,
        "control_out_count": control_out_count,
        "process_grade": process_grade,
        "data_count": len(values),
    }


def render_sidebar() -> tuple[str | None, str, int]:
    """SPC 분석 조건을 선택합니다."""

    st.sidebar.title("SPC Filter")

    equipment_ids = load_equipment_ids()

    selected_equipment = st.sidebar.selectbox(
        "설비 선택",
        options=["전체 설비"] + equipment_ids,
    )

    parameter_options = {
        specification["label"]: parameter
        for parameter, specification
        in SPECIFICATIONS.items()
    }

    selected_label = st.sidebar.selectbox(
        "분석 변수",
        options=list(parameter_options.keys()),
    )

    lot_count = st.sidebar.slider(
        "분석 LOT 수",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
    )

    equipment_id = (
        None
        if selected_equipment == "전체 설비"
        else selected_equipment
    )

    parameter = parameter_options[selected_label]

    if st.sidebar.button(
        "데이터 새로고침",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    return equipment_id, parameter, lot_count


def render_header(
    equipment_id: str | None,
    parameter: str,
) -> None:
    """페이지 제목과 분석 대상을 표시합니다."""

    equipment_text = (
        equipment_id
        if equipment_id is not None
        else "전체 설비"
    )

    parameter_label = str(
        SPECIFICATIONS[parameter]["label"]
    )

    st.title("Wet Cleaning SPC Dashboard")

    st.caption(
        f"분석 대상: {equipment_text}"
        f" | 분석 변수: {parameter_label}"
    )


def render_spc_kpis(
    statistics: dict[str, float | int | str],
    unit: str,
) -> None:
    """SPC 핵심 지표를 표시합니다."""

    column1, column2, column3, column4 = st.columns(4)

    column1.metric(
        "평균",
        f"{float(statistics['mean']):.3f} {unit}",
    )

    column2.metric(
        "표준편차",
        f"{float(statistics['standard_deviation']):.3f}",
    )

    column3.metric(
        "Cp",
        f"{float(statistics['cp']):.3f}",
    )

    column4.metric(
        "Cpk",
        f"{float(statistics['cpk']):.3f}",
    )

    column5, column6, column7, column8 = st.columns(4)

    column5.metric(
        "UCL",
        f"{float(statistics['ucl']):.3f}",
    )

    column6.metric(
        "LCL",
        f"{float(statistics['lcl']):.3f}",
    )

    column7.metric(
        "규격 이탈 LOT",
        f"{int(statistics['specification_out_count'])}개",
    )

    column8.metric(
        "공정 판정",
        str(statistics["process_grade"]),
    )


def create_control_chart(
    dataframe: pd.DataFrame,
    parameter: str,
    statistics: dict[str, float | int | str],
    lsl: float,
    usl: float,
) -> go.Figure:
    """LOT별 관리도를 생성합니다."""

    chart_dataframe = dataframe.copy()

    chart_dataframe["specification_status"] = (
        "정상"
    )

    chart_dataframe.loc[
        (
            chart_dataframe[parameter] < lsl
        )
        | (
            chart_dataframe[parameter] > usl
        ),
        "specification_status",
    ] = "규격 이탈"

    figure = px.line(
        chart_dataframe,
        x="start_time",
        y=parameter,
        color="equipment_id",
        markers=True,
        hover_data=[
            "lot_id",
            "severity",
            "specification_status",
        ],
        labels={
            "start_time": "생산 시간",
            parameter: str(
                SPECIFICATIONS[parameter]["label"]
            ),
            "equipment_id": "설비",
        },
        title="Individual Control Chart",
    )

    figure.add_hline(
        y=float(statistics["mean"]),
        line_dash="solid",
        annotation_text="Mean",
    )

    figure.add_hline(
        y=float(statistics["ucl"]),
        line_dash="dash",
        annotation_text="UCL",
    )

    figure.add_hline(
        y=float(statistics["lcl"]),
        line_dash="dash",
        annotation_text="LCL",
    )

    figure.add_hline(
        y=usl,
        line_dash="dot",
        annotation_text="USL",
    )

    figure.add_hline(
        y=lsl,
        line_dash="dot",
        annotation_text="LSL",
    )

    figure.update_layout(
        height=500,
        legend_title_text="설비",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure


def create_moving_average_chart(
    dataframe: pd.DataFrame,
    parameter: str,
) -> go.Figure:
    """이동평균 추세 차트를 생성합니다."""

    chart_dataframe = dataframe.copy()

    chart_dataframe["moving_average"] = (
        chart_dataframe[parameter]
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_dataframe["start_time"],
            y=chart_dataframe[parameter],
            mode="lines+markers",
            name="실측값",
            marker=dict(size=5),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_dataframe["start_time"],
            y=chart_dataframe["moving_average"],
            mode="lines",
            name="5 LOT 이동평균",
            line=dict(width=3),
        )
    )

    figure.update_layout(
        title="Moving Average Trend",
        xaxis_title="생산 시간",
        yaxis_title=str(
            SPECIFICATIONS[parameter]["label"]
        ),
        height=420,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure


def create_capability_chart(
    dataframe: pd.DataFrame,
    parameter: str,
    lsl: float,
    usl: float,
) -> go.Figure:
    """공정능력 분포 차트를 생성합니다."""

    figure = px.histogram(
        dataframe,
        x=parameter,
        nbins=20,
        marginal="box",
        labels={
            parameter: str(
                SPECIFICATIONS[parameter]["label"]
            )
        },
        title="Process Capability Distribution",
    )

    figure.add_vline(
        x=lsl,
        line_dash="dash",
        annotation_text="LSL",
    )

    figure.add_vline(
        x=usl,
        line_dash="dash",
        annotation_text="USL",
    )

    figure.update_layout(
        height=420,
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure


def render_out_of_control_lots(
    dataframe: pd.DataFrame,
    parameter: str,
    statistics: dict[str, float | int | str],
    lsl: float,
    usl: float,
) -> None:
    """관리한계 또는 규격을 벗어난 LOT를 표시합니다."""

    ucl = float(statistics["ucl"])
    lcl = float(statistics["lcl"])

    abnormal_dataframe = dataframe[
        (
            dataframe[parameter] > ucl
        )
        | (
            dataframe[parameter] < lcl
        )
        | (
            dataframe[parameter] > usl
        )
        | (
            dataframe[parameter] < lsl
        )
    ].copy()

    st.markdown(
        '<div class="spc-section-title">'
        '관리·규격 이탈 LOT'
        '</div>',
        unsafe_allow_html=True,
    )

    if abnormal_dataframe.empty:
        st.success(
            "현재 분석 구간에서 관리한계 또는 규격을 "
            "벗어난 LOT가 없습니다."
        )
        return

    abnormal_dataframe["판정"] = "관리한계 이탈"

    abnormal_dataframe.loc[
        (
            abnormal_dataframe[parameter] > usl
        )
        | (
            abnormal_dataframe[parameter] < lsl
        ),
        "판정",
    ] = "규격 이탈"

    display_dataframe = abnormal_dataframe[
        [
            "lot_id",
            "equipment_id",
            "start_time",
            parameter,
            "severity",
            "판정",
        ]
    ].copy()

    display_dataframe["start_time"] = (
        display_dataframe["start_time"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    display_dataframe = display_dataframe.rename(
        columns={
            "lot_id": "LOT",
            "equipment_id": "설비",
            "start_time": "생산 시간",
            parameter: str(
                SPECIFICATIONS[parameter]["label"]
            ),
            "severity": "이상 등급",
        }
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
        hide_index=True,
    )


def render_engineer_interpretation(
    statistics: dict[str, float | int | str],
) -> None:
    """Cp와 Cpk 기반 공정능력 해석을 표시합니다."""

    cp = float(statistics["cp"])
    cpk = float(statistics["cpk"])

    if cpk >= 1.67:
        message = (
            "현재 공정은 충분한 공정능력을 확보하고 있습니다. "
            "현재 조건을 유지하면서 장기 추세를 관리하는 것이 적절합니다."
        )

    elif cpk >= 1.33:
        message = (
            "현재 공정은 양산에 적합한 수준의 공정능력을 보입니다. "
            "평균값 이동 여부를 지속적으로 확인해야 합니다."
        )

    elif cpk >= 1.0:
        message = (
            "공정능력이 경계 수준입니다. "
            "설비 간 편차와 중심값 이동 원인을 점검해야 합니다."
        )

    else:
        message = (
            "공정능력이 부족합니다. 규격 이탈 가능성이 높으므로 "
            "설비 조건, 소재 및 공정 중심값을 우선 개선해야 합니다."
        )

    centering_gap = cp - cpk

    if centering_gap >= 0.2:
        message += (
            " 또한 Cp보다 Cpk가 낮아 공정 평균이 "
            "규격 중심에서 벗어난 것으로 판단됩니다."
        )

    st.info(message)


def main() -> None:
    """SPC Dashboard를 실행합니다."""

    configure_page()
    apply_custom_css()

    try:
        equipment_id, parameter, lot_count = (
            render_sidebar()
        )

        dataframe = load_spc_data(
            equipment_id=equipment_id
        )

        if dataframe.empty:
            st.warning("SPC 분석 데이터가 없습니다.")
            return

        dataframe = (
            dataframe.sort_values("start_time")
            .tail(lot_count)
            .copy()
        )

        specification = SPECIFICATIONS[parameter]

        lsl = float(specification["lsl"])
        usl = float(specification["usl"])
        unit = str(specification["unit"])

        statistics = calculate_spc_statistics(
            dataframe=dataframe,
            parameter=parameter,
            lsl=lsl,
            usl=usl,
        )

        render_header(
            equipment_id=equipment_id,
            parameter=parameter,
        )

        render_spc_kpis(
            statistics=statistics,
            unit=unit,
        )

        render_engineer_interpretation(
            statistics=statistics
        )

        st.markdown(
            '<div class="spc-section-title">'
            'SPC 관리도'
            '</div>',
            unsafe_allow_html=True,
        )

        control_chart = create_control_chart(
            dataframe=dataframe,
            parameter=parameter,
            statistics=statistics,
            lsl=lsl,
            usl=usl,
        )

        st.plotly_chart(
            control_chart,
            use_container_width=True,
        )

        column1, column2 = st.columns(2)

        with column1:
            moving_average_chart = (
                create_moving_average_chart(
                    dataframe=dataframe,
                    parameter=parameter,
                )
            )

            st.plotly_chart(
                moving_average_chart,
                use_container_width=True,
            )

        with column2:
            capability_chart = (
                create_capability_chart(
                    dataframe=dataframe,
                    parameter=parameter,
                    lsl=lsl,
                    usl=usl,
                )
            )

            st.plotly_chart(
                capability_chart,
                use_container_width=True,
            )

        render_out_of_control_lots(
            dataframe=dataframe,
            parameter=parameter,
            statistics=statistics,
            lsl=lsl,
            usl=usl,
        )

    except Exception as error:
        st.error(
            "SPC Dashboard 실행 중 오류가 발생했습니다."
        )
        st.exception(error)


if __name__ == "__main__":
    main()