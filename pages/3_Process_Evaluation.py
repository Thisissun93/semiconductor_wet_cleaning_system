from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation import (  # noqa: E402
    FACTORS,
    RESPONSES,
    build_face_centered_ccd,
    confirmation_plan,
    fit_quadratic,
    predict_grid,
    recommend_condition,
    simulate_responses,
    simulate_wafer_map,
    uniformity_summary,
)
from src.evaluation.design import actual_to_coded  # noqa: E402


FACTOR_BY_KEY: Final = {factor.key: factor for factor in FACTORS}
RESPONSE_BY_KEY: Final = {response.key: response for response in RESPONSES}


def configure_page() -> None:
    """평가 페이지 기본 설정을 적용합니다."""

    st.set_page_config(
        page_title="Wet Cleaning Process Evaluation",
        page_icon="🧪",
        layout="wide",
    )


def apply_custom_css() -> None:
    """평가 페이지용 CSS를 적용합니다."""

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

        .eval-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }

        .eval-description {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, description: str) -> None:
    st.markdown(
        f'<div class="eval-section-title">{title}</div>'
        f'<div class="eval-description">{description}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, int, float]:
    """실험 조건 설정 사이드바."""

    st.sidebar.title("Evaluation Setup")
    center_points = st.sidebar.slider(
        "중심점 반복 수",
        min_value=2,
        max_value=6,
        value=4,
        help="중심점 반복으로 순수 오차와 곡률을 확인합니다.",
    )
    seed = int(
        st.sidebar.number_input(
            "합성 데이터 Seed",
            min_value=1,
            max_value=9999,
            value=11,
            step=1,
        )
    )
    noise = st.sidebar.slider(
        "측정 노이즈 배율",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
        help="노이즈가 커지면 예측구간이 넓어지고 확인 웨이퍼가 더 필요해집니다.",
    )
    st.sidebar.caption("모든 결과는 공개용 합성 데이터입니다.")
    return center_points, seed, noise


@st.cache_data
def run_evaluation(center_points: int, seed: int, noise: float):
    design = build_face_centered_ccd(center_points=center_points, seed=7)
    results = simulate_responses(design, seed=seed, noise_scale=noise)
    fits = {response.key: fit_quadratic(results, response) for response in RESPONSES}
    grid = predict_grid(fits, steps=21)
    return design, results, fits, grid


def render_header() -> None:
    st.title("Wet Cleaning Process Evaluation (DOE)")
    st.caption(
        "평가 웨이퍼 실험 계획 → 결과 회귀 → 판정 기준을 만족하는 조건 → 확인 실험 계획 "
        "| 공개용 합성 데이터 · 실제 장비·고객 조건 아님"
    )


def render_plan(design: pd.DataFrame) -> None:
    section(
        "1. 평가 인자와 판정 기준",
        "SC1 세정의 세 인자를 3수준으로 두고, 세정력·막 손실·패턴 손상을 함께 판정합니다.",
    )
    left, right = st.columns(2)
    with left:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "인자": f.label,
                        "단위": f.unit,
                        "-1": f.low,
                        "0": f.center,
                        "+1": f.high,
                    }
                    for f in FACTORS
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "평가 항목": r.label,
                        "단위": r.unit,
                        "판정 기준": (
                            f"≥ {r.spec:g}" if r.goal == "maximize" else f"≤ {r.spec:g}"
                        ),
                        "회귀 변환": {
                            "none": "없음",
                            "logit_pct": "logit (0~100% 포화)",
                            "sqrt": "제곱근 (개수 데이터)",
                        }[r.transform],
                    }
                    for r in RESPONSES
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    section(
        "2. 실험 계획표 (Face-centered CCD)",
        f"요인점 8 + 축점 6 + 중심점 {int((design['point_type'] == '중심점').sum())} "
        f"= {len(design)}매. 실행 순서는 무작위화해 장비 시간 변동이 특정 조건에 몰리지 않게 합니다.",
    )


def render_results(results: pd.DataFrame) -> None:
    columns = ["wafer_id", "run_order", "point_type"] + [f.key for f in FACTORS] + [
        r.key for r in RESPONSES
    ]
    renamed = results[columns].rename(
        columns={
            "wafer_id": "Wafer",
            "run_order": "실행 순서",
            "point_type": "점 종류",
            **{f.key: f"{f.label} ({f.unit})" for f in FACTORS},
            **{r.key: f"{r.label} ({r.unit})" for r in RESPONSES},
        }
    )
    st.dataframe(renamed, hide_index=True, use_container_width=True, height=360)
    st.download_button(
        "실험 계획·결과 CSV 내려받기",
        renamed.to_csv(index=False).encode("utf-8-sig"),
        file_name="wet_clean_doe_results.csv",
        mime="text/csv",
    )


def render_model_quality(fits: dict) -> None:
    section(
        "3. 회귀 모델 품질",
        "2차 반응표면(주효과·교호작용·제곱항)으로 적합했습니다. "
        "조정 R²가 낮은 항목은 예측구간이 넓어 확인 실험에서 더 보수적으로 봅니다.",
    )
    cols = st.columns(len(RESPONSES))
    for col, response in zip(cols, RESPONSES):
        fit = fits[response.key]
        col.metric(
            response.label,
            f"조정 R² {fit.adj_r_squared:.3f}",
            f"R² {fit.r_squared:.3f}",
            delta_color="off",
        )

    selected = st.selectbox(
        "계수표를 볼 항목",
        options=[r.key for r in RESPONSES],
        format_func=lambda key: RESPONSE_BY_KEY[key].label,
    )
    table = fits[selected].coefficient_table()
    st.dataframe(
        table.style.format({"계수": "{:.4f}", "표준오차": "{:.4f}", "t": "{:.2f}", "p": "{:.4f}"}),
        hide_index=True,
        use_container_width=True,
    )
    if RESPONSE_BY_KEY[selected].transform != "none":
        st.caption("변환한 척도에서의 계수입니다. 부호와 유의성을 해석에 사용합니다.")


def contour_figure(fits: dict, response_key: str, nh4oh_coded: float) -> go.Figure:
    """NH4OH 농도를 고정한 온도 × 메가소닉 출력 등고선."""

    steps = 41
    levels = np.linspace(-1.0, 1.0, steps)
    t_grid, p_grid = np.meshgrid(levels, levels)
    coded = np.column_stack(
        [t_grid.ravel(), np.full(t_grid.size, nh4oh_coded), p_grid.ravel()]
    )
    response = RESPONSE_BY_KEY[response_key]
    z = fits[response_key].predict(coded).reshape(t_grid.shape)

    t_factor = FACTOR_BY_KEY["temperature_c"]
    p_factor = FACTOR_BY_KEY["megasonic_w"]
    x_axis = t_factor.center + levels * t_factor.half_range
    y_axis = p_factor.center + levels * p_factor.half_range

    feasible = np.ones(t_grid.size, dtype=bool)
    for other in RESPONSES:
        predicted = fits[other.key].predict(coded)
        feasible &= (
            predicted >= other.spec if other.goal == "maximize" else predicted <= other.spec
        )

    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=x_axis,
            y=y_axis,
            z=z,
            colorscale="Teal" if response.goal == "maximize" else "Oranges",
            contours={"showlabels": True},
            colorbar={"title": response.unit},
            name=response.label,
        )
    )
    figure.add_trace(
        go.Contour(
            x=x_axis,
            y=y_axis,
            z=feasible.reshape(t_grid.shape).astype(float),
            contours={"start": 0.5, "end": 0.5, "coloring": "none"},
            line={"color": "#111827", "width": 3, "dash": "dash"},
            showscale=False,
            hoverinfo="skip",
            name="판정 기준 모두 만족",
        )
    )
    figure.update_layout(
        title=f"{response.label} — NH4OH 고정, 점선 안쪽이 세 기준 모두 만족",
        xaxis_title=f"{t_factor.label} ({t_factor.unit})",
        yaxis_title=f"{p_factor.label} ({p_factor.unit})",
        height=430,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return figure


def render_contours(fits: dict, recommended: pd.Series | None) -> None:
    section(
        "4. 조건 창(Process Window)",
        "세정력은 출력·온도를 올릴수록 좋아지지만, 막 손실과 패턴 손상이 같이 늘어납니다. "
        "점선 안쪽이 세 판정 기준을 동시에 만족하는 영역입니다.",
    )
    nh4 = FACTOR_BY_KEY["nh4oh_wt_pct"]
    default = float(recommended["nh4oh_wt_pct"]) if recommended is not None else nh4.center
    nh4_value = st.slider(
        f"{nh4.label} 고정값 ({nh4.unit})",
        min_value=nh4.low,
        max_value=nh4.high,
        value=default,
        step=0.04,
    )
    nh4_coded = float(actual_to_coded(pd.DataFrame({
        "temperature_c": [FACTOR_BY_KEY["temperature_c"].center],
        "nh4oh_wt_pct": [nh4_value],
        "megasonic_w": [FACTOR_BY_KEY["megasonic_w"].center],
    }))[0, 1])

    cols = st.columns(len(RESPONSES))
    for col, response in zip(cols, RESPONSES):
        with col:
            st.plotly_chart(
                contour_figure(fits, response.key, nh4_coded),
                use_container_width=True,
            )


def render_recommendation(fits: dict, recommended: pd.Series | None) -> None:
    section(
        "5. 권장 조건과 확인 실험 계획",
        "격자 탐색으로 세 기준을 모두 만족하는 조건 중 종합 만족도가 가장 높은 조건을 고르고, "
        "새 웨이퍼의 95% 예측구간이 기준 안에 드는지 확인합니다.",
    )
    if recommended is None:
        st.warning(
            "현재 모델에서는 세 기준을 동시에 만족하는 조건이 없습니다. "
            "인자 범위를 넓히거나 판정 기준·하드웨어 조건을 재검토해야 합니다."
        )
        return

    cols = st.columns(len(FACTORS) + 1)
    for col, factor in zip(cols, FACTORS):
        col.metric(factor.label, f"{recommended[factor.key]:.4g} {factor.unit}")
    cols[-1].metric("종합 만족도", f"{recommended['desirability']:.2f}")

    st.dataframe(
        confirmation_plan(fits, recommended),
        hide_index=True,
        use_container_width=True,
    )


def render_wafer_map(recommended: pd.Series | None) -> None:
    if recommended is None:
        return
    section(
        "6. 권장 조건 49점 식각량 지도",
        "매엽 스핀 세정에서 약액이 중심에서 바깥으로 퍼지며 식는다고 가정한 시연용 분포입니다. "
        "온도가 높을수록 중심이 더 많이 식각되는 경향을 보여 줍니다.",
    )
    wafer = simulate_wafer_map(
        mean_loss_a=float(recommended["oxide_loss_a"]),
        temperature_coded=float(recommended["temperature_c_coded"]),
    )
    summary = uniformity_summary(wafer)

    left, right = st.columns([3, 2])
    with left:
        theta = np.linspace(0, 2 * np.pi, 200)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=150 * np.cos(theta),
                y=150 * np.sin(theta),
                mode="lines",
                line={"color": "#9CA3AF"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=wafer["x_mm"],
                y=wafer["y_mm"],
                mode="markers+text",
                text=wafer["oxide_loss_a"].map(lambda v: f"{v:.2f}"),
                textposition="top center",
                marker={
                    "size": 14,
                    "color": wafer["oxide_loss_a"],
                    "colorscale": "Oranges",
                    "colorbar": {"title": "Å"},
                    "line": {"width": 0.5, "color": "#374151"},
                },
                hovertemplate="Site %{customdata}<br>%{marker.color:.3f} Å<extra></extra>",
                customdata=wafer["site"],
                showlegend=False,
            )
        )
        figure.update_layout(
            height=520,
            xaxis={"visible": False, "scaleanchor": "y"},
            yaxis={"visible": False},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(figure, use_container_width=True)
    with right:
        st.metric("평균 식각 손실", f"{summary['mean']:.2f} Å")
        st.metric("비균일도 (1σ)", f"{summary['nu_1sigma_pct']:.2f} %")
        st.metric("비균일도 (Range/2·Mean)", f"{summary['nu_range_pct']:.2f} %")
        st.caption(
            "확인 실험에서는 평균값과 함께 균일도를 같이 판정합니다. "
            "중심-가장자리 차이가 크면 노즐 스캔·회전수·약액 온도 보상을 다음 평가 인자로 검토합니다."
        )


def main() -> None:
    configure_page()
    apply_custom_css()
    render_header()

    try:
        center_points, seed, noise = render_sidebar()
        design, results, fits, grid = run_evaluation(center_points, seed, noise)
        recommended = recommend_condition(grid)

        render_plan(design)
        render_results(results)
        render_model_quality(fits)
        render_contours(fits, recommended)
        render_recommendation(fits, recommended)
        render_wafer_map(recommended)

    except Exception as error:
        st.error("Process Evaluation 실행 중 오류가 발생했습니다.")
        st.exception(error)


if __name__ == "__main__":
    main()
