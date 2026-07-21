from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_processing.analysis_repository import (  # noqa: E402
    load_equipment_summary,
    load_lot_trend,
    load_overall_kpis,
    load_recent_lot_summary,
    load_top_actions,
    load_top_root_causes,
)


REPORT_DIRECTORY: Final[Path] = PROJECT_ROOT / "reports"
TEMP_DIRECTORY: Final[Path] = REPORT_DIRECTORY / "temp"

FONT_NAME: Final[str] = "MalgunGothic"
FONT_BOLD_NAME: Final[str] = "MalgunGothicBold"

STATUS_ORDER: Final[dict[str, int]] = {
    "NORMAL": 1,
    "WATCH": 2,
    "WARNING": 3,
    "CRITICAL": 4,
}


def register_korean_fonts() -> None:
    """
    PDF에서 한글이 정상적으로 출력되도록
    Windows의 맑은 고딕 폰트를 등록합니다.
    """

    regular_font_paths = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
    ]

    bold_font_paths = [
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]

    regular_font_path = next(
        (
            path
            for path in regular_font_paths
            if path.exists()
        ),
        None,
    )

    bold_font_path = next(
        (
            path
            for path in bold_font_paths
            if path.exists()
        ),
        None,
    )

    if regular_font_path is None:
        raise FileNotFoundError(
            "맑은 고딕 폰트를 찾을 수 없습니다.\n"
            "확인 경로: C:/Windows/Fonts/malgun.ttf"
        )

    if bold_font_path is None:
        raise FileNotFoundError(
            "맑은 고딕 Bold 폰트를 찾을 수 없습니다.\n"
            "확인 경로: C:/Windows/Fonts/malgunbd.ttf"
        )

    pdfmetrics.registerFont(
        TTFont(
            FONT_NAME,
            str(regular_font_path),
        )
    )

    pdfmetrics.registerFont(
        TTFont(
            FONT_BOLD_NAME,
            str(bold_font_path),
        )
    )
    
    # Matplotlib 한글 폰트 설정
    mpl.rcParams["font.family"] = "Malgun Gothic"
    mpl.rcParams["axes.unicode_minus"] = False

def prepare_directories() -> None:
    """보고서와 임시 차트 저장 폴더를 생성합니다."""

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_report_styles() -> dict[str, ParagraphStyle]:
    """Executive Report에 사용할 문단 스타일을 생성합니다."""

    base_styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            name="ReportTitle",
            parent=base_styles["Title"],
            fontName=FONT_BOLD_NAME,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            name="ReportSubtitle",
            parent=base_styles["Normal"],
            fontName=FONT_NAME,
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#667085"),
            spaceAfter=18,
        ),
        "heading1": ParagraphStyle(
            name="Heading1Korean",
            parent=base_styles["Heading1"],
            fontName=FONT_BOLD_NAME,
            fontSize=16,
            leading=21,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=10,
        ),
        "heading2": ParagraphStyle(
            name="Heading2Korean",
            parent=base_styles["Heading2"],
            fontName=FONT_BOLD_NAME,
            fontSize=12,
            leading=17,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=5,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            name="BodyKorean",
            parent=base_styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=16,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#374151"),
        ),
        "body_bold": ParagraphStyle(
            name="BodyBoldKorean",
            parent=base_styles["BodyText"],
            fontName=FONT_BOLD_NAME,
            fontSize=9.5,
            leading=16,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
        ),
        "small": ParagraphStyle(
            name="SmallKorean",
            parent=base_styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#667085"),
        ),
        "status": ParagraphStyle(
            name="StatusKorean",
            parent=base_styles["BodyText"],
            fontName=FONT_BOLD_NAME,
            fontSize=14,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
    }


def get_status_background(status: str) -> colors.Color:
    """상태에 맞는 배경색을 반환합니다."""

    status_colors = {
        "NORMAL": colors.HexColor("#2E7D32"),
        "WATCH": colors.HexColor("#F9A825"),
        "WARNING": colors.HexColor("#EF6C00"),
        "CRITICAL": colors.HexColor("#C62828"),
    }

    return status_colors.get(
        status,
        colors.HexColor("#64748B"),
    )


def create_status_table(
    status: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    """보고서 상단의 공정 상태 배너를 생성합니다."""

    messages = {
        "NORMAL": "현재 Wet Cleaning 공정은 정상 상태입니다.",
        "WATCH": "일부 지표에 대한 추세 관찰이 필요합니다.",
        "WARNING": "품질 또는 설비 이상에 대한 우선 점검이 필요합니다.",
        "CRITICAL": "위험 수준의 이상이 확인되어 즉시 조치가 필요합니다.",
    }

    message = messages.get(
        status,
        "공정 상태를 확인할 수 없습니다.",
    )

    table = Table(
        [
            [
                Paragraph(
                    f"공정 상태: {status}<br/>"
                    f"<font size='9'>{message}</font>",
                    styles["status"],
                )
            ]
        ],
        colWidths=[175 * mm],
        rowHeights=[22 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    get_status_background(status),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    get_status_background(status),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ]
        )
    )

    return table


def create_kpi_table(
    kpis: dict[str, float | int | str],
    styles: dict[str, ParagraphStyle],
) -> Table:
    """핵심 KPI를 2행 4열 카드 형태로 생성합니다."""

    total_lots = int(kpis["total_lots"])
    abnormal_lots = int(kpis["abnormal_lots"])

    abnormal_rate = 0.0

    if total_lots > 0:
        abnormal_rate = (
            abnormal_lots
            / total_lots
            * 100
        )

    kpi_items = [
        ("전체 LOT", f"{total_lots}개"),
        (
            "평균 수율",
            f"{float(kpis['average_yield']):.2f}%",
        ),
        (
            "평균 Particle",
            f"{float(kpis['average_particle']):.2f}",
        ),
        ("이상 LOT", f"{abnormal_lots}개"),
        (
            "CRITICAL LOT",
            f"{int(kpis['critical_lots'])}개",
        ),
        (
            "FAIL LOT",
            f"{int(kpis['fail_lots'])}개",
        ),
        (
            "Open Action",
            f"{int(kpis['open_actions'])}건",
        ),
        (
            "이상 LOT 비율",
            f"{abnormal_rate:.1f}%",
        ),
    ]

    rows: list[list[Paragraph]] = []

    for row_start in range(0, len(kpi_items), 4):
        row: list[Paragraph] = []

        for label, value in kpi_items[
            row_start:row_start + 4
        ]:
            cell_text = (
                f"<font size='8' color='#667085'>"
                f"{label}</font><br/>"
                f"<font size='14'><b>{value}</b></font>"
            )

            row.append(
                Paragraph(
                    cell_text,
                    styles["body"],
                )
            )

        rows.append(row)

    table = Table(
        rows,
        colWidths=[43.75 * mm] * 4,
        rowHeights=[20 * mm, 20 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#E2E8F0"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    return table


def create_equipment_table(
    dataframe: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    """설비별 품질 및 이상 현황 표를 생성합니다."""

    headers = [
        "설비",
        "LOT",
        "평균 수율",
        "Particle",
        "Filter 차압",
        "FAIL",
        "CRITICAL",
        "이상 건수",
    ]

    table_data: list[list[object]] = [
        [
            Paragraph(
                header,
                styles["body_bold"],
            )
            for header in headers
        ]
    ]

    for _, row in dataframe.iterrows():
        table_data.append(
            [
                str(row["equipment_id"]),
                int(row["lot_count"]),
                f"{float(row['average_yield']):.2f}%",
                f"{float(row['average_particle']):.2f}",
                f"{float(row['average_filter_dp']):.3f}",
                int(row["fail_lots"]),
                int(row["critical_lots"]),
                int(row["anomaly_count"]),
            ]
        )

    table = Table(
        table_data,
        colWidths=[
            20 * mm,
            15 * mm,
            25 * mm,
            23 * mm,
            28 * mm,
            18 * mm,
            22 * mm,
            22 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1F4E78"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    FONT_BOLD_NAME,
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    FONT_NAME,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F8FAFC"),
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def create_yield_particle_chart(
    dataframe: pd.DataFrame,
) -> Path:
    """최근 LOT 수율과 Particle 추세 차트를 생성합니다."""

    output_path = TEMP_DIRECTORY / "yield_particle_trend.png"

    chart_dataframe = (
        dataframe.sort_values("start_time")
        .tail(40)
        .copy()
    )

    figure, yield_axis = plt.subplots(
        figsize=(10, 4.6)
    )

    particle_axis = yield_axis.twinx()

    yield_axis.plot(
        chart_dataframe["start_time"],
        chart_dataframe["yield_percent"],
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="Yield",
    )

    particle_axis.plot(
        chart_dataframe["start_time"],
        chart_dataframe["particle_count"],
        marker="s",
        markersize=3,
        linewidth=1.2,
        label="Particle",
    )

    yield_axis.axhline(
        y=96.0,
        linestyle="--",
        linewidth=1,
        label="Yield 기준",
    )

    particle_axis.axhline(
        y=14,
        linestyle=":",
        linewidth=1,
        label="Particle 기준",
    )

    yield_axis.set_title(
        "최근 LOT 수율 및 Particle 추세"
    )

    yield_axis.set_xlabel("생산 시간")
    yield_axis.set_ylabel("수율(%)")
    particle_axis.set_ylabel("Particle")

    yield_axis.tick_params(
        axis="x",
        rotation=35,
        labelsize=7,
    )

    yield_axis.grid(
        axis="y",
        alpha=0.25,
    )

    lines1, labels1 = (
        yield_axis.get_legend_handles_labels()
    )

    lines2, labels2 = (
        particle_axis.get_legend_handles_labels()
    )

    yield_axis.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_equipment_chart(
    dataframe: pd.DataFrame,
) -> Path:
    """설비별 평균 수율과 Particle 비교 차트를 생성합니다."""

    output_path = TEMP_DIRECTORY / "equipment_comparison.png"

    equipment_ids = dataframe["equipment_id"]

    figure, yield_axis = plt.subplots(
        figsize=(9.5, 4.3)
    )

    particle_axis = yield_axis.twinx()

    position = range(len(dataframe))

    yield_axis.bar(
        [
            index - 0.18
            for index in position
        ],
        dataframe["average_yield"],
        width=0.36,
        label="평균 수율",
    )

    particle_axis.bar(
        [
            index + 0.18
            for index in position
        ],
        dataframe["average_particle"],
        width=0.36,
        alpha=0.65,
        label="평균 Particle",
    )

    yield_axis.set_xticks(
        list(position),
        equipment_ids,
    )

    yield_axis.set_ylabel("평균 수율(%)")
    particle_axis.set_ylabel("평균 Particle")

    yield_axis.set_title(
        "설비별 품질 지표 비교"
    )

    yield_axis.grid(
        axis="y",
        alpha=0.25,
    )

    lines1, labels1 = (
        yield_axis.get_legend_handles_labels()
    )

    lines2, labels2 = (
        particle_axis.get_legend_handles_labels()
    )

    yield_axis.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_root_cause_table(
    dataframe: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    """Root Cause TOP 5 표를 생성합니다."""

    table_data: list[list[object]] = [
        [
            "순위",
            "Root Cause",
            "1순위 판정",
            "발생 건수",
            "평균 기여도",
        ]
    ]

    for rank, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        table_data.append(
            [
                rank,
                Paragraph(
                    str(row["cause_name"]),
                    styles["body"],
                ),
                int(row["first_rank_count"]),
                int(row["occurrence_count"]),
                (
                    f"{float(row['average_contribution']):.2f}%"
                ),
            ]
        )

    table = Table(
        table_data,
        colWidths=[
            14 * mm,
            75 * mm,
            28 * mm,
            25 * mm,
            32 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        get_standard_table_style()
    )

    return table


def create_action_table(
    dataframe: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    """권장 조치 TOP 5 표를 생성합니다."""

    table_data: list[list[object]] = [
        [
            "순위",
            "권장 조치",
            "담당 부서",
            "1순위 추천",
            "URGENT",
        ]
    ]

    for rank, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        table_data.append(
            [
                rank,
                Paragraph(
                    str(row["action_name"]),
                    styles["body"],
                ),
                Paragraph(
                    str(row["responsible_department"]),
                    styles["body"],
                ),
                int(row["first_rank_count"]),
                int(row["urgent_count"]),
            ]
        )

    table = Table(
        table_data,
        colWidths=[
            14 * mm,
            78 * mm,
            38 * mm,
            24 * mm,
            20 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        get_standard_table_style()
    )

    return table


def get_standard_table_style() -> TableStyle:
    """보고서 표에 공통으로 사용할 스타일입니다."""

    return TableStyle(
        [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1F4E78"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                FONT_BOLD_NAME,
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                FONT_NAME,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#CBD5E1"),
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER",
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F8FAFC"),
                ],
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
        ]
    )


def create_engineer_opinion(
    recent_lots: pd.DataFrame,
) -> dict[str, str | float | int]:
    """가장 위험도가 높은 LOT의 엔지니어 의견을 생성합니다."""

    if recent_lots.empty:
        return {
            "equipment_id": "-",
            "lot_id": "-",
            "severity": "NORMAL",
            "yield_percent": 0.0,
            "particle_count": 0,
            "top_cause": "분석 결과 없음",
            "top_action": "권장 조치 없음",
            "priority": "-",
        }

    dataframe = recent_lots.copy()

    dataframe["severity_score"] = (
        dataframe["severity"]
        .map(STATUS_ORDER)
        .fillna(0)
    )

    target_row = dataframe.sort_values(
        by=[
            "severity_score",
            "start_time",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    return {
        "equipment_id": str(
            target_row["equipment_id"]
        ),
        "lot_id": str(target_row["lot_id"]),
        "severity": str(target_row["severity"]),
        "yield_percent": float(
            target_row["yield_percent"]
        ),
        "particle_count": int(
            target_row["particle_count"]
        ),
        "top_cause": (
            str(target_row["top_cause"])
            if pd.notna(target_row["top_cause"])
            else "추가 분석 필요"
        ),
        "top_action": (
            str(target_row["top_action"])
            if pd.notna(target_row["top_action"])
            else "현장 점검 필요"
        ),
        "priority": (
            str(target_row["priority"])
            if pd.notna(target_row["priority"])
            else "미정"
        ),
    }


def create_opinion_paragraph(
    opinion: dict[str, str | float | int],
    styles: dict[str, ParagraphStyle],
) -> Paragraph:
    """AI Engineer Opinion 문단을 생성합니다."""

    text = (
        f"<b>분석 대상:</b> "
        f"{opinion['equipment_id']} / "
        f"{opinion['lot_id']}<br/><br/>"

        f"최근 주요 이상 등급은 "
        f"<b>{opinion['severity']}</b>입니다.<br/>"

        f"해당 LOT의 수율은 "
        f"<b>{float(opinion['yield_percent']):.2f}%</b>, "
        f"Particle은 "
        f"<b>{int(opinion['particle_count'])}개</b>입니다.<br/>"

        f"가장 가능성이 높은 원인은 "
        f"<b>{opinion['top_cause']}</b>로 분석됐습니다.<br/>"

        f"우선 권장 조치는 "
        f"<b>{opinion['top_action']}</b>이며, "
        f"조치 우선순위는 "
        f"<b>{opinion['priority']}</b>입니다."
    )

    return Paragraph(
        text,
        styles["body"],
    )


def create_conclusion(
    kpis: dict[str, float | int | str],
    equipment_summary: pd.DataFrame,
    opinion: dict[str, str | float | int],
) -> str:
    """보고서 결론 문장을 생성합니다."""

    if equipment_summary.empty:
        worst_equipment = "-"
    else:
        ranked_equipment = (
            equipment_summary.copy()
        )

        ranked_equipment["risk_score"] = (
            ranked_equipment["critical_lots"] * 5
            + ranked_equipment["fail_lots"] * 3
            + ranked_equipment["anomaly_count"]
        )

        worst_equipment = str(
            ranked_equipment.sort_values(
                by="risk_score",
                ascending=False,
            ).iloc[0]["equipment_id"]
        )

    status = str(kpis["overall_status"])

    if status == "CRITICAL":
        first_sentence = (
            "현재 공정은 CRITICAL 상태로, "
            "즉각적인 현장 대응이 필요합니다."
        )
    elif status == "WARNING":
        first_sentence = (
            "현재 공정은 WARNING 상태로, "
            "품질 및 설비 조건의 우선 점검이 필요합니다."
        )
    elif status == "WATCH":
        first_sentence = (
            "현재 공정은 WATCH 상태로, "
            "주요 지표의 지속적인 추세 관찰이 필요합니다."
        )
    else:
        first_sentence = (
            "현재 공정은 정상 범위에서 운영되고 있습니다."
        )

    return (
        f"{first_sentence} "
        f"설비 위험도 기준 우선 점검 대상은 "
        f"{worst_equipment}이며, "
        f"대표 원인은 {opinion['top_cause']}입니다. "
        f"권장 조치인 {opinion['top_action']}를 수행한 후 "
        f"Filter 차압, Particle 및 수율의 정상 복귀 여부를 "
        f"확인해야 합니다."
    )


def add_page_number(
    canvas,
    document,
) -> None:
    """각 페이지 하단에 보고서명과 페이지 번호를 표시합니다."""

    canvas.saveState()

    canvas.setFont(
        FONT_NAME,
        8,
    )

    canvas.setFillColor(
        colors.HexColor("#667085")
    )

    canvas.drawString(
        20 * mm,
        12 * mm,
        "Wet Cleaning Executive Report",
    )

    canvas.drawRightString(
        190 * mm,
        12 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def generate_executive_report(
    equipment_id: str | None = None,
) -> Path:
    """
    Executive PDF Report를 생성합니다.

    Args:
        equipment_id:
            특정 설비 보고서를 만들 경우 설비 ID
            전체 설비이면 None

    Returns:
        생성된 PDF 파일 경로
    """

    register_korean_fonts()
    prepare_directories()

    kpis = load_overall_kpis()
    equipment_summary = load_equipment_summary()
    lot_trend = load_lot_trend(
        equipment_id=equipment_id
    )
    recent_lots = load_recent_lot_summary(
        limit=12,
        equipment_id=equipment_id,
    )
    top_causes = load_top_root_causes(
        limit=5,
        equipment_id=equipment_id,
    )
    top_actions = load_top_actions(
        limit=5,
        equipment_id=equipment_id,
    )

    if lot_trend.empty:
        raise RuntimeError(
            "Executive Report를 생성할 LOT 데이터가 없습니다."
        )

    report_time = datetime.now()

    target_name = (
        equipment_id
        if equipment_id is not None
        else "ALL"
    )

    output_path = (
        REPORT_DIRECTORY
        / (
            "wet_cleaning_executive_report_"
            f"{target_name}_"
            f"{report_time:%Y%m%d_%H%M%S}.pdf"
        )
    )

    yield_particle_chart = (
        create_yield_particle_chart(
            lot_trend
        )
    )

    equipment_chart = (
        create_equipment_chart(
            equipment_summary
        )
    )

    styles = create_report_styles()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Wet Cleaning Executive Report",
        author="Semiconductor Wet Cleaning Monitoring System",
    )

    story: list[object] = []

    story.append(
        Paragraph(
            "Wet Cleaning Executive Report",
            styles["title"],
        )
    )

    target_text = (
        equipment_id
        if equipment_id is not None
        else "전체 설비"
    )

    story.append(
        Paragraph(
            (
                f"분석 대상: {target_text}"
                f" | 생성 시각: "
                f"{report_time:%Y-%m-%d %H:%M:%S}"
            ),
            styles["subtitle"],
        )
    )

    story.append(
        create_status_table(
            status=str(kpis["overall_status"]),
            styles=styles,
        )
    )

    story.append(Spacer(1, 7 * mm))

    story.append(
        Paragraph(
            "1. Executive Summary",
            styles["heading1"],
        )
    )

    story.append(
        create_kpi_table(
            kpis=kpis,
            styles=styles,
        )
    )

    story.append(Spacer(1, 7 * mm))

    story.append(
        Paragraph(
            "2. 설비별 품질 및 이상 현황",
            styles["heading1"],
        )
    )

    story.append(
        create_equipment_table(
            dataframe=equipment_summary,
            styles=styles,
        )
    )

    story.append(Spacer(1, 6 * mm))

    story.append(
        Image(
            str(equipment_chart),
            width=170 * mm,
            height=77 * mm,
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "3. 최근 LOT 품질 추세",
            styles["heading1"],
        )
    )

    story.append(
        Paragraph(
            (
                "최근 생산 LOT의 수율과 Particle 변화를 함께 비교하여 "
                "설비 열화가 품질에 미치는 영향을 확인합니다."
            ),
            styles["body"],
        )
    )

    story.append(Spacer(1, 4 * mm))

    story.append(
        Image(
            str(yield_particle_chart),
            width=175 * mm,
            height=81 * mm,
        )
    )

    story.append(Spacer(1, 7 * mm))

    story.append(
        Paragraph(
            "4. 주요 Root Cause",
            styles["heading1"],
        )
    )

    if top_causes.empty:
        story.append(
            Paragraph(
                "Root Cause 분석 결과가 없습니다.",
                styles["body"],
            )
        )
    else:
        story.append(
            create_root_cause_table(
                dataframe=top_causes,
                styles=styles,
            )
        )

    story.append(Spacer(1, 7 * mm))

    story.append(
        Paragraph(
            "5. 우선 권장 조치",
            styles["heading1"],
        )
    )

    if top_actions.empty:
        story.append(
            Paragraph(
                "권장 조치 결과가 없습니다.",
                styles["body"],
            )
        )
    else:
        story.append(
            create_action_table(
                dataframe=top_actions,
                styles=styles,
            )
        )

    story.append(PageBreak())

    opinion = create_engineer_opinion(
        recent_lots
    )

    story.append(
        Paragraph(
            "6. AI Engineer Opinion",
            styles["heading1"],
        )
    )

    opinion_box = Table(
        [
            [
                create_opinion_paragraph(
                    opinion=opinion,
                    styles=styles,
                )
            ]
        ],
        colWidths=[175 * mm],
    )

    opinion_box.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
            ]
        )
    )

    story.append(opinion_box)

    story.append(Spacer(1, 10 * mm))

    story.append(
        Paragraph(
            "7. Conclusion",
            styles["heading1"],
        )
    )

    conclusion = create_conclusion(
        kpis=kpis,
        equipment_summary=equipment_summary,
        opinion=opinion,
    )

    conclusion_box = Table(
        [
            [
                Paragraph(
                    conclusion,
                    styles["body"],
                )
            ]
        ],
        colWidths=[175 * mm],
    )

    conclusion_box.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#FFF7ED"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    colors.HexColor("#F59E0B"),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
            ]
        )
    )

    story.append(conclusion_box)

    story.append(Spacer(1, 12 * mm))

    story.append(
        Paragraph(
            (
                "본 보고서는 가상 Wet Cleaning 생산·센서·품질 "
                "데이터를 기반으로 자동 생성된 포트폴리오용 "
                "Executive Report입니다."
            ),
            styles["small"],
        )
    )

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return output_path


def main() -> None:
    """터미널에서 Executive Report를 생성합니다."""

    try:
        report_path = generate_executive_report()

        print("=" * 80)
        print("Executive Report 생성 완료")
        print("=" * 80)
        print(f"파일 경로: {report_path}")
        print("=" * 80)

    except Exception as error:
        print("=" * 80)
        print("Executive Report 생성 실패")
        print("=" * 80)
        print(error)


if __name__ == "__main__":
    main()