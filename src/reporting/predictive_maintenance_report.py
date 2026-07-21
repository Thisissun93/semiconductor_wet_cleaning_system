from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
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
REPORT_DIRECTORY: Final[Path] = PROJECT_ROOT / "reports"
TEMP_DIRECTORY: Final[Path] = REPORT_DIRECTORY / "temp"

FONT_NAME: Final[str] = "PMKorean"
FONT_BOLD_NAME: Final[str] = "PMKoreanBold"

RISK_ORDER: Final[dict[str, int]] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


def _find_font(candidates: list[Path]) -> Path:
    """Windows와 Linux 배포 환경에서 사용 가능한 한글 폰트를 찾습니다."""

    for candidate in candidates:
        if candidate.exists():
            return candidate

    linux_font_roots = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]

    preferred_names = [
        "NanumGothic.ttf",
        "NanumGothicBold.ttf",
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJK-Bold.ttc",
        "NotoSansKR-Regular.ttf",
        "NotoSansKR-Bold.ttf",
    ]

    for font_root in linux_font_roots:
        if not font_root.exists():
            continue

        for font_name in preferred_names:
            matches = list(font_root.rglob(font_name))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        "PDF 한글 폰트를 찾지 못했습니다. "
        "Streamlit Cloud에서는 저장소 루트의 packages.txt에 "
        "fonts-nanum을 추가한 뒤 앱을 재부팅하세요."
    )


def register_korean_fonts() -> None:
    """ReportLab과 Matplotlib에서 사용할 한글 폰트를 등록합니다."""

    regular_path = _find_font(
        [
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum-gothic/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
    )

    bold_path = _find_font(
        [
            Path("C:/Windows/Fonts/malgunbd.ttf"),
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
            Path("/usr/share/fonts/truetype/nanum-gothic/NanumGothicBold.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            regular_path,
        ]
    )

    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(FONT_NAME, str(regular_path))
        )

    if FONT_BOLD_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(FONT_BOLD_NAME, str(bold_path))
        )

    if "Windows" in str(regular_path):
        mpl.rcParams["font.family"] = "Malgun Gothic"
    elif "Nanum" in regular_path.name:
        mpl.rcParams["font.family"] = "NanumGothic"
    else:
        mpl.rcParams["font.family"] = "sans-serif"

    mpl.rcParams["axes.unicode_minus"] = False


def prepare_directories() -> None:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)


def create_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "PMTitle",
            parent=base["Title"],
            fontName=FONT_BOLD_NAME,
            fontSize=21,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "PMSubtitle",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#667085"),
            spaceAfter=15,
        ),
        "heading1": ParagraphStyle(
            "PMHeading1",
            parent=base["Heading1"],
            fontName=FONT_BOLD_NAME,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=7,
            spaceAfter=8,
        ),
        "heading2": ParagraphStyle(
            "PMHeading2",
            parent=base["Heading2"],
            fontName=FONT_BOLD_NAME,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=5,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "PMBody",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#374151"),
        ),
        "body_bold": ParagraphStyle(
            "PMBodyBold",
            parent=base["BodyText"],
            fontName=FONT_BOLD_NAME,
            fontSize=9,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
        ),
        "small": ParagraphStyle(
            "PMSmall",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#667085"),
        ),
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_background(risk_level: str) -> colors.Color:
    return {
        "CRITICAL": colors.HexColor("#B91C1C"),
        "HIGH": colors.HexColor("#EA580C"),
        "MEDIUM": colors.HexColor("#D97706"),
        "LOW": colors.HexColor("#2563EB"),
    }.get(risk_level, colors.HexColor("#64748B"))


def create_summary_table(
    prediction_dataframe: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    equipment_count = len(prediction_dataframe)
    highest_probability = _safe_float(
        prediction_dataframe["failure_probability_percent"].max()
    )
    average_probability = _safe_float(
        prediction_dataframe["failure_probability_percent"].mean()
    )
    critical_count = int(
        (prediction_dataframe["risk_level"] == "CRITICAL").sum()
    )
    high_count = int(
        prediction_dataframe["risk_level"]
        .isin(["CRITICAL", "HIGH"])
        .sum()
    )

    items = [
        ("분석 설비", f"{equipment_count}대"),
        ("최고 이상 확률", f"{highest_probability:.1f}%"),
        ("평균 이상 확률", f"{average_probability:.1f}%"),
        ("Critical 설비", f"{critical_count}대"),
        ("고위험 설비", f"{high_count}대"),
    ]

    row = [
        Paragraph(
            (
                f"<font size='8' color='#667085'>{label}</font><br/>"
                f"<font size='14'><b>{value}</b></font>"
            ),
            styles["body"],
        )
        for label, value in items
    ]

    table = Table(
        [row],
        colWidths=[35 * mm] * 5,
        rowHeights=[22 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    return table


def create_risk_table(
    prediction_dataframe: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    headers = [
        "설비",
        "기준 LOT",
        "이상 확률",
        "위험 등급",
        "현재 이상",
        "Particle",
        "수율",
    ]

    table_data: list[list[object]] = [
        [Paragraph(header, styles["body_bold"]) for header in headers]
    ]

    dataframe = prediction_dataframe.copy()
    dataframe["risk_score"] = (
        dataframe["risk_level"].map(RISK_ORDER).fillna(0)
    )
    dataframe = dataframe.sort_values(
        ["risk_score", "failure_probability_percent"],
        ascending=[False, False],
    )

    for _, row in dataframe.iterrows():
        table_data.append(
            [
                str(row.get("equipment_id", "-")),
                str(row.get("lot_id", "-")),
                f"{_safe_float(row.get('failure_probability_percent')):.1f}%",
                str(row.get("risk_level", "-")),
                str(row.get("severity", "-")),
                f"{_safe_float(row.get('particle_count')):.2f}",
                f"{_safe_float(row.get('yield_percent')):.2f}%",
            ]
        )

    table = Table(
        table_data,
        colWidths=[
            20 * mm,
            34 * mm,
            24 * mm,
            24 * mm,
            24 * mm,
            23 * mm,
            22 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD_NAME),
                ("FONTNAME", (0, 1), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F8FAFC")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    return table


def create_probability_chart(
    prediction_dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    dataframe = prediction_dataframe.copy()
    dataframe["risk_score"] = (
        dataframe["risk_level"].map(RISK_ORDER).fillna(0)
    )
    dataframe = dataframe.sort_values(
        ["risk_score", "failure_probability_percent"],
        ascending=[True, True],
    )

    figure, axis = plt.subplots(figsize=(9.5, 4.6))
    axis.barh(
        dataframe["equipment_id"].astype(str),
        dataframe["failure_probability_percent"].astype(float),
    )

    for index, value in enumerate(
        dataframe["failure_probability_percent"].astype(float)
    ):
        axis.text(
            min(value + 1.0, 96),
            index,
            f"{value:.1f}%",
            va="center",
            fontsize=8,
        )

    axis.axvline(40, linestyle=":", linewidth=1)
    axis.axvline(60, linestyle=":", linewidth=1)
    axis.axvline(80, linestyle=":", linewidth=1)
    axis.set_xlim(0, 100)
    axis.set_xlabel("향후 10 LOT 이내 이상 발생 확률 (%)")
    axis.set_ylabel("설비")
    axis.set_title("Equipment Failure Risk")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    return output_path


def create_feature_importance_chart(
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    if feature_importance.empty:
        return None

    dataframe = (
        feature_importance
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance", ascending=True)
        .copy()
    )

    labels = (
        dataframe.get("feature_label", dataframe["feature"])
        .astype(str)
        .str.replace("_", " ", regex=False)
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    axis.barh(labels, dataframe["importance"].astype(float))
    axis.set_xlabel("Feature Importance")
    axis.set_title("Random Forest Global Feature Importance")
    axis.grid(axis="x", alpha=0.2)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    return output_path


def create_shap_chart(
    shap_dataframe: pd.DataFrame,
    equipment_id: str,
    output_path: Path,
) -> Path | None:
    dataframe = shap_dataframe[
        shap_dataframe["equipment_id"].astype(str) == equipment_id
    ].copy()

    if dataframe.empty:
        return None

    dataframe["absolute_shap_value"] = pd.to_numeric(
        dataframe["absolute_shap_value"],
        errors="coerce",
    )
    dataframe["shap_value"] = pd.to_numeric(
        dataframe["shap_value"],
        errors="coerce",
    )
    dataframe = (
        dataframe.dropna(subset=["absolute_shap_value", "shap_value"])
        .sort_values("absolute_shap_value", ascending=False)
        .head(10)
        .sort_values("shap_value", ascending=True)
    )

    labels = dataframe["feature_label"].astype(str)

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.barh(labels, dataframe["shap_value"])
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("SHAP Value")
    axis.set_title(f"{equipment_id} Local SHAP Analysis")
    axis.grid(axis="x", alpha=0.2)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    return output_path


def create_recommendation_table(
    prediction_dataframe: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    headers = ["설비", "위험 등급", "이상 확률", "권장 조치"]

    table_data: list[list[object]] = [
        [Paragraph(header, styles["body_bold"]) for header in headers]
    ]

    dataframe = prediction_dataframe.copy()
    dataframe["risk_score"] = (
        dataframe["risk_level"].map(RISK_ORDER).fillna(0)
    )
    dataframe = dataframe.sort_values(
        ["risk_score", "failure_probability_percent"],
        ascending=[False, False],
    )

    for _, row in dataframe.iterrows():
        table_data.append(
            [
                str(row.get("equipment_id", "-")),
                str(row.get("risk_level", "-")),
                f"{_safe_float(row.get('failure_probability_percent')):.1f}%",
                Paragraph(
                    str(row.get("recommended_action", "추가 점검 필요")),
                    styles["body"],
                ),
            ]
        )

    table = Table(
        table_data,
        colWidths=[22 * mm, 25 * mm, 25 * mm, 103 * mm],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD_NAME),
                ("FONTNAME", (0, 1), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ALIGN", (0, 0), (2, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F8FAFC")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return table


def create_model_performance_table(
    metrics: dict[str, object],
    styles: dict[str, ParagraphStyle],
) -> Table:
    items = [
        ("Accuracy", _safe_float(metrics.get("accuracy"))),
        ("Precision", _safe_float(metrics.get("precision"))),
        ("Recall", _safe_float(metrics.get("recall"))),
        ("ROC-AUC", _safe_float(metrics.get("roc_auc"))),
    ]

    row = [
        Paragraph(
            (
                f"<font size='8' color='#667085'>{label}</font><br/>"
                f"<font size='14'><b>{value:.3f}</b></font>"
            ),
            styles["body"],
        )
        for label, value in items
    ]

    table = Table(
        [row],
        colWidths=[43.75 * mm] * 4,
        rowHeights=[21 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def create_engineer_opinion(
    prediction_dataframe: pd.DataFrame,
    shap_dataframe: pd.DataFrame,
) -> str:
    dataframe = prediction_dataframe.copy()
    dataframe["risk_score"] = (
        dataframe["risk_level"].map(RISK_ORDER).fillna(0)
    )
    target = dataframe.sort_values(
        ["risk_score", "failure_probability_percent"],
        ascending=[False, False],
    ).iloc[0]

    equipment_id = str(target["equipment_id"])
    probability = _safe_float(target["failure_probability_percent"])
    risk_level = str(target["risk_level"])

    target_shap = shap_dataframe[
        shap_dataframe["equipment_id"].astype(str) == equipment_id
    ].copy()

    risk_factors: list[str] = []
    if not target_shap.empty:
        target_shap["absolute_shap_value"] = pd.to_numeric(
            target_shap["absolute_shap_value"],
            errors="coerce",
        )
        risk_factors = (
            target_shap[
                target_shap["contribution_direction"] == "위험 증가"
            ]
            .sort_values("absolute_shap_value", ascending=False)
            .head(3)["feature_label"]
            .astype(str)
            .tolist()
        )

    factors_text = (
        ", ".join(risk_factors)
        if risk_factors
        else "단일 주요 위험 인자가 명확하지 않음"
    )

    return (
        f"{equipment_id}의 향후 10 LOT 이내 이상 발생 확률은 "
        f"{probability:.1f}%이며 위험 등급은 {risk_level}입니다. "
        f"주요 위험 증가 요인은 {factors_text}입니다. "
        "실제 현장 판단 시 Filter 교체 이력, Chemical 공급 상태, "
        "최근 Alarm 및 센서 원시 추세를 함께 확인해야 합니다."
    )


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(
        18 * mm,
        12 * mm,
        "Predictive Maintenance Report",
    )
    canvas.drawRightString(
        192 * mm,
        12 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def generate_predictive_maintenance_report(
    prediction_dataframe: pd.DataFrame,
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
    shap_dataframe: pd.DataFrame,
    selected_equipment: str = "전체 설비",
) -> Path:
    """현재 대시보드 데이터를 이용해 Predictive Maintenance PDF를 생성합니다."""

    if prediction_dataframe.empty:
        raise ValueError("PDF 보고서를 생성할 예측 데이터가 없습니다.")

    register_korean_fonts()
    prepare_directories()

    report_time = datetime.now()
    target_name = (
        "ALL"
        if selected_equipment == "전체 설비"
        else selected_equipment
    )

    output_path = (
        REPORT_DIRECTORY
        / (
            "predictive_maintenance_report_"
            f"{target_name}_{report_time:%Y%m%d_%H%M%S}.pdf"
        )
    )

    probability_chart = create_probability_chart(
        prediction_dataframe,
        TEMP_DIRECTORY / "pm_probability.png",
    )

    importance_chart = create_feature_importance_chart(
        feature_importance,
        TEMP_DIRECTORY / "pm_feature_importance.png",
    )

    highest_equipment = str(
        prediction_dataframe.sort_values(
            "failure_probability_percent",
            ascending=False,
        ).iloc[0]["equipment_id"]
    )

    shap_chart = create_shap_chart(
        shap_dataframe,
        highest_equipment,
        TEMP_DIRECTORY / "pm_shap.png",
    )

    styles = create_styles()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="Predictive Maintenance Report",
        author="Semiconductor Wet Cleaning Monitoring System",
    )

    story: list[object] = []

    story.append(
        Paragraph(
            "Predictive Maintenance Report",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            (
                f"분석 대상: {selected_equipment}"
                f" | 생성 시각: {report_time:%Y-%m-%d %H:%M:%S}"
            ),
            styles["subtitle"],
        )
    )

    top_row = prediction_dataframe.sort_values(
        "failure_probability_percent",
        ascending=False,
    ).iloc[0]
    top_risk = str(top_row["risk_level"])

    banner = Table(
        [
            [
                Paragraph(
                    (
                        f"<font size='15'><b>최고 위험 등급: {top_risk}</b></font><br/>"
                        f"<font size='9'>우선 점검 설비: "
                        f"{top_row['equipment_id']} / "
                        f"이상 확률 {_safe_float(top_row['failure_probability_percent']):.1f}%"
                        "</font>"
                    ),
                    ParagraphStyle(
                        "Banner",
                        parent=styles["body"],
                        fontName=FONT_NAME,
                        alignment=TA_CENTER,
                        textColor=colors.white,
                        leading=18,
                    ),
                )
            ]
        ],
        colWidths=[175 * mm],
        rowHeights=[23 * mm],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _risk_background(top_risk)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, _risk_background(top_risk)),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 7 * mm))

    story.append(Paragraph("1. Executive Summary", styles["heading1"]))
    story.append(create_summary_table(prediction_dataframe, styles))
    story.append(Spacer(1, 7 * mm))

    story.append(Paragraph("2. 설비별 이상 발생 위험", styles["heading1"]))
    story.append(
        Paragraph(
            "최신 LOT 기준 향후 10 LOT 이내 이상 발생 확률을 설비별로 비교합니다.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Image(
            str(probability_chart),
            width=174 * mm,
            height=84 * mm,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(create_risk_table(prediction_dataframe, styles))

    story.append(PageBreak())
    story.append(Paragraph("3. 예방보전 권장 조치", styles["heading1"]))
    story.append(create_recommendation_table(prediction_dataframe, styles))
    story.append(Spacer(1, 7 * mm))

    story.append(Paragraph("4. Model Performance", styles["heading1"]))
    if metrics:
        story.append(create_model_performance_table(metrics, styles))
        story.append(Spacer(1, 4 * mm))
        story.append(
            Paragraph(
                "예지보전 모델은 실제 이상을 정상으로 판정하는 False Negative가 중요하므로 "
                "Accuracy와 함께 Recall을 중점적으로 확인해야 합니다.",
                styles["body"],
            )
        )
    else:
        story.append(Paragraph("모델 평가 결과가 없습니다.", styles["body"]))

    if importance_chart is not None:
        story.append(Spacer(1, 7 * mm))
        story.append(
            Paragraph(
                "5. Global Feature Importance",
                styles["heading1"],
            )
        )
        story.append(
            Image(
                str(importance_chart),
                width=174 * mm,
                height=91 * mm,
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("6. SHAP Prediction Reason", styles["heading1"]))

    if shap_chart is not None:
        story.append(
            Paragraph(
                f"최고 위험 설비 {highest_equipment}의 Local SHAP 분석입니다. "
                "양수는 이상 위험 증가, 음수는 이상 위험 감소 방향을 의미합니다.",
                styles["body"],
            )
        )
        story.append(Spacer(1, 4 * mm))
        story.append(
            Image(
                str(shap_chart),
                width=174 * mm,
                height=95 * mm,
            )
        )
    else:
        story.append(
            Paragraph(
                "SHAP 분석 결과가 없습니다.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("7. AI Engineer Opinion", styles["heading1"]))

    opinion = create_engineer_opinion(
        prediction_dataframe,
        shap_dataframe,
    )

    opinion_box = Table(
        [[Paragraph(opinion, styles["body"])]],
        colWidths=[175 * mm],
    )
    opinion_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(opinion_box)
    story.append(Spacer(1, 10 * mm))

    story.append(
        Paragraph(
            (
                "본 보고서는 포트폴리오용 가상 Wet Cleaning 생산·센서·품질 "
                "데이터와 머신러닝 분석 결과를 기반으로 자동 생성되었습니다. "
                "SHAP은 모델 예측 기여도를 설명하며 실제 인과관계를 확정하지 않습니다."
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
