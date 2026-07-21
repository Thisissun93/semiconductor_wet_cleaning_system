from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)

ANALYSIS_METHOD: Final[str] = "RULE_BASED_RCA"

SEVERITY_WEIGHT: Final[dict[str, float]] = {
    "WATCH": 1.0,
    "WARNING": 2.0,
    "CRITICAL": 3.0,
}


ROOT_CAUSE_RULES: Final[dict[str, dict[str, Any]]] = {
    "FILTER_CLOGGING": {
        "cause_name": "Filter 막힘 및 교체주기 초과",
        "parameter_weights": {
            "filter_differential_pressure": 4.0,
            "flow_rate": 2.5,
            "pump_pressure": 1.5,
            "motor_current": 1.0,
            "vibration": 1.0,
            "particle_count": 2.5,
            "yield_percent": 1.0,
        },
        "minimum_score": 2.0,
    },
    "CHEMICAL_BATH_AGING": {
        "cause_name": "약액 Bath 노화 및 교체주기 초과",
        "parameter_weights": {
            "bath_age_hours": 4.0,
            "particle_count": 1.5,
            "metal_contamination_ppb": 2.0,
            "organic_residue_count": 2.0,
            "yield_percent": 1.0,
        },
        "minimum_score": 2.0,
    },
    "DIW_QUALITY_DEGRADATION": {
        "cause_name": "DI Water 품질 저하",
        "parameter_weights": {
            "diw_resistivity": 4.0,
            "water_mark_count": 3.0,
            "metal_contamination_ppb": 1.0,
            "yield_percent": 1.0,
        },
        "minimum_score": 2.0,
    },
    "NOZZLE_FLOW_INSTABILITY": {
        "cause_name": "Nozzle 분사압 및 유량 불안정",
        "parameter_weights": {
            "nozzle_pressure": 4.0,
            "flow_rate": 3.0,
            "pump_pressure": 1.0,
            "particle_count": 1.5,
            "water_mark_count": 1.0,
            "yield_percent": 1.0,
        },
        "minimum_score": 2.0,
    },
    "CHEMICAL_CONCENTRATION_DEVIATION": {
        "cause_name": "약액 농도 편차",
        "parameter_weights": {
            "concentration_deviation_percent": 4.0,
            "metal_contamination_ppb": 1.5,
            "organic_residue_count": 1.5,
            "particle_count": 1.0,
            "yield_percent": 1.0,
        },
        "minimum_score": 2.0,
    },
    "BATH_TEMPERATURE_DEVIATION": {
        "cause_name": "Bath 온도 조건 이탈",
        "parameter_weights": {
            "temperature_deviation": 4.0,
            "organic_residue_count": 2.0,
            "particle_count": 1.0,
            "yield_percent": 1.0,
        },
        "minimum_score": 2.0,
    },
    "PUMP_MOTOR_DEGRADATION": {
        "cause_name": "Pump 및 Motor 성능 저하",
        "parameter_weights": {
            "pump_pressure": 3.0,
            "motor_current": 3.0,
            "vibration": 3.0,
            "flow_rate": 2.0,
            "nozzle_pressure": 1.0,
            "yield_percent": 0.5,
        },
        "minimum_score": 2.0,
    },
    "EXHAUST_SYSTEM_INSTABILITY": {
        "cause_name": "배기 계통 압력 불안정",
        "parameter_weights": {
            "exhaust_pressure": 4.0,
            "particle_count": 1.5,
            "organic_residue_count": 1.0,
            "yield_percent": 0.5,
        },
        "minimum_score": 2.0,
    },
}


def connect_database() -> sqlite3.Connection:
    """SQLite 데이터베이스에 연결합니다."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"데이터베이스를 찾을 수 없습니다.\n"
            f"경로: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def load_anomaly_data(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    규칙 기반 이상 감지 결과를 조회합니다.

    원인 분석은 NORMAL을 제외하고 anomaly_result에 저장된
    WATCH, WARNING, CRITICAL 항목을 사용합니다.
    """

    query = """
    SELECT
        anomaly_id,
        lot_id,
        equipment_id,
        anomaly_type,
        parameter_name,
        measured_value,
        reference_value,
        severity,
        message
    FROM anomaly_result
    WHERE detection_method = 'RULE_BASED'
    ORDER BY
        lot_id,
        CASE severity
            WHEN 'CRITICAL' THEN 1
            WHEN 'WARNING' THEN 2
            WHEN 'WATCH' THEN 3
            ELSE 4
        END,
        parameter_name;
    """

    return pd.read_sql_query(query, connection)


def calculate_rule_score(
    lot_anomalies: pd.DataFrame,
    rule: dict[str, Any],
) -> tuple[float, list[str], set[str]]:
    """
    한 LOT에서 특정 원인 후보의 점수와 근거를 계산합니다.

    점수 계산:

    원인-항목 연관 가중치 × 이상 심각도 가중치
    """

    parameter_weights: dict[str, float] = rule[
        "parameter_weights"
    ]

    score = 0.0
    evidence_items: list[str] = []
    matched_parameters: set[str] = set()

    for _, anomaly in lot_anomalies.iterrows():
        parameter_name = str(anomaly["parameter_name"])

        if parameter_name not in parameter_weights:
            continue

        severity = str(anomaly["severity"])

        severity_weight = SEVERITY_WEIGHT.get(
            severity,
            0.0,
        )

        relation_weight = parameter_weights[
            parameter_name
        ]

        parameter_score = (
            relation_weight * severity_weight
        )

        score += parameter_score
        matched_parameters.add(parameter_name)

        measured_value = float(
            anomaly["measured_value"]
        )

        evidence_items.append(
            f"{parameter_name}="
            f"{measured_value:.2f}"
            f"({severity})"
        )

    return score, evidence_items, matched_parameters


def apply_combination_bonus(
    cause_code: str,
    matched_parameters: set[str],
) -> tuple[float, list[str]]:
    """
    함께 발생할 때 원인 가능성이 더 높아지는
    이상 조합에 추가 점수를 부여합니다.
    """

    bonus_score = 0.0
    bonus_evidence: list[str] = []

    if cause_code == "FILTER_CLOGGING":
        if {
            "filter_differential_pressure",
            "flow_rate",
        }.issubset(matched_parameters):
            bonus_score += 5.0
            bonus_evidence.append(
                "Filter 차압 상승과 유량 저하가 동시에 발생"
            )

        if {
            "filter_differential_pressure",
            "particle_count",
        }.issubset(matched_parameters):
            bonus_score += 4.0
            bonus_evidence.append(
                "Filter 차압 상승과 Particle 증가가 동시에 발생"
            )

    elif cause_code == "CHEMICAL_BATH_AGING":
        if {
            "bath_age_hours",
            "organic_residue_count",
        }.issubset(matched_parameters):
            bonus_score += 4.0
            bonus_evidence.append(
                "Bath 사용시간 증가와 유기 잔사 증가가 동시에 발생"
            )

        if {
            "bath_age_hours",
            "metal_contamination_ppb",
        }.issubset(matched_parameters):
            bonus_score += 4.0
            bonus_evidence.append(
                "Bath 노화와 금속 오염도 증가가 동시에 발생"
            )

    elif cause_code == "DIW_QUALITY_DEGRADATION":
        if {
            "diw_resistivity",
            "water_mark_count",
        }.issubset(matched_parameters):
            bonus_score += 5.0
            bonus_evidence.append(
                "DIW 비저항 저하와 Water Mark 증가가 동시에 발생"
            )

    elif cause_code == "NOZZLE_FLOW_INSTABILITY":
        if {
            "nozzle_pressure",
            "flow_rate",
        }.issubset(matched_parameters):
            bonus_score += 5.0
            bonus_evidence.append(
                "Nozzle 압력 이상과 유량 저하가 동시에 발생"
            )

    elif cause_code == "PUMP_MOTOR_DEGRADATION":
        if {
            "motor_current",
            "vibration",
        }.issubset(matched_parameters):
            bonus_score += 5.0
            bonus_evidence.append(
                "Motor 전류 상승과 진동 증가가 동시에 발생"
            )

        if {
            "pump_pressure",
            "flow_rate",
        }.issubset(matched_parameters):
            bonus_score += 3.0
            bonus_evidence.append(
                "Pump 압력 이상과 유량 저하가 동시에 발생"
            )

    return bonus_score, bonus_evidence


def determine_confidence_level(
    score: float,
    matched_parameter_count: int,
) -> str:
    """원인 점수와 근거 항목 수를 기준으로 신뢰도를 정합니다."""

    if score >= 18.0 and matched_parameter_count >= 3:
        return "HIGH"

    if score >= 9.0 and matched_parameter_count >= 2:
        return "MEDIUM"

    return "LOW"


def analyze_lot_root_causes(
    lot_id: str,
    lot_anomalies: pd.DataFrame,
    top_count: int = 3,
) -> list[dict[str, Any]]:
    """한 LOT의 원인 후보를 계산하고 상위 원인을 반환합니다."""

    cause_candidates: list[dict[str, Any]] = []

    for cause_code, rule in ROOT_CAUSE_RULES.items():
        (
            base_score,
            evidence_items,
            matched_parameters,
        ) = calculate_rule_score(
            lot_anomalies,
            rule,
        )

        bonus_score, bonus_evidence = (
            apply_combination_bonus(
                cause_code,
                matched_parameters,
            )
        )

        total_score = base_score + bonus_score

        minimum_score = float(
            rule.get("minimum_score", 0.0)
        )

        if total_score < minimum_score:
            continue

        all_evidence = (
            evidence_items + bonus_evidence
        )

        confidence_level = determine_confidence_level(
            score=total_score,
            matched_parameter_count=len(
                matched_parameters
            ),
        )

        cause_candidates.append(
            {
                "lot_id": lot_id,
                "cause_code": cause_code,
                "cause_name": rule["cause_name"],
                "score": total_score,
                "evidence": "; ".join(all_evidence),
                "confidence_level": confidence_level,
            }
        )

    cause_candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected_causes = cause_candidates[:top_count]

    total_selected_score = sum(
        float(cause["score"])
        for cause in selected_causes
    )

    for rank, cause in enumerate(
        selected_causes,
        start=1,
    ):
        if total_selected_score > 0:
            contribution_percent = (
                float(cause["score"])
                / total_selected_score
                * 100
            )
        else:
            contribution_percent = 0.0

        cause["cause_rank"] = rank
        cause["contribution_percent"] = round(
            contribution_percent,
            2,
        )

    return selected_causes


def analyze_all_lots(
    anomaly_dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """이상이 존재하는 모든 LOT에 대해 원인 분석을 실행합니다."""

    all_results: list[dict[str, Any]] = []

    for lot_id, lot_group in anomaly_dataframe.groupby(
        "lot_id",
        sort=False,
    ):
        lot_results = analyze_lot_root_causes(
            lot_id=str(lot_id),
            lot_anomalies=lot_group,
            top_count=3,
        )

        all_results.extend(lot_results)

    return all_results


def clear_existing_results(
    connection: sqlite3.Connection,
) -> None:
    """기존 규칙 기반 원인 분석 결과를 삭제합니다."""

    connection.execute(
        """
        DELETE FROM root_cause_result
        WHERE analysis_method = ?;
        """,
        (ANALYSIS_METHOD,),
    )


def save_root_cause_results(
    connection: sqlite3.Connection,
    results: list[dict[str, Any]],
) -> None:
    """원인 분석 결과를 root_cause_result에 저장합니다."""

    if not results:
        return

    sql = """
    INSERT INTO root_cause_result (
        lot_id,
        cause_rank,
        cause_name,
        evidence,
        contribution_percent,
        confidence_level,
        analysis_method
    )
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """

    rows = [
        (
            result["lot_id"],
            result["cause_rank"],
            result["cause_name"],
            result["evidence"],
            result["contribution_percent"],
            result["confidence_level"],
            ANALYSIS_METHOD,
        )
        for result in results
    ]

    connection.executemany(sql, rows)


def run_root_cause_analysis() -> list[dict[str, Any]]:
    """규칙 기반 원인 분석 전체 과정을 실행합니다."""

    connection = connect_database()

    try:
        anomaly_dataframe = load_anomaly_data(
            connection
        )

        if anomaly_dataframe.empty:
            raise RuntimeError(
                "분석할 이상 감지 결과가 없습니다.\n"
                "먼저 다음 명령을 실행하세요.\n"
                "python src/detection/rule_detector.py"
            )

        clear_existing_results(connection)

        results = analyze_all_lots(
            anomaly_dataframe
        )

        save_root_cause_results(
            connection,
            results,
        )

        connection.commit()

        return results

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def print_summary() -> None:
    """원인 분석 결과를 터미널에 요약 출력합니다."""

    connection = connect_database()

    try:
        cause_summary_query = """
        SELECT
            cause_name,
            COUNT(*) AS occurrence_count,
            ROUND(
                AVG(contribution_percent),
                2
            ) AS avg_contribution
        FROM root_cause_result
        WHERE analysis_method = ?
        GROUP BY cause_name
        ORDER BY occurrence_count DESC,
                 avg_contribution DESC
        LIMIT 10;
        """

        equipment_summary_query = """
        SELECT
            l.equipment_id,
            r.cause_name,
            COUNT(*) AS occurrence_count
        FROM root_cause_result AS r
        INNER JOIN lot_history AS l
            ON r.lot_id = l.lot_id
        WHERE
            r.analysis_method = ?
            AND r.cause_rank = 1
        GROUP BY
            l.equipment_id,
            r.cause_name
        ORDER BY
            l.equipment_id,
            occurrence_count DESC;
        """

        wc03_recent_query = """
        SELECT
            r.lot_id,
            r.cause_rank,
            r.cause_name,
            r.contribution_percent,
            r.confidence_level
        FROM root_cause_result AS r
        INNER JOIN lot_history AS l
            ON r.lot_id = l.lot_id
        WHERE
            r.analysis_method = ?
            AND l.equipment_id = 'WC-03'
        ORDER BY
            l.start_time DESC,
            r.cause_rank
        LIMIT 9;
        """

        cause_rows = connection.execute(
            cause_summary_query,
            (ANALYSIS_METHOD,),
        ).fetchall()

        equipment_rows = connection.execute(
            equipment_summary_query,
            (ANALYSIS_METHOD,),
        ).fetchall()

        wc03_rows = connection.execute(
            wc03_recent_query,
            (ANALYSIS_METHOD,),
        ).fetchall()

        print("\n" + "=" * 84)
        print("전체 주요 원인 요약")
        print("=" * 84)

        for (
            cause_name,
            occurrence_count,
            avg_contribution,
        ) in cause_rows:
            print(
                f"{cause_name:<38} | "
                f"{occurrence_count:3d}건 | "
                f"평균 기여도 "
                f"{avg_contribution:6.2f}%"
            )

        print("\n" + "=" * 84)
        print("설비별 1순위 원인")
        print("=" * 84)

        for (
            equipment_id,
            cause_name,
            occurrence_count,
        ) in equipment_rows:
            print(
                f"{equipment_id:<8} | "
                f"{cause_name:<38} | "
                f"{occurrence_count:3d}건"
            )

        print("\n" + "=" * 84)
        print("WC-03 최근 원인 분석 결과")
        print("=" * 84)

        for (
            lot_id,
            cause_rank,
            cause_name,
            contribution_percent,
            confidence_level,
        ) in wc03_rows:
            print(
                f"{lot_id} | "
                f"순위 {cause_rank} | "
                f"{cause_name:<34} | "
                f"{contribution_percent:6.2f}% | "
                f"{confidence_level}"
            )

    finally:
        connection.close()


def main() -> None:
    try:
        results = run_root_cause_analysis()

        analyzed_lot_count = len(
            {
                result["lot_id"]
                for result in results
            }
        )

        print("=" * 84)
        print("규칙 기반 Root Cause Analysis 완료")
        print("=" * 84)
        print(
            f"분석 LOT 수: {analyzed_lot_count}개"
        )
        print(
            f"저장된 원인 결과: {len(results)}건"
        )

        print_summary()

    except (
        FileNotFoundError,
        RuntimeError,
        sqlite3.Error,
        KeyError,
        ValueError,
    ) as error:
        print("=" * 84)
        print("Root Cause Analysis 실패")
        print(error)
        print("=" * 84)


if __name__ == "__main__":
    main()