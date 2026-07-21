from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)

ANALYSIS_METHOD: Final[str] = "RULE_BASED_RCA"
DETECTION_METHOD: Final[str] = "RULE_BASED"


def connect_database() -> sqlite3.Connection:
    """
    Wet Cleaning SQLite 데이터베이스에 연결합니다.

    Returns:
        sqlite3.Connection: SQLite 연결 객체
    """

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "wet_cleaning.db를 찾을 수 없습니다.\n"
            "먼저 다음 명령을 실행하세요.\n"
            "python run_pipeline.py"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def load_overall_kpis() -> dict[str, float | int | str]:
    """
    Dashboard와 Report 상단에 표시할 전체 KPI를 조회합니다.

    반환 항목:
        전체 LOT
        평균 수율
        평균 Particle
        이상 LOT
        Critical LOT
        Fail LOT
        Open Action
        전체 상태
    """

    query = """
    WITH lot_anomaly AS (
        SELECT
            lot_id,
            MAX(
                CASE severity
                    WHEN 'CRITICAL' THEN 3
                    WHEN 'WARNING' THEN 2
                    WHEN 'WATCH' THEN 1
                    ELSE 0
                END
            ) AS severity_score
        FROM anomaly_result
        WHERE detection_method = ?
        GROUP BY lot_id
    )

    SELECT
        COUNT(DISTINCT l.lot_id) AS total_lots,

        ROUND(
            AVG(q.yield_percent),
            2
        ) AS average_yield,

        ROUND(
            AVG(q.particle_count),
            2
        ) AS average_particle,

        COUNT(
            DISTINCT CASE
                WHEN a.severity_score >= 1
                THEN l.lot_id
            END
        ) AS abnormal_lots,

        COUNT(
            DISTINCT CASE
                WHEN a.severity_score = 3
                THEN l.lot_id
            END
        ) AS critical_lots,

        COUNT(
            DISTINCT CASE
                WHEN l.lot_status = 'FAIL'
                THEN l.lot_id
            END
        ) AS fail_lots,

        (
            SELECT COUNT(*)
            FROM action_recommendation
            WHERE action_status = 'OPEN'
        ) AS open_actions

    FROM lot_history AS l

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    LEFT JOIN lot_anomaly AS a
        ON l.lot_id = a.lot_id;
    """

    connection = connect_database()

    try:
        row = connection.execute(
            query,
            (DETECTION_METHOD,),
        ).fetchone()

    finally:
        connection.close()

    if row is None:
        raise RuntimeError("KPI 데이터를 조회하지 못했습니다.")

    total_lots = int(row[0] or 0)
    average_yield = float(row[1] or 0.0)
    average_particle = float(row[2] or 0.0)
    abnormal_lots = int(row[3] or 0)
    critical_lots = int(row[4] or 0)
    fail_lots = int(row[5] or 0)
    open_actions = int(row[6] or 0)

    if critical_lots > 0:
        overall_status = "CRITICAL"
    elif fail_lots > 0:
        overall_status = "WARNING"
    elif abnormal_lots > 0:
        overall_status = "WATCH"
    else:
        overall_status = "NORMAL"

    return {
        "total_lots": total_lots,
        "average_yield": average_yield,
        "average_particle": average_particle,
        "abnormal_lots": abnormal_lots,
        "critical_lots": critical_lots,
        "fail_lots": fail_lots,
        "open_actions": open_actions,
        "overall_status": overall_status,
    }


def load_equipment_summary() -> pd.DataFrame:
    """설비별 품질 및 이상 현황을 조회합니다."""

    query = """
    WITH sensor_average AS (
        SELECT
            lot_id,
            AVG(
                filter_differential_pressure
            ) AS filter_differential_pressure,
            AVG(flow_rate) AS flow_rate,
            AVG(vibration) AS vibration
        FROM sensor_history
        GROUP BY lot_id
    ),

    lot_anomaly AS (
        SELECT
            lot_id,
            COUNT(*) AS anomaly_count,
            MAX(
                CASE severity
                    WHEN 'CRITICAL' THEN 3
                    WHEN 'WARNING' THEN 2
                    WHEN 'WATCH' THEN 1
                    ELSE 0
                END
            ) AS severity_score
        FROM anomaly_result
        WHERE detection_method = ?
        GROUP BY lot_id
    )

    SELECT
        l.equipment_id,

        COUNT(DISTINCT l.lot_id) AS lot_count,

        ROUND(
            AVG(q.yield_percent),
            2
        ) AS average_yield,

        ROUND(
            AVG(q.particle_count),
            2
        ) AS average_particle,

        ROUND(
            AVG(s.filter_differential_pressure),
            3
        ) AS average_filter_dp,

        ROUND(
            AVG(s.flow_rate),
            2
        ) AS average_flow_rate,

        ROUND(
            AVG(s.vibration),
            3
        ) AS average_vibration,

        SUM(
            CASE
                WHEN l.lot_status = 'FAIL' THEN 1
                ELSE 0
            END
        ) AS fail_lots,

        SUM(
            CASE
                WHEN l.lot_status = 'HOLD' THEN 1
                ELSE 0
            END
        ) AS hold_lots,

        SUM(
            CASE
                WHEN a.severity_score = 3 THEN 1
                ELSE 0
            END
        ) AS critical_lots,

        COALESCE(
            SUM(a.anomaly_count),
            0
        ) AS anomaly_count

    FROM lot_history AS l

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    INNER JOIN sensor_average AS s
        ON l.lot_id = s.lot_id

    LEFT JOIN lot_anomaly AS a
        ON l.lot_id = a.lot_id

    GROUP BY l.equipment_id

    ORDER BY l.equipment_id;
    """

    connection = connect_database()

    try:
        return pd.read_sql_query(
            query,
            connection,
            params=(DETECTION_METHOD,),
        )

    finally:
        connection.close()


def load_lot_trend(
    equipment_id: str | None = None,
) -> pd.DataFrame:
    """
    LOT별 품질 및 주요 센서 추세를 조회합니다.

    Args:
        equipment_id:
            특정 설비만 조회할 경우 설비 ID를 전달합니다.
            None이면 전체 설비를 조회합니다.
    """

    query = """
    WITH sensor_average AS (
        SELECT
            lot_id,
            AVG(flow_rate) AS flow_rate,
            AVG(
                filter_differential_pressure
            ) AS filter_differential_pressure,
            AVG(motor_current) AS motor_current,
            AVG(vibration) AS vibration
        FROM sensor_history
        GROUP BY lot_id
    ),

    lot_anomaly AS (
        SELECT
            lot_id,
            MAX(
                CASE severity
                    WHEN 'CRITICAL' THEN 3
                    WHEN 'WARNING' THEN 2
                    WHEN 'WATCH' THEN 1
                    ELSE 0
                END
            ) AS severity_score
        FROM anomaly_result
        WHERE detection_method = ?
        GROUP BY lot_id
    )

    SELECT
        l.lot_id,
        l.equipment_id,
        l.recipe_id,
        l.start_time,
        l.lot_status,

        q.yield_percent,
        q.particle_count,
        q.metal_contamination_ppb,
        q.water_mark_count,

        p.bath_age_hours,
        p.diw_resistivity,

        s.flow_rate,
        s.filter_differential_pressure,
        s.motor_current,
        s.vibration,

        CASE COALESCE(a.severity_score, 0)
            WHEN 3 THEN 'CRITICAL'
            WHEN 2 THEN 'WARNING'
            WHEN 1 THEN 'WATCH'
            ELSE 'NORMAL'
        END AS severity

    FROM lot_history AS l

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    INNER JOIN process_condition AS p
        ON l.lot_id = p.lot_id

    INNER JOIN sensor_average AS s
        ON l.lot_id = s.lot_id

    LEFT JOIN lot_anomaly AS a
        ON l.lot_id = a.lot_id

    WHERE
        (? IS NULL OR l.equipment_id = ?)

    ORDER BY l.start_time;
    """

    connection = connect_database()

    try:
        dataframe = pd.read_sql_query(
            query,
            connection,
            params=(
                DETECTION_METHOD,
                equipment_id,
                equipment_id,
            ),
        )

    finally:
        connection.close()

    if not dataframe.empty:
        dataframe["start_time"] = pd.to_datetime(
            dataframe["start_time"]
        )

    return dataframe


def load_recent_lot_summary(
    limit: int = 12,
    equipment_id: str | None = None,
) -> pd.DataFrame:
    """
    최근 LOT의 상태, 원인 및 조치 정보를 조회합니다.
    """

    if limit <= 0:
        raise ValueError("limit은 1 이상이어야 합니다.")

    query = """
    WITH anomaly_summary AS (
        SELECT
            lot_id,
            COUNT(*) AS anomaly_count,
            MAX(
                CASE severity
                    WHEN 'CRITICAL' THEN 3
                    WHEN 'WARNING' THEN 2
                    WHEN 'WATCH' THEN 1
                    ELSE 0
                END
            ) AS severity_score
        FROM anomaly_result
        WHERE detection_method = ?
        GROUP BY lot_id
    ),

    top_cause AS (
        SELECT
            lot_id,
            cause_name,
            contribution_percent,
            confidence_level
        FROM root_cause_result
        WHERE
            analysis_method = ?
            AND cause_rank = 1
    ),

    top_action AS (
        SELECT
            lot_id,
            action_name,
            priority,
            responsible_department,
            action_status
        FROM action_recommendation
        WHERE action_rank = 1
    )

    SELECT
        l.lot_id,
        l.equipment_id,
        l.recipe_id,
        l.start_time,
        l.lot_status,

        ROUND(
            q.yield_percent,
            2
        ) AS yield_percent,

        q.particle_count,

        COALESCE(a.anomaly_count, 0)
            AS anomaly_count,

        CASE COALESCE(a.severity_score, 0)
            WHEN 3 THEN 'CRITICAL'
            WHEN 2 THEN 'WARNING'
            WHEN 1 THEN 'WATCH'
            ELSE 'NORMAL'
        END AS severity,

        c.cause_name AS top_cause,
        c.contribution_percent,
        c.confidence_level,

        ac.action_name AS top_action,
        ac.priority,
        ac.responsible_department,
        ac.action_status

    FROM lot_history AS l

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    LEFT JOIN anomaly_summary AS a
        ON l.lot_id = a.lot_id

    LEFT JOIN top_cause AS c
        ON l.lot_id = c.lot_id

    LEFT JOIN top_action AS ac
        ON l.lot_id = ac.lot_id

    WHERE
        (? IS NULL OR l.equipment_id = ?)

    ORDER BY l.start_time DESC

    LIMIT ?;
    """

    connection = connect_database()

    try:
        dataframe = pd.read_sql_query(
            query,
            connection,
            params=(
                DETECTION_METHOD,
                ANALYSIS_METHOD,
                equipment_id,
                equipment_id,
                limit,
            ),
        )

    finally:
        connection.close()

    if not dataframe.empty:
        dataframe["start_time"] = pd.to_datetime(
            dataframe["start_time"]
        )

    return dataframe


def load_top_root_causes(
    limit: int = 5,
    equipment_id: str | None = None,
) -> pd.DataFrame:
    """주요 Root Cause 발생 순위를 조회합니다."""

    if limit <= 0:
        raise ValueError("limit은 1 이상이어야 합니다.")

    query = """
    SELECT
        r.cause_name,

        COUNT(*) AS occurrence_count,

        SUM(
            CASE
                WHEN r.cause_rank = 1 THEN 1
                ELSE 0
            END
        ) AS first_rank_count,

        ROUND(
            AVG(r.contribution_percent),
            2
        ) AS average_contribution,

        SUM(
            CASE
                WHEN r.confidence_level = 'HIGH' THEN 1
                ELSE 0
            END
        ) AS high_confidence_count

    FROM root_cause_result AS r

    INNER JOIN lot_history AS l
        ON r.lot_id = l.lot_id

    WHERE
        r.analysis_method = ?
        AND (? IS NULL OR l.equipment_id = ?)

    GROUP BY r.cause_name

    ORDER BY
        first_rank_count DESC,
        occurrence_count DESC,
        average_contribution DESC

    LIMIT ?;
    """

    connection = connect_database()

    try:
        return pd.read_sql_query(
            query,
            connection,
            params=(
                ANALYSIS_METHOD,
                equipment_id,
                equipment_id,
                limit,
            ),
        )

    finally:
        connection.close()


def load_top_actions(
    limit: int = 5,
    equipment_id: str | None = None,
) -> pd.DataFrame:
    """주요 추천 조치 순위를 조회합니다."""

    if limit <= 0:
        raise ValueError("limit은 1 이상이어야 합니다.")

    query = """
    SELECT
        a.action_name,
        a.target,
        a.responsible_department,

        COUNT(*) AS recommendation_count,

        SUM(
            CASE
                WHEN a.action_rank = 1 THEN 1
                ELSE 0
            END
        ) AS first_rank_count,

        SUM(
            CASE
                WHEN a.priority = 'URGENT' THEN 1
                ELSE 0
            END
        ) AS urgent_count,

        SUM(
            CASE
                WHEN a.priority = 'HIGH' THEN 1
                ELSE 0
            END
        ) AS high_count

    FROM action_recommendation AS a

    INNER JOIN lot_history AS l
        ON a.lot_id = l.lot_id

    WHERE
        a.action_status = 'OPEN'
        AND (? IS NULL OR l.equipment_id = ?)

    GROUP BY
        a.action_name,
        a.target,
        a.responsible_department

    ORDER BY
        first_rank_count DESC,
        urgent_count DESC,
        recommendation_count DESC

    LIMIT ?;
    """

    connection = connect_database()

    try:
        return pd.read_sql_query(
            query,
            connection,
            params=(
                equipment_id,
                equipment_id,
                limit,
            ),
        )

    finally:
        connection.close()


def load_anomaly_distribution(
    equipment_id: str | None = None,
) -> pd.DataFrame:
    """심각도별 이상 발생 건수를 조회합니다."""

    query = """
    SELECT
        a.severity,
        COUNT(*) AS anomaly_count
    FROM anomaly_result AS a

    WHERE
        a.detection_method = ?
        AND (? IS NULL OR a.equipment_id = ?)

    GROUP BY a.severity

    ORDER BY
        CASE a.severity
            WHEN 'CRITICAL' THEN 1
            WHEN 'WARNING' THEN 2
            WHEN 'WATCH' THEN 3
            ELSE 4
        END;
    """

    connection = connect_database()

    try:
        return pd.read_sql_query(
            query,
            connection,
            params=(
                DETECTION_METHOD,
                equipment_id,
                equipment_id,
            ),
        )

    finally:
        connection.close()


def load_equipment_ids() -> list[str]:
    """Dashboard 필터에서 사용할 설비 ID를 조회합니다."""

    query = """
    SELECT equipment_id
    FROM equipment_master
    ORDER BY equipment_id;
    """

    connection = connect_database()

    try:
        rows = connection.execute(query).fetchall()

    finally:
        connection.close()

    return [str(row[0]) for row in rows]


def validate_repository_data() -> dict[str, int]:
    """
    공통 조회 모듈이 정상적으로 데이터를 반환하는지 확인합니다.
    """

    overall_kpis = load_overall_kpis()
    equipment_summary = load_equipment_summary()
    lot_trend = load_lot_trend()
    recent_lots = load_recent_lot_summary()
    top_causes = load_top_root_causes()
    top_actions = load_top_actions()
    anomaly_distribution = load_anomaly_distribution()
    equipment_ids = load_equipment_ids()

    return {
        "total_lots": int(overall_kpis["total_lots"]),
        "equipment_rows": len(equipment_summary),
        "trend_rows": len(lot_trend),
        "recent_lot_rows": len(recent_lots),
        "root_cause_rows": len(top_causes),
        "action_rows": len(top_actions),
        "severity_rows": len(anomaly_distribution),
        "equipment_count": len(equipment_ids),
    }


def main() -> None:
    """공통 데이터 조회 모듈의 동작을 확인합니다."""

    try:
        validation = validate_repository_data()
        kpis = load_overall_kpis()

        print("=" * 80)
        print("Dashboard / Report 공통 데이터 조회 확인")
        print("=" * 80)

        print(
            f"전체 상태:       "
            f"{kpis['overall_status']}"
        )
        print(
            f"전체 LOT:        "
            f"{validation['total_lots']}개"
        )
        print(
            f"설비 수:         "
            f"{validation['equipment_count']}대"
        )
        print(
            f"설비 요약 행:    "
            f"{validation['equipment_rows']}개"
        )
        print(
            f"LOT 추세 행:     "
            f"{validation['trend_rows']}개"
        )
        print(
            f"최근 LOT 행:     "
            f"{validation['recent_lot_rows']}개"
        )
        print(
            f"Root Cause 행:   "
            f"{validation['root_cause_rows']}개"
        )
        print(
            f"추천 조치 행:    "
            f"{validation['action_rows']}개"
        )
        print(
            f"이상 등급 행:    "
            f"{validation['severity_rows']}개"
        )

        print("=" * 80)
        print("공통 데이터 조회 정상")
        print("=" * 80)

    except (
        FileNotFoundError,
        RuntimeError,
        sqlite3.Error,
        ValueError,
        KeyError,
    ) as error:
        print("=" * 80)
        print("공통 데이터 조회 실패")
        print("=" * 80)
        print(error)


if __name__ == "__main__":
    main()