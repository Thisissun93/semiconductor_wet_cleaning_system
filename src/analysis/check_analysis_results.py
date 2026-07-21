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


def validate_required_data(
    connection: sqlite3.Connection,
) -> None:
    """분석 결과가 생성되어 있는지 확인합니다."""

    required_tables = {
        "lot_history": (
            "생산 LOT 데이터가 없습니다.\n"
            "python src/data_generation/generate_data.py"
        ),
        "anomaly_result": (
            "이상 감지 결과가 없습니다.\n"
            "python src/detection/rule_detector.py"
        ),
        "root_cause_result": (
            "원인 분석 결과가 없습니다.\n"
            "python src/root_cause/rule_based_rca.py"
        ),
        "action_recommendation": (
            "조치 추천 결과가 없습니다.\n"
            "python src/recommendation/action_engine.py"
        ),
    }

    for table_name, error_message in required_tables.items():
        query = f"SELECT COUNT(*) FROM {table_name};"
        count = connection.execute(query).fetchone()[0]

        if count == 0:
            raise RuntimeError(error_message)


def load_lot_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    LOT별 품질 상태와 분석 결과 건수를 통합 조회합니다.
    """

    query = """
    WITH anomaly_summary AS (
        SELECT
            lot_id,
            COUNT(*) AS anomaly_count,
            SUM(
                CASE
                    WHEN severity = 'CRITICAL' THEN 1
                    ELSE 0
                END
            ) AS critical_count,
            SUM(
                CASE
                    WHEN severity = 'WARNING' THEN 1
                    ELSE 0
                END
            ) AS warning_count,
            SUM(
                CASE
                    WHEN severity = 'WATCH' THEN 1
                    ELSE 0
                END
            ) AS watch_count,
            MAX(
                CASE severity
                    WHEN 'CRITICAL' THEN 3
                    WHEN 'WARNING' THEN 2
                    WHEN 'WATCH' THEN 1
                    ELSE 0
                END
            ) AS maximum_severity_score
        FROM anomaly_result
        WHERE detection_method = 'RULE_BASED'
        GROUP BY lot_id
    ),

    root_cause_summary AS (
        SELECT
            lot_id,
            COUNT(*) AS root_cause_count,
            MAX(
                CASE
                    WHEN cause_rank = 1 THEN cause_name
                    ELSE NULL
                END
            ) AS top_cause,
            MAX(
                CASE
                    WHEN cause_rank = 1
                    THEN contribution_percent
                    ELSE NULL
                END
            ) AS top_cause_contribution,
            MAX(
                CASE
                    WHEN cause_rank = 1
                    THEN confidence_level
                    ELSE NULL
                END
            ) AS top_cause_confidence
        FROM root_cause_result
        WHERE analysis_method = ?
        GROUP BY lot_id
    ),

    action_summary AS (
        SELECT
            lot_id,
            COUNT(*) AS action_count,
            MAX(
                CASE
                    WHEN action_rank = 1 THEN action_name
                    ELSE NULL
                END
            ) AS top_action,
            MAX(
                CASE
                    WHEN action_rank = 1 THEN priority
                    ELSE NULL
                END
            ) AS top_action_priority,
            MAX(
                CASE
                    WHEN action_rank = 1
                    THEN responsible_department
                    ELSE NULL
                END
            ) AS top_action_department
        FROM action_recommendation
        WHERE action_status = 'OPEN'
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
        q.inspection_result,

        COALESCE(a.anomaly_count, 0) AS anomaly_count,
        COALESCE(a.critical_count, 0) AS critical_count,
        COALESCE(a.warning_count, 0) AS warning_count,
        COALESCE(a.watch_count, 0) AS watch_count,

        CASE COALESCE(a.maximum_severity_score, 0)
            WHEN 3 THEN 'CRITICAL'
            WHEN 2 THEN 'WARNING'
            WHEN 1 THEN 'WATCH'
            ELSE 'NORMAL'
        END AS maximum_severity,

        COALESCE(r.root_cause_count, 0)
            AS root_cause_count,
        r.top_cause,
        r.top_cause_contribution,
        r.top_cause_confidence,

        COALESCE(ac.action_count, 0) AS action_count,
        ac.top_action,
        ac.top_action_priority,
        ac.top_action_department

    FROM lot_history AS l

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    LEFT JOIN anomaly_summary AS a
        ON l.lot_id = a.lot_id

    LEFT JOIN root_cause_summary AS r
        ON l.lot_id = r.lot_id

    LEFT JOIN action_summary AS ac
        ON l.lot_id = ac.lot_id

    ORDER BY l.start_time;
    """

    return pd.read_sql_query(
        query,
        connection,
        params=(ANALYSIS_METHOD,),
    )


def load_equipment_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """설비별 품질 및 이상 현황을 조회합니다."""

    query = """
    WITH lot_anomaly AS (
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
            ) AS maximum_severity_score
        FROM anomaly_result
        WHERE detection_method = 'RULE_BASED'
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

        SUM(
            CASE
                WHEN l.lot_status = 'FAIL' THEN 1
                ELSE 0
            END
        ) AS fail_lot_count,

        SUM(
            CASE
                WHEN l.lot_status = 'HOLD' THEN 1
                ELSE 0
            END
        ) AS hold_lot_count,

        SUM(
            CASE
                WHEN a.maximum_severity_score = 3 THEN 1
                ELSE 0
            END
        ) AS critical_lot_count,

        SUM(
            CASE
                WHEN a.maximum_severity_score = 2 THEN 1
                ELSE 0
            END
        ) AS warning_lot_count,

        COALESCE(
            SUM(a.anomaly_count),
            0
        ) AS total_anomaly_count

    FROM lot_history AS l

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    LEFT JOIN lot_anomaly AS a
        ON l.lot_id = a.lot_id

    GROUP BY l.equipment_id
    ORDER BY l.equipment_id;
    """

    return pd.read_sql_query(query, connection)


def validate_analysis_connections(
    lot_summary: pd.DataFrame,
) -> list[str]:
    """
    분석 단계 간 연결 오류를 검사합니다.

    반환값:
        검증 중 발견된 문제 메시지 목록
    """

    issues: list[str] = []

    anomaly_lots = lot_summary[
        lot_summary["anomaly_count"] > 0
    ]

    missing_root_causes = anomaly_lots[
        anomaly_lots["root_cause_count"] == 0
    ]

    if not missing_root_causes.empty:
        issues.append(
            "이상이 있지만 Root Cause 결과가 없는 LOT가 "
            f"{len(missing_root_causes)}개 있습니다."
        )

    root_cause_lots = lot_summary[
        lot_summary["root_cause_count"] > 0
    ]

    missing_actions = root_cause_lots[
        root_cause_lots["action_count"] == 0
    ]

    if not missing_actions.empty:
        issues.append(
            "Root Cause 결과가 있지만 추천 조치가 없는 LOT가 "
            f"{len(missing_actions)}개 있습니다."
        )

    too_many_causes = lot_summary[
        lot_summary["root_cause_count"] > 3
    ]

    if not too_many_causes.empty:
        issues.append(
            "Root Cause가 3개를 초과한 LOT가 "
            f"{len(too_many_causes)}개 있습니다."
        )

    too_many_actions = lot_summary[
        lot_summary["action_count"] > 3
    ]

    if not too_many_actions.empty:
        issues.append(
            "추천 조치가 3개를 초과한 LOT가 "
            f"{len(too_many_actions)}개 있습니다."
        )

    invalid_contributions = lot_summary[
        (
            lot_summary["top_cause_contribution"].notna()
        )
        & (
            (
                lot_summary["top_cause_contribution"] < 0
            )
            | (
                lot_summary["top_cause_contribution"] > 100
            )
        )
    ]

    if not invalid_contributions.empty:
        issues.append(
            "원인 기여도가 0~100% 범위를 벗어난 LOT가 "
            f"{len(invalid_contributions)}개 있습니다."
        )

    return issues


def print_overall_summary(
    lot_summary: pd.DataFrame,
) -> None:
    """전체 분석 결과 건수를 출력합니다."""

    total_lots = len(lot_summary)

    abnormal_lot_count = int(
        (lot_summary["anomaly_count"] > 0).sum()
    )

    root_cause_lot_count = int(
        (lot_summary["root_cause_count"] > 0).sum()
    )

    action_lot_count = int(
        (lot_summary["action_count"] > 0).sum()
    )

    critical_lot_count = int(
        (
            lot_summary["maximum_severity"]
            == "CRITICAL"
        ).sum()
    )

    warning_lot_count = int(
        (
            lot_summary["maximum_severity"]
            == "WARNING"
        ).sum()
    )

    print("=" * 100)
    print("Wet Cleaning 분석 결과 통합 검증")
    print("=" * 100)
    print(f"전체 생산 LOT:          {total_lots:4d}개")
    print(
        f"이상 감지 LOT:          "
        f"{abnormal_lot_count:4d}개"
    )
    print(
        f"Root Cause 분석 LOT:    "
        f"{root_cause_lot_count:4d}개"
    )
    print(
        f"조치 추천 LOT:          "
        f"{action_lot_count:4d}개"
    )
    print(
        f"CRITICAL LOT:           "
        f"{critical_lot_count:4d}개"
    )
    print(
        f"WARNING LOT:            "
        f"{warning_lot_count:4d}개"
    )


def print_equipment_summary(
    equipment_summary: pd.DataFrame,
) -> None:
    """설비별 현황을 출력합니다."""

    print("\n" + "=" * 100)
    print("설비별 품질 및 이상 현황")
    print("=" * 100)

    for _, row in equipment_summary.iterrows():
        print(
            f"{row['equipment_id']} | "
            f"LOT {int(row['lot_count']):3d}개 | "
            f"평균 수율 "
            f"{float(row['average_yield']):6.2f}% | "
            f"평균 Particle "
            f"{float(row['average_particle']):6.2f} | "
            f"FAIL {int(row['fail_lot_count']):2d}개 | "
            f"HOLD {int(row['hold_lot_count']):2d}개 | "
            f"CRITICAL "
            f"{int(row['critical_lot_count']):2d}개 | "
            f"이상 총 "
            f"{int(row['total_anomaly_count']):3d}건"
        )


def print_recent_wc03_results(
    lot_summary: pd.DataFrame,
) -> None:
    """WC-03의 최근 분석 결과를 출력합니다."""

    wc03_recent = (
        lot_summary[
            lot_summary["equipment_id"] == "WC-03"
        ]
        .sort_values(
            by="start_time",
            ascending=False,
        )
        .head(8)
    )

    print("\n" + "=" * 100)
    print("WC-03 최근 LOT 통합 결과")
    print("=" * 100)

    for _, row in wc03_recent.iterrows():
        top_cause = (
            str(row["top_cause"])
            if pd.notna(row["top_cause"])
            else "-"
        )

        top_action = (
            str(row["top_action"])
            if pd.notna(row["top_action"])
            else "-"
        )

        priority = (
            str(row["top_action_priority"])
            if pd.notna(row["top_action_priority"])
            else "-"
        )

        print(
            f"{row['lot_id']} | "
            f"{row['maximum_severity']:<8} | "
            f"수율 {float(row['yield_percent']):6.2f}% | "
            f"Particle {int(row['particle_count']):3d} | "
            f"원인: {top_cause} | "
            f"조치: {top_action} | "
            f"{priority}"
        )


def print_top_causes(
    connection: sqlite3.Connection,
) -> None:
    """전체 Root Cause 발생 순위를 출력합니다."""

    query = """
    SELECT
        cause_name,
        COUNT(*) AS occurrence_count,
        ROUND(
            AVG(contribution_percent),
            2
        ) AS average_contribution,
        SUM(
            CASE
                WHEN cause_rank = 1 THEN 1
                ELSE 0
            END
        ) AS first_rank_count
    FROM root_cause_result
    WHERE analysis_method = ?
    GROUP BY cause_name
    ORDER BY
        first_rank_count DESC,
        occurrence_count DESC
    LIMIT 8;
    """

    rows = connection.execute(
        query,
        (ANALYSIS_METHOD,),
    ).fetchall()

    print("\n" + "=" * 100)
    print("주요 Root Cause TOP 8")
    print("=" * 100)

    for (
        cause_name,
        occurrence_count,
        average_contribution,
        first_rank_count,
    ) in rows:
        print(
            f"{cause_name:<40} | "
            f"발생 {occurrence_count:3d}건 | "
            f"1순위 {first_rank_count:3d}건 | "
            f"평균 기여도 "
            f"{float(average_contribution):6.2f}%"
        )


def print_top_actions(
    connection: sqlite3.Connection,
) -> None:
    """전체 추천 조치 순위를 출력합니다."""

    query = """
    SELECT
        action_name,
        COUNT(*) AS recommendation_count,
        SUM(
            CASE
                WHEN priority = 'URGENT' THEN 1
                ELSE 0
            END
        ) AS urgent_count,
        SUM(
            CASE
                WHEN action_rank = 1 THEN 1
                ELSE 0
            END
        ) AS first_rank_count
    FROM action_recommendation
    WHERE action_status = 'OPEN'
    GROUP BY action_name
    ORDER BY
        first_rank_count DESC,
        recommendation_count DESC
    LIMIT 8;
    """

    rows = connection.execute(query).fetchall()

    print("\n" + "=" * 100)
    print("주요 추천 조치 TOP 8")
    print("=" * 100)

    for (
        action_name,
        recommendation_count,
        urgent_count,
        first_rank_count,
    ) in rows:
        print(
            f"{action_name:<45} | "
            f"추천 {recommendation_count:3d}건 | "
            f"1순위 {first_rank_count:3d}건 | "
            f"URGENT {urgent_count:3d}건"
        )


def print_validation_result(
    issues: list[str],
) -> None:
    """통합 검증 결과를 출력합니다."""

    print("\n" + "=" * 100)
    print("분석 파이프라인 연결 검증")
    print("=" * 100)

    if not issues:
        print(
            "PASS | 이상 감지 → 원인 분석 → 조치 추천 연결이 "
            "정상입니다."
        )
        return

    print(f"WARNING | 총 {len(issues)}개의 문제가 발견됐습니다.")

    for index, issue in enumerate(issues, start=1):
        print(f"{index}. {issue}")


def run_integrated_check() -> None:
    """통합 분석 검증을 실행합니다."""

    connection = connect_database()

    try:
        validate_required_data(connection)

        lot_summary = load_lot_summary(connection)
        equipment_summary = load_equipment_summary(connection)

        issues = validate_analysis_connections(
            lot_summary
        )

        print_overall_summary(lot_summary)
        print_equipment_summary(equipment_summary)
        print_recent_wc03_results(lot_summary)
        print_top_causes(connection)
        print_top_actions(connection)
        print_validation_result(issues)

    finally:
        connection.close()


def main() -> None:
    try:
        run_integrated_check()

    except (
        FileNotFoundError,
        RuntimeError,
        sqlite3.Error,
        KeyError,
        ValueError,
    ) as error:
        print("=" * 100)
        print("통합 분석 결과 확인 실패")
        print("=" * 100)
        print(error)


if __name__ == "__main__":
    main()