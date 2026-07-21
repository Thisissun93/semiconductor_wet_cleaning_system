from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)

ANALYSIS_METHOD: Final[str] = "RULE_BASED_RCA"
DEFAULT_ACTION_STATUS: Final[str] = "OPEN"


ACTION_RULES: Final[dict[str, list[dict[str, Any]]]] = {
    "Filter 막힘 및 교체주기 초과": [
        {
            "action_name": "Wet Cleaning Filter 즉시 교체",
            "target": "Filter",
            "base_priority": 100,
            "expected_effect": (
                "Filter 차압 정상화, 세정 유량 회복 및 "
                "Particle 발생 감소"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Filter Housing 및 배관 오염 점검",
            "target": "Filter Housing / Chemical Line",
            "base_priority": 78,
            "expected_effect": (
                "잔류 Particle 및 배관 내부 오염원 제거"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Filter 교체주기 기준 재설정",
            "target": "Preventive Maintenance Standard",
            "base_priority": 62,
            "expected_effect": (
                "차압 상승 전 예방 교체로 반복 이상 방지"
            ),
            "responsible_department": "설비기술",
        },
    ],
    "약액 Bath 노화 및 교체주기 초과": [
        {
            "action_name": "Chemical Bath 즉시 교체",
            "target": "Chemical Bath",
            "base_priority": 95,
            "expected_effect": (
                "약액 성능 회복, 잔사 및 오염 발생 감소"
            ),
            "responsible_department": "공정기술",
        },
        {
            "action_name": "Bath Life 및 누적 LOT 기준 확인",
            "target": "Bath Life Management",
            "base_priority": 72,
            "expected_effect": (
                "약액 과사용 방지 및 교체 기준 신뢰성 향상"
            ),
            "responsible_department": "공정기술",
        },
        {
            "action_name": "약액 농도와 금속 오염도 재측정",
            "target": "Chemical Concentration / Contamination",
            "base_priority": 68,
            "expected_effect": (
                "약액 열화 여부 확인 및 품질 영향 검증"
            ),
            "responsible_department": "품질기술",
        },
    ],
    "DI Water 품질 저하": [
        {
            "action_name": "DI Water 비저항 및 공급 상태 점검",
            "target": "DI Water Supply",
            "base_priority": 92,
            "expected_effect": (
                "DIW 품질 회복 및 Water Mark 발생 감소"
            ),
            "responsible_department": "Utility",
        },
        {
            "action_name": "Rinse Line 및 Final Rinse 상태 점검",
            "target": "Rinse Line",
            "base_priority": 75,
            "expected_effect": (
                "잔류 약액 제거 성능 개선 및 건조 불량 예방"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "DIW 품질 샘플링 검사 실시",
            "target": "DI Water Quality",
            "base_priority": 66,
            "expected_effect": (
                "비저항 저하 및 오염 발생 지점 확인"
            ),
            "responsible_department": "품질기술",
        },
    ],
    "Nozzle 분사압 및 유량 불안정": [
        {
            "action_name": "Nozzle 막힘 및 분사 상태 점검",
            "target": "Nozzle",
            "base_priority": 90,
            "expected_effect": (
                "분사 균일도 및 세정 유량 회복"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Nozzle 세정 또는 교체",
            "target": "Nozzle",
            "base_priority": 83,
            "expected_effect": (
                "국부 세정 불량과 Particle 잔류 감소"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Flow Sensor 교정 상태 확인",
            "target": "Flow Sensor",
            "base_priority": 64,
            "expected_effect": (
                "유량 측정 신뢰성 확보 및 이상 오판 방지"
            ),
            "responsible_department": "계측기술",
        },
    ],
    "약액 농도 편차": [
        {
            "action_name": "약액 농도 재측정 및 보정",
            "target": "Chemical Concentration",
            "base_priority": 88,
            "expected_effect": (
                "세정 반응 안정화 및 잔사 발생 감소"
            ),
            "responsible_department": "공정기술",
        },
        {
            "action_name": "Chemical Mixing System 점검",
            "target": "Chemical Mixing System",
            "base_priority": 76,
            "expected_effect": (
                "약액 혼합 편차 및 농도 변동 재발 방지"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "농도 Sensor 교정",
            "target": "Concentration Sensor",
            "base_priority": 60,
            "expected_effect": (
                "농도 측정 정확도 향상"
            ),
            "responsible_department": "계측기술",
        },
    ],
    "Bath 온도 조건 이탈": [
        {
            "action_name": "Bath Heater 및 온도 제어 상태 점검",
            "target": "Bath Heater / Temperature Controller",
            "base_priority": 88,
            "expected_effect": (
                "Bath 온도 정상화 및 세정 반응 안정화"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Temperature Sensor 교정",
            "target": "Temperature Sensor",
            "base_priority": 70,
            "expected_effect": (
                "실제 온도와 측정값 편차 제거"
            ),
            "responsible_department": "계측기술",
        },
        {
            "action_name": "Recipe 온도 조건 재확인",
            "target": "Cleaning Recipe",
            "base_priority": 58,
            "expected_effect": (
                "제품별 적정 세정 온도 조건 유지"
            ),
            "responsible_department": "공정기술",
        },
    ],
    "Pump 및 Motor 성능 저하": [
        {
            "action_name": "Pump 및 Motor 상태 즉시 점검",
            "target": "Pump / Motor",
            "base_priority": 94,
            "expected_effect": (
                "압력과 유량 안정화 및 설비 정지 예방"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Bearing 및 회전체 진동 점검",
            "target": "Bearing / Rotating Parts",
            "base_priority": 82,
            "expected_effect": (
                "진동 원인 제거 및 Motor 열화 방지"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "Motor 전류와 Pump 토출압 추세 확인",
            "target": "Motor Current / Pump Pressure",
            "base_priority": 65,
            "expected_effect": (
                "설비 열화 속도와 교체 시점 판단"
            ),
            "responsible_department": "설비기술",
        },
    ],
    "배기 계통 압력 불안정": [
        {
            "action_name": "Exhaust 압력 및 배기 Line 점검",
            "target": "Exhaust System",
            "base_priority": 86,
            "expected_effect": (
                "Chamber 내부 배기 안정화 및 오염 재유입 방지"
            ),
            "responsible_department": "Utility",
        },
        {
            "action_name": "Damper 및 배기 Filter 상태 확인",
            "target": "Exhaust Damper / Filter",
            "base_priority": 72,
            "expected_effect": (
                "배기 유량 정상화 및 압력 변동 감소"
            ),
            "responsible_department": "설비기술",
        },
        {
            "action_name": "배기 압력 Sensor 교정",
            "target": "Exhaust Pressure Sensor",
            "base_priority": 58,
            "expected_effect": (
                "배기 상태 측정 신뢰성 확보"
            ),
            "responsible_department": "계측기술",
        },
    ],
}


CONFIDENCE_WEIGHT: Final[dict[str, float]] = {
    "HIGH": 1.20,
    "MEDIUM": 1.00,
    "LOW": 0.80,
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


def load_root_cause_results(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """규칙 기반 원인 분석 결과를 조회합니다."""

    query = """
    SELECT
        r.cause_id,
        r.lot_id,
        r.cause_rank,
        r.cause_name,
        r.evidence,
        r.contribution_percent,
        r.confidence_level,
        l.equipment_id,
        l.start_time,
        l.lot_status
    FROM root_cause_result AS r
    INNER JOIN lot_history AS l
        ON r.lot_id = l.lot_id
    WHERE r.analysis_method = ?
    ORDER BY
        l.start_time,
        r.cause_rank;
    """

    return pd.read_sql_query(
        query,
        connection,
        params=(ANALYSIS_METHOD,),
    )


def get_lot_severity(
    connection: sqlite3.Connection,
) -> dict[str, str]:
    """LOT별 가장 높은 이상 심각도를 조회합니다."""

    query = """
    SELECT
        lot_id,
        CASE MAX(
            CASE severity
                WHEN 'CRITICAL' THEN 3
                WHEN 'WARNING' THEN 2
                WHEN 'WATCH' THEN 1
                ELSE 0
            END
        )
            WHEN 3 THEN 'CRITICAL'
            WHEN 2 THEN 'WARNING'
            WHEN 1 THEN 'WATCH'
            ELSE 'NORMAL'
        END AS maximum_severity
    FROM anomaly_result
    WHERE detection_method = 'RULE_BASED'
    GROUP BY lot_id;
    """

    rows = connection.execute(query).fetchall()

    return {
        str(lot_id): str(severity)
        for lot_id, severity in rows
    }


def calculate_action_score(
    base_priority: float,
    contribution_percent: float,
    confidence_level: str,
    cause_rank: int,
    lot_severity: str,
) -> float:
    """
    조치 우선순위 점수를 계산합니다.

    반영 항목:
    - 조치 자체 중요도
    - 원인 기여도
    - 원인 분석 신뢰도
    - 원인 순위
    - LOT의 최고 이상 등급
    """

    confidence_weight = CONFIDENCE_WEIGHT.get(
        confidence_level,
        0.80,
    )

    rank_weight = {
        1: 1.00,
        2: 0.85,
        3: 0.70,
    }.get(cause_rank, 0.60)

    severity_weight = {
        "CRITICAL": 1.25,
        "WARNING": 1.10,
        "WATCH": 1.00,
        "NORMAL": 0.90,
    }.get(lot_severity, 1.00)

    contribution_weight = max(
        contribution_percent / 100.0,
        0.10,
    )

    score = (
        base_priority
        * confidence_weight
        * rank_weight
        * severity_weight
        * (0.70 + contribution_weight)
    )

    return round(score, 2)


def determine_priority(
    action_score: float,
    lot_severity: str,
) -> str:
    """계산된 점수를 사람이 이해하기 쉬운 우선순위로 변환합니다."""

    if lot_severity == "CRITICAL" and action_score >= 85:
        return "URGENT"

    if action_score >= 95:
        return "URGENT"

    if action_score >= 70:
        return "HIGH"

    if action_score >= 45:
        return "MEDIUM"

    return "LOW"


def create_action_candidates(
    root_cause_dataframe: pd.DataFrame,
    lot_severity_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """모든 LOT의 원인 결과를 조치 후보로 변환합니다."""

    results: list[dict[str, Any]] = []

    for _, cause in root_cause_dataframe.iterrows():
        lot_id = str(cause["lot_id"])
        cause_name = str(cause["cause_name"])
        cause_rank = int(cause["cause_rank"])

        contribution_percent = float(
            cause["contribution_percent"]
        )

        confidence_level = str(
            cause["confidence_level"]
        )

        lot_severity = lot_severity_mapping.get(
            lot_id,
            "WATCH",
        )

        action_templates = ACTION_RULES.get(
            cause_name,
            [],
        )

        for template_index, template in enumerate(
            action_templates,
            start=1,
        ):
            action_score = calculate_action_score(
                base_priority=float(
                    template["base_priority"]
                ),
                contribution_percent=contribution_percent,
                confidence_level=confidence_level,
                cause_rank=cause_rank,
                lot_severity=lot_severity,
            )

            results.append(
                {
                    "lot_id": lot_id,
                    "cause_name": cause_name,
                    "cause_rank": cause_rank,
                    "template_rank": template_index,
                    "action_name": template["action_name"],
                    "target": template["target"],
                    "priority": determine_priority(
                        action_score,
                        lot_severity,
                    ),
                    "expected_effect": (
                        template["expected_effect"]
                    ),
                    "responsible_department": (
                        template[
                            "responsible_department"
                        ]
                    ),
                    "action_score": action_score,
                    "action_status": (
                        DEFAULT_ACTION_STATUS
                    ),
                    "created_at": datetime.now().isoformat(
                        sep=" ",
                        timespec="seconds",
                    ),
                }
            )

    return results


def remove_duplicate_actions(
    action_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    한 LOT에서 동일한 조치가 여러 원인으로 추천되는 경우
    가장 높은 점수만 남깁니다.
    """

    unique_actions: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for action in action_candidates:
        key = (
            str(action["lot_id"]),
            str(action["action_name"]),
        )

        existing_action = unique_actions.get(key)

        if (
            existing_action is None
            or float(action["action_score"])
            > float(existing_action["action_score"])
        ):
            unique_actions[key] = action

    return list(unique_actions.values())


def select_top_actions(
    action_candidates: list[dict[str, Any]],
    top_count: int = 3,
) -> list[dict[str, Any]]:
    """LOT별 점수가 높은 조치 TOP N을 선택합니다."""

    actions_by_lot: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for action in action_candidates:
        lot_id = str(action["lot_id"])

        actions_by_lot.setdefault(
            lot_id,
            [],
        ).append(action)

    selected_actions: list[dict[str, Any]] = []

    for lot_id, lot_actions in actions_by_lot.items():
        sorted_actions = sorted(
            lot_actions,
            key=lambda item: (
                float(item["action_score"]),
                -int(item["cause_rank"]),
                -int(item["template_rank"]),
            ),
            reverse=True,
        )

        for action_rank, action in enumerate(
            sorted_actions[:top_count],
            start=1,
        ):
            action["action_rank"] = action_rank
            selected_actions.append(action)

    return selected_actions


def clear_existing_actions(
    connection: sqlite3.Connection,
) -> None:
    """
    기존 OPEN 상태의 자동 추천 결과를 삭제합니다.

    사용자가 완료 처리한 CLOSED 데이터는 보존할 수 있도록
    OPEN 상태만 삭제합니다.
    """

    connection.execute(
        """
        DELETE FROM action_recommendation
        WHERE action_status = 'OPEN';
        """
    )


def save_action_results(
    connection: sqlite3.Connection,
    results: list[dict[str, Any]],
) -> None:
    """추천 조치를 action_recommendation 테이블에 저장합니다."""

    if not results:
        return

    sql = """
    INSERT INTO action_recommendation (
        lot_id,
        action_rank,
        action_name,
        target,
        priority,
        expected_effect,
        responsible_department,
        action_status,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    rows = [
        (
            result["lot_id"],
            result["action_rank"],
            result["action_name"],
            result["target"],
            result["priority"],
            result["expected_effect"],
            result["responsible_department"],
            result["action_status"],
            result["created_at"],
        )
        for result in results
    ]

    connection.executemany(sql, rows)


def run_action_engine() -> list[dict[str, Any]]:
    """조치 추천 전체 과정을 실행합니다."""

    connection = connect_database()

    try:
        root_cause_dataframe = load_root_cause_results(
            connection
        )

        if root_cause_dataframe.empty:
            raise RuntimeError(
                "조치 추천에 사용할 Root Cause 결과가 없습니다.\n"
                "먼저 다음 명령을 실행하세요.\n"
                "python src/root_cause/rule_based_rca.py"
            )

        lot_severity_mapping = get_lot_severity(
            connection
        )

        action_candidates = create_action_candidates(
            root_cause_dataframe,
            lot_severity_mapping,
        )

        action_candidates = remove_duplicate_actions(
            action_candidates
        )

        selected_actions = select_top_actions(
            action_candidates,
            top_count=3,
        )

        clear_existing_actions(connection)

        save_action_results(
            connection,
            selected_actions,
        )

        connection.commit()

        return selected_actions

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def print_action_summary() -> None:
    """추천 조치 결과를 터미널에 요약 출력합니다."""

    connection = connect_database()

    try:
        priority_query = """
        SELECT
            priority,
            COUNT(*) AS action_count
        FROM action_recommendation
        WHERE action_status = 'OPEN'
        GROUP BY priority
        ORDER BY
            CASE priority
                WHEN 'URGENT' THEN 1
                WHEN 'HIGH' THEN 2
                WHEN 'MEDIUM' THEN 3
                WHEN 'LOW' THEN 4
                ELSE 5
            END;
        """

        top_action_query = """
        SELECT
            action_name,
            COUNT(*) AS recommendation_count
        FROM action_recommendation
        WHERE action_status = 'OPEN'
        GROUP BY action_name
        ORDER BY recommendation_count DESC
        LIMIT 10;
        """

        wc03_query = """
        SELECT
            a.lot_id,
            a.action_rank,
            a.priority,
            a.action_name,
            a.target,
            a.responsible_department
        FROM action_recommendation AS a
        INNER JOIN lot_history AS l
            ON a.lot_id = l.lot_id
        WHERE
            a.action_status = 'OPEN'
            AND l.equipment_id = 'WC-03'
        ORDER BY
            l.start_time DESC,
            a.action_rank
        LIMIT 9;
        """

        priority_rows = connection.execute(
            priority_query
        ).fetchall()

        top_action_rows = connection.execute(
            top_action_query
        ).fetchall()

        wc03_rows = connection.execute(
            wc03_query
        ).fetchall()

        print("\n" + "=" * 100)
        print("우선순위별 조치 요약")
        print("=" * 100)

        for priority, count in priority_rows:
            print(
                f"{priority:<10} | "
                f"{count:4d}건"
            )

        print("\n" + "=" * 100)
        print("주요 추천 조치 TOP 10")
        print("=" * 100)

        for action_name, count in top_action_rows:
            print(
                f"{action_name:<48} | "
                f"{count:4d}건"
            )

        print("\n" + "=" * 100)
        print("WC-03 최근 조치 추천")
        print("=" * 100)

        for (
            lot_id,
            action_rank,
            priority,
            action_name,
            target,
            department,
        ) in wc03_rows:
            print(
                f"{lot_id} | "
                f"순위 {action_rank} | "
                f"{priority:<7} | "
                f"{action_name:<38} | "
                f"{department}"
            )

    finally:
        connection.close()


def main() -> None:
    try:
        results = run_action_engine()

        analyzed_lot_count = len(
            {
                result["lot_id"]
                for result in results
            }
        )

        print("=" * 100)
        print("Action Recommendation 생성 완료")
        print("=" * 100)
        print(
            f"조치 추천 LOT 수: {analyzed_lot_count}개"
        )
        print(
            f"저장된 조치 추천: {len(results)}건"
        )

        print_action_summary()

    except (
        FileNotFoundError,
        RuntimeError,
        sqlite3.Error,
        KeyError,
        ValueError,
    ) as error:
        print("=" * 100)
        print("Action Recommendation 생성 실패")
        print(error)
        print("=" * 100)


if __name__ == "__main__":
    main()