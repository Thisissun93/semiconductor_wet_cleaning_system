from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)

LIMITS_PATH: Final[Path] = (
    PROJECT_ROOT / "config" / "process_limits.yaml"
)


SEVERITY_SCORE: Final[dict[str, int]] = {
    "NORMAL": 0,
    "WATCH": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}


def connect_database() -> sqlite3.Connection:
    """SQLite 데이터베이스에 연결합니다."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"데이터베이스가 존재하지 않습니다: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def load_limits() -> dict[str, Any]:
    """process_limits.yaml 설정을 읽어 반환합니다."""

    if not LIMITS_PATH.exists():
        raise FileNotFoundError(
            f"공정 기준 설정 파일이 없습니다: {LIMITS_PATH}"
        )

    with LIMITS_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "process_limits.yaml의 형식이 올바르지 않습니다."
        )

    return config


def load_lot_data(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    LOT별 공정·설비·품질 데이터를 하나로 조회합니다.

    센서 데이터는 한 LOT에 여러 건이므로
    LOT별 평균값으로 집계합니다.
    """

    query = """
    WITH sensor_average AS (
        SELECT
            lot_id,
            equipment_id,
            AVG(pump_pressure) AS pump_pressure,
            AVG(flow_rate) AS flow_rate,
            AVG(
                filter_differential_pressure
            ) AS filter_differential_pressure,
            AVG(nozzle_pressure) AS nozzle_pressure,
            AVG(motor_current) AS motor_current,
            AVG(vibration) AS vibration,
            AVG(exhaust_pressure) AS exhaust_pressure
        FROM sensor_history
        GROUP BY
            lot_id,
            equipment_id
    )

    SELECT
        l.lot_id,
        l.equipment_id,
        l.recipe_id,
        l.start_time,
        l.lot_status,

        p.chemical_concentration,
        p.bath_temperature,
        p.cleaning_time,
        p.rinse_time,
        p.drying_time,
        p.megasonic_power,
        p.spin_speed,
        p.diw_resistivity,
        p.bath_age_hours,

        r.target_concentration,
        r.target_bath_temperature,

        s.pump_pressure,
        s.flow_rate,
        s.filter_differential_pressure,
        s.nozzle_pressure,
        s.motor_current,
        s.vibration,
        s.exhaust_pressure,

        q.particle_count,
        q.metal_contamination_ppb,
        q.water_mark_count,
        q.organic_residue_count,
        q.pattern_collapse_count,
        q.yield_percent,
        q.inspection_result

    FROM lot_history AS l

    INNER JOIN process_condition AS p
        ON l.lot_id = p.lot_id

    INNER JOIN recipe_master AS r
        ON l.recipe_id = r.recipe_id

    INNER JOIN sensor_average AS s
        ON l.lot_id = s.lot_id

    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id

    ORDER BY l.start_time;
    """

    return pd.read_sql_query(query, connection)


def classify_lower_is_better(
    value: float,
    limits: dict[str, Any],
) -> str:
    """
    값이 낮을수록 좋은 항목을 판정합니다.

    예:
    Particle, Filter 차압, Bath Age
    """

    critical_min = limits.get("critical_min")
    warning_max = limits.get("warning_max")
    watch_max = limits.get("watch_max")
    normal_max = limits.get("normal_max")

    if critical_min is not None and value >= critical_min:
        return "CRITICAL"

    if warning_max is not None and value <= warning_max:
        if watch_max is not None and value <= watch_max:
            if normal_max is not None and value <= normal_max:
                return "NORMAL"

            return "WATCH"

        return "WARNING"

    if warning_max is not None and value > warning_max:
        return "CRITICAL"

    if normal_max is not None and value <= normal_max:
        return "NORMAL"

    if watch_max is not None and value <= watch_max:
        return "WATCH"

    return "WARNING"


def classify_higher_is_better(
    value: float,
    limits: dict[str, Any],
) -> str:
    """
    값이 높을수록 좋은 항목을 판정합니다.

    예:
    수율, 유량, DI Water 비저항
    """

    normal_min = limits.get("normal_min")
    watch_min = limits.get("watch_min")
    warning_min = limits.get("warning_min")
    critical_min = limits.get("critical_min")
    critical_max = limits.get("critical_max")

    critical_boundary = (
        critical_max
        if critical_max is not None
        else critical_min
    )

    if (
        critical_boundary is not None
        and value < critical_boundary
    ):
        return "CRITICAL"

    if warning_min is not None and value < warning_min:
        return "WARNING"

    if watch_min is not None and value < watch_min:
        return "WARNING"

    if normal_min is not None and value < normal_min:
        return "WATCH"

    return "NORMAL"


def classify_range(
    value: float,
    limits: dict[str, Any],
) -> str:
    """
    정상 범위의 상·하한이 모두 있는 항목을 판정합니다.

    예:
    Pump Pressure, Nozzle Pressure
    """

    normal_min = limits.get("normal_min")
    normal_max = limits.get("normal_max")

    warning_min = limits.get("warning_min")
    warning_max = limits.get("warning_max")

    critical_min = limits.get("critical_min")
    critical_max = limits.get("critical_max")

    if normal_min is None or normal_max is None:
        raise ValueError(
            "범위 판정에는 normal_min과 normal_max가 필요합니다."
        )

    if normal_min <= value <= normal_max:
        return "NORMAL"

    if warning_min is not None and warning_max is not None:
        if warning_min <= value <= warning_max:
            return "WARNING"

    if critical_min is not None and value < critical_min:
        return "CRITICAL"

    if critical_max is not None and value > critical_max:
        return "CRITICAL"

    return "WATCH"


def classify_value(
    value: float,
    limits: dict[str, Any],
) -> str:
    """설정된 방향에 따라 적절한 판정 함수를 실행합니다."""

    direction = limits.get("direction")

    if direction == "lower_is_better":
        return classify_lower_is_better(value, limits)

    if direction == "higher_is_better":
        return classify_higher_is_better(value, limits)

    return classify_range(value, limits)


def create_message(
    display_name: str,
    value: float,
    unit: str,
    severity: str,
) -> str:
    """Report와 Dashboard에 표시할 문장을 생성합니다."""

    if severity == "WATCH":
        return (
            f"{display_name}가 정상 범위를 벗어나 "
            f"관찰이 필요합니다. "
            f"현재값: {value:.2f} {unit}"
        )

    if severity == "WARNING":
        return (
            f"{display_name} 이상이 확인되었습니다. "
            f"원인 점검이 필요합니다. "
            f"현재값: {value:.2f} {unit}"
        )

    if severity == "CRITICAL":
        return (
            f"{display_name}가 위험 기준을 초과했습니다. "
            f"즉시 조치가 필요합니다. "
            f"현재값: {value:.2f} {unit}"
        )

    return (
        f"{display_name}는 정상 범위입니다. "
        f"현재값: {value:.2f} {unit}"
    )


def create_reference_value(
    limits: dict[str, Any],
) -> float | None:
    """이상 감지 결과에 저장할 대표 기준값을 반환합니다."""

    direction = limits.get("direction")

    if direction == "lower_is_better":
        value = limits.get("normal_max")

    elif direction == "higher_is_better":
        value = limits.get("normal_min")

    else:
        normal_min = limits.get("normal_min")
        normal_max = limits.get("normal_max")

        if normal_min is not None and normal_max is not None:
            value = (normal_min + normal_max) / 2
        else:
            value = None

    if value is None:
        return None

    return float(value)


def calculate_derived_values(
    row: pd.Series,
) -> dict[str, float]:
    """Recipe 목표값과 실제값의 편차를 계산합니다."""

    target_concentration = float(
        row["target_concentration"]
    )

    actual_concentration = float(
        row["chemical_concentration"]
    )

    if target_concentration == 0:
        concentration_deviation = 0.0
    else:
        concentration_deviation = abs(
            actual_concentration - target_concentration
        ) / target_concentration * 100

    temperature_deviation = abs(
        float(row["bath_temperature"])
        - float(row["target_bath_temperature"])
    )

    return {
        "concentration_deviation_percent": (
            concentration_deviation
        ),
        "temperature_deviation": temperature_deviation,
    }


def evaluate_parameter(
    lot_id: str,
    equipment_id: str,
    parameter_name: str,
    value: float,
    limits: dict[str, Any],
    anomaly_type: str,
) -> dict[str, Any] | None:
    """한 개 항목을 평가하고 이상 결과를 생성합니다."""

    severity = classify_value(value, limits)

    if severity == "NORMAL":
        return None

    display_name = limits.get(
        "display_name",
        parameter_name,
    )

    unit = limits.get("unit", "")

    return {
        "lot_id": lot_id,
        "equipment_id": equipment_id,
        "detected_at": datetime.now().isoformat(
            sep=" ",
            timespec="seconds",
        ),
        "anomaly_type": anomaly_type,
        "parameter_name": parameter_name,
        "measured_value": float(value),
        "reference_value": create_reference_value(limits),
        "severity": severity,
        "detection_method": "RULE_BASED",
        "message": create_message(
            display_name,
            value,
            unit,
            severity,
        ),
    }


def detect_row_anomalies(
    row: pd.Series,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """하나의 LOT에서 모든 이상 항목을 탐지합니다."""

    results: list[dict[str, Any]] = []

    lot_id = str(row["lot_id"])
    equipment_id = str(row["equipment_id"])

    derived_values = calculate_derived_values(row)

    evaluation_groups = [
        (
            "QUALITY",
            config["quality"],
            {
                "yield_percent": float(
                    row["yield_percent"]
                ),
                "particle_count": float(
                    row["particle_count"]
                ),
                "metal_contamination_ppb": float(
                    row["metal_contamination_ppb"]
                ),
                "water_mark_count": float(
                    row["water_mark_count"]
                ),
                "organic_residue_count": float(
                    row["organic_residue_count"]
                ),
                "pattern_collapse_count": float(
                    row["pattern_collapse_count"]
                ),
            },
        ),
        (
            "EQUIPMENT",
            config["equipment"],
            {
                "pump_pressure": float(
                    row["pump_pressure"]
                ),
                "flow_rate": float(
                    row["flow_rate"]
                ),
                "filter_differential_pressure": float(
                    row[
                        "filter_differential_pressure"
                    ]
                ),
                "nozzle_pressure": float(
                    row["nozzle_pressure"]
                ),
                "motor_current": float(
                    row["motor_current"]
                ),
                "vibration": float(
                    row["vibration"]
                ),
                "exhaust_pressure": float(
                    row["exhaust_pressure"]
                ),
            },
        ),
        (
            "PROCESS",
            config["process"],
            {
                "bath_age_hours": float(
                    row["bath_age_hours"]
                ),
                "diw_resistivity": float(
                    row["diw_resistivity"]
                ),
                "concentration_deviation_percent": (
                    derived_values[
                        "concentration_deviation_percent"
                    ]
                ),
                "temperature_deviation": (
                    derived_values[
                        "temperature_deviation"
                    ]
                ),
            },
        ),
    ]

    for anomaly_type, limits_group, values in evaluation_groups:
        for parameter_name, value in values.items():
            parameter_limits = limits_group[parameter_name]

            result = evaluate_parameter(
                lot_id=lot_id,
                equipment_id=equipment_id,
                parameter_name=parameter_name,
                value=value,
                limits=parameter_limits,
                anomaly_type=anomaly_type,
            )

            if result is not None:
                results.append(result)

    return results


def clear_existing_results(
    connection: sqlite3.Connection,
) -> None:
    """기존 규칙 기반 이상 감지 결과를 삭제합니다."""

    connection.execute(
        """
        DELETE FROM anomaly_result
        WHERE detection_method = 'RULE_BASED';
        """
    )


def save_results(
    connection: sqlite3.Connection,
    results: list[dict[str, Any]],
) -> None:
    """이상 감지 결과를 DB에 저장합니다."""

    if not results:
        return

    sql = """
    INSERT INTO anomaly_result (
        lot_id,
        equipment_id,
        detected_at,
        anomaly_type,
        parameter_name,
        measured_value,
        reference_value,
        severity,
        detection_method,
        message
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    rows = [
        (
            result["lot_id"],
            result["equipment_id"],
            result["detected_at"],
            result["anomaly_type"],
            result["parameter_name"],
            result["measured_value"],
            result["reference_value"],
            result["severity"],
            result["detection_method"],
            result["message"],
        )
        for result in results
    ]

    connection.executemany(sql, rows)


def run_rule_detection() -> list[dict[str, Any]]:
    """전체 LOT에 대해 규칙 기반 이상 감지를 실행합니다."""

    config = load_limits()
    connection = connect_database()

    try:
        dataframe = load_lot_data(connection)

        if dataframe.empty:
            raise RuntimeError(
                "분석할 LOT 데이터가 없습니다."
            )

        clear_existing_results(connection)

        all_results: list[dict[str, Any]] = []

        for _, row in dataframe.iterrows():
            row_results = detect_row_anomalies(
                row,
                config,
            )

            all_results.extend(row_results)

        save_results(
            connection,
            all_results,
        )

        connection.commit()

        return all_results

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def print_detection_summary() -> None:
    """이상 감지 결과 요약을 터미널에 출력합니다."""

    connection = connect_database()

    try:
        severity_query = """
        SELECT
            severity,
            COUNT(*) AS anomaly_count
        FROM anomaly_result
        WHERE detection_method = 'RULE_BASED'
        GROUP BY severity
        ORDER BY
            CASE severity
                WHEN 'CRITICAL' THEN 1
                WHEN 'WARNING' THEN 2
                WHEN 'WATCH' THEN 3
                ELSE 4
            END;
        """

        equipment_query = """
        SELECT
            equipment_id,
            COUNT(*) AS anomaly_count
        FROM anomaly_result
        WHERE detection_method = 'RULE_BASED'
        GROUP BY equipment_id
        ORDER BY anomaly_count DESC;
        """

        top_parameter_query = """
        SELECT
            parameter_name,
            COUNT(*) AS anomaly_count
        FROM anomaly_result
        WHERE detection_method = 'RULE_BASED'
        GROUP BY parameter_name
        ORDER BY anomaly_count DESC
        LIMIT 5;
        """

        severity_rows = connection.execute(
            severity_query
        ).fetchall()

        equipment_rows = connection.execute(
            equipment_query
        ).fetchall()

        parameter_rows = connection.execute(
            top_parameter_query
        ).fetchall()

        print("\n" + "=" * 70)
        print("이상 감지 등급별 요약")
        print("=" * 70)

        for severity, count in severity_rows:
            print(f"{severity:8s} | {count:4d}건")

        print("\n" + "=" * 70)
        print("설비별 이상 건수")
        print("=" * 70)

        for equipment_id, count in equipment_rows:
            print(f"{equipment_id:8s} | {count:4d}건")

        print("\n" + "=" * 70)
        print("주요 이상 항목 TOP 5")
        print("=" * 70)

        for parameter_name, count in parameter_rows:
            print(
                f"{parameter_name:35s} | "
                f"{count:4d}건"
            )

    finally:
        connection.close()


def main() -> None:
    try:
        results = run_rule_detection()

        print("=" * 70)
        print("규칙 기반 이상 감지 완료")
        print("=" * 70)
        print(f"전체 이상 감지 결과: {len(results)}건")

        print_detection_summary()

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        sqlite3.Error,
        KeyError,
    ) as error:
        print("=" * 70)
        print("이상 감지 실패")
        print(error)
        print("=" * 70)


if __name__ == "__main__":
    main()