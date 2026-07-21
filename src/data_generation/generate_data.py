from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)

RANDOM_SEED: Final[int] = 42
NUMBER_OF_LOTS: Final[int] = 120
WAFERS_PER_LOT: Final[int] = 25
SENSOR_SAMPLES_PER_LOT: Final[int] = 6


def connect_database() -> sqlite3.Connection:
    """SQLite 데이터베이스에 연결합니다."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "wet_cleaning.db가 존재하지 않습니다.\n"
            "먼저 다음 명령을 실행하세요.\n"
            "python src/database/create_database.py"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def validate_master_data(
    connection: sqlite3.Connection,
) -> None:
    """필수 마스터 데이터가 등록되어 있는지 확인합니다."""

    required_tables = {
        "equipment_master": 3,
        "recipe_master": 4,
        "chemical_lot": 4,
    }

    for table_name, minimum_count in required_tables.items():
        query = f"SELECT COUNT(*) FROM {table_name};"
        count = connection.execute(query).fetchone()[0]

        if count < minimum_count:
            raise RuntimeError(
                f"{table_name} 마스터 데이터가 부족합니다.\n"
                "먼저 다음 명령을 실행하세요.\n"
                "python src/database/seed_master_data.py"
            )


def clear_generated_data(
    connection: sqlite3.Connection,
) -> None:
    """
    기존 가상 생산 데이터를 삭제합니다.

    마스터 데이터는 유지하고 LOT, 센서, 품질 등의
    생성 데이터만 초기화합니다.
    """

    tables = [
        "action_recommendation",
        "root_cause_result",
        "anomaly_result",
        "quality_result",
        "sensor_history",
        "process_condition",
        "wafer_history",
        "maintenance_history",
        "lot_history",
    ]

    for table_name in tables:
        connection.execute(f"DELETE FROM {table_name};")


def get_recipe_master(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, float | str]]:
    """Recipe 기준정보를 딕셔너리 형태로 조회합니다."""

    query = """
    SELECT
        recipe_id,
        chemical_type,
        target_concentration,
        target_bath_temperature,
        target_cleaning_time,
        target_rinse_time,
        target_drying_time,
        target_megasonic_power,
        target_spin_speed,
        target_diw_resistivity
    FROM recipe_master;
    """

    rows = connection.execute(query).fetchall()

    recipes = {}

    for row in rows:
        recipes[row[0]] = {
            "chemical_type": row[1],
            "concentration": row[2],
            "bath_temperature": row[3],
            "cleaning_time": row[4],
            "rinse_time": row[5],
            "drying_time": row[6],
            "megasonic_power": row[7],
            "spin_speed": row[8],
            "diw_resistivity": row[9],
        }

    return recipes


def get_chemical_lot_mapping(
    connection: sqlite3.Connection,
) -> dict[str, str]:
    """약액 종류별 Chemical LOT ID를 조회합니다."""

    query = """
    SELECT
        chemical_type,
        chemical_lot_id
    FROM chemical_lot
    WHERE coa_result = 'PASS';
    """

    rows = connection.execute(query).fetchall()

    return {
        chemical_type: chemical_lot_id
        for chemical_type, chemical_lot_id in rows
    }


def calculate_degradation_factor(
    lot_index: int,
    equipment_id: str,
) -> float:
    """
    WC-03 설비의 점진적인 Filter 열화 정도를 계산합니다.

    80번째 LOT 이후 WC-03에서만 열화가 진행됩니다.
    """

    if equipment_id != "WC-03":
        return 0.0

    if lot_index < 80:
        return 0.0

    return min((lot_index - 79) / 25, 1.0)


def create_process_values(
    rng: np.random.Generator,
    recipe: dict[str, float | str],
    degradation_factor: float,
) -> dict[str, float]:
    """Recipe 목표값을 기준으로 실제 공정조건을 생성합니다."""

    bath_age = rng.uniform(2.0, 14.0)

    if degradation_factor > 0:
        bath_age += degradation_factor * rng.uniform(5.0, 12.0)

    return {
        "chemical_concentration": float(
            rng.normal(recipe["concentration"], 0.05)
        ),
        "bath_temperature": float(
            rng.normal(recipe["bath_temperature"], 0.7)
        ),
        "cleaning_time": float(
            rng.normal(recipe["cleaning_time"], 5.0)
        ),
        "rinse_time": float(
            rng.normal(recipe["rinse_time"], 4.0)
        ),
        "drying_time": float(
            rng.normal(recipe["drying_time"], 3.0)
        ),
        "megasonic_power": float(
            rng.normal(recipe["megasonic_power"], 1.5)
        ),
        "spin_speed": float(
            rng.normal(recipe["spin_speed"], 15.0)
        ),
        "diw_resistivity": float(
            rng.normal(recipe["diw_resistivity"], 0.08)
        ),
        "bath_age_hours": float(bath_age),
    }


def create_sensor_values(
    rng: np.random.Generator,
    degradation_factor: float,
) -> dict[str, float]:
    """
    설비 센서값을 생성합니다.

    열화가 진행될수록:
    - Filter 차압 상승
    - 유량 감소
    - Pump 압력 상승
    - 진동 증가
    """

    return {
        "pump_pressure": float(
            rng.normal(
                2.10 + degradation_factor * 0.35,
                0.05,
            )
        ),
        "flow_rate": float(
            rng.normal(
                18.0 - degradation_factor * 2.7,
                0.35,
            )
        ),
        "filter_differential_pressure": float(
            rng.normal(
                0.85 + degradation_factor * 1.25,
                0.07,
            )
        ),
        "nozzle_pressure": float(
            rng.normal(
                1.55 - degradation_factor * 0.18,
                0.04,
            )
        ),
        "motor_current": float(
            rng.normal(
                7.20 + degradation_factor * 0.45,
                0.12,
            )
        ),
        "vibration": float(
            rng.normal(
                0.75 + degradation_factor * 0.55,
                0.08,
            )
        ),
        "exhaust_pressure": float(
            rng.normal(-0.42, 0.025)
        ),
    }


def create_quality_values(
    rng: np.random.Generator,
    process_values: dict[str, float],
    average_sensor: dict[str, float],
    degradation_factor: float,
) -> dict[str, float | int | str]:
    """공정조건과 설비 상태를 반영해 품질 결과를 생성합니다."""

    filter_effect = max(
        average_sensor["filter_differential_pressure"] - 1.0,
        0.0,
    )

    flow_effect = max(
        17.0 - average_sensor["flow_rate"],
        0.0,
    )

    bath_effect = max(
        process_values["bath_age_hours"] - 12.0,
        0.0,
    )

    diw_effect = max(
        17.8 - process_values["diw_resistivity"],
        0.0,
    )

    expected_particles = (
        7.0
        + filter_effect * 12.0
        + flow_effect * 2.8
        + bath_effect * 0.8
        + degradation_factor * 5.0
    )

    particle_count = max(
        int(rng.poisson(max(expected_particles, 1.0))),
        0,
    )

    metal_contamination = max(
        float(
            rng.normal(
                0.35 + bath_effect * 0.035,
                0.08,
            )
        ),
        0.05,
    )

    water_mark_count = max(
        int(
            rng.poisson(
                0.7 + diw_effect * 3.0
            )
        ),
        0,
    )

    organic_residue_count = max(
        int(
            rng.poisson(
                0.5 + bath_effect * 0.25
            )
        ),
        0,
    )

    excessive_megasonic = max(
        process_values["megasonic_power"] - 62.0,
        0.0,
    )

    pattern_collapse_count = max(
        int(
            rng.poisson(
                0.15 + excessive_megasonic * 0.12
            )
        ),
        0,
    )

    yield_loss = (
        particle_count * 0.075
        + metal_contamination * 0.7
        + water_mark_count * 0.18
        + organic_residue_count * 0.22
        + pattern_collapse_count * 0.35
    )

    yield_percent = float(
        np.clip(
            rng.normal(98.3 - yield_loss, 0.35),
            85.0,
            99.5,
        )
    )

    if yield_percent < 94.0 or particle_count >= 25:
        inspection_result = "FAIL"
        lot_status = "FAIL"
    elif yield_percent < 96.0 or particle_count >= 18:
        inspection_result = "HOLD"
        lot_status = "HOLD"
    else:
        inspection_result = "PASS"
        lot_status = "PASS"

    return {
        "particle_count": particle_count,
        "metal_contamination_ppb": metal_contamination,
        "water_mark_count": water_mark_count,
        "organic_residue_count": organic_residue_count,
        "pattern_collapse_count": pattern_collapse_count,
        "yield_percent": yield_percent,
        "inspection_result": inspection_result,
        "lot_status": lot_status,
    }


def insert_maintenance_history(
    connection: sqlite3.Connection,
) -> None:
    """설비별 초기 유지보수 이력을 등록합니다."""

    maintenance_rows = [
        (
            "WC-01",
            "2026-06-20 09:00:00",
            "FILTER",
            "REPLACEMENT",
            "정기 PM에 따른 Filter 교체",
            1,
            680.0,
            720.0,
            "ENGINEER_A",
        ),
        (
            "WC-02",
            "2026-06-24 09:00:00",
            "NOZZLE",
            "CLEANING",
            "Nozzle 분사 상태 점검 및 세정",
            0,
            430.0,
            290.0,
            "ENGINEER_B",
        ),
        (
            "WC-03",
            "2026-05-25 09:00:00",
            "FILTER",
            "INSPECTION",
            "Filter 차압 점검, 교체 보류",
            0,
            690.0,
            30.0,
            "ENGINEER_C",
        ),
    ]

    sql = """
    INSERT INTO maintenance_history (
        equipment_id,
        maintenance_date,
        component,
        maintenance_type,
        action_description,
        replacement_flag,
        usage_hours,
        remaining_life_hours,
        engineer_name
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    connection.executemany(sql, maintenance_rows)


def generate_production_data() -> None:
    """가상의 Wet Cleaning 생산 데이터를 생성합니다."""

    rng = np.random.default_rng(RANDOM_SEED)
    connection = connect_database()

    try:
        validate_master_data(connection)
        clear_generated_data(connection)

        recipes = get_recipe_master(connection)
        chemical_mapping = get_chemical_lot_mapping(connection)

        equipment_ids = ["WC-01", "WC-02", "WC-03"]
        recipe_ids = list(recipes.keys())

        start_datetime = datetime(2026, 7, 1, 8, 0, 0)

        for lot_index in range(NUMBER_OF_LOTS):
            lot_number = lot_index + 1
            lot_id = f"LOT-202607-{lot_number:04d}"

            equipment_id = equipment_ids[lot_index % 3]

            recipe_id = str(
                rng.choice(
                    recipe_ids,
                    p=[0.35, 0.25, 0.25, 0.15],
                )
            )

            recipe = recipes[recipe_id]
            chemical_type = str(recipe["chemical_type"])
            chemical_lot_id = chemical_mapping[chemical_type]

            product_type = str(
                rng.choice(
                    ["DRAM", "NAND", "HBM"],
                    p=[0.45, 0.35, 0.20],
                )
            )

            process_step = str(
                rng.choice(
                    [
                        "PRE_CLEAN",
                        "POST_ETCH_CLEAN",
                        "PRE_DEPOSITION_CLEAN",
                    ],
                    p=[0.30, 0.40, 0.30],
                )
            )

            lot_start = start_datetime + timedelta(
                hours=lot_index * 2
            )

            lot_end = lot_start + timedelta(
                minutes=float(rng.normal(52.0, 3.0))
            )

            degradation_factor = calculate_degradation_factor(
                lot_index,
                equipment_id,
            )

            process_values = create_process_values(
                rng,
                recipe,
                degradation_factor,
            )

            sensor_rows = []
            sensor_values_collection = []

            for sample_index in range(SENSOR_SAMPLES_PER_LOT):
                measured_at = lot_start + timedelta(
                    minutes=sample_index * 8
                )

                sensor_values = create_sensor_values(
                    rng,
                    degradation_factor,
                )

                sensor_values_collection.append(sensor_values)

                alarm_code = None

                if (
                    sensor_values[
                        "filter_differential_pressure"
                    ]
                    >= 2.0
                ):
                    alarm_code = "ALM-FILTER-DP-HIGH"

                sensor_rows.append(
                    (
                        lot_id,
                        equipment_id,
                        measured_at.isoformat(
                            sep=" ",
                            timespec="seconds",
                        ),
                        sensor_values["pump_pressure"],
                        sensor_values["flow_rate"],
                        sensor_values[
                            "filter_differential_pressure"
                        ],
                        sensor_values["nozzle_pressure"],
                        sensor_values["motor_current"],
                        sensor_values["vibration"],
                        sensor_values["exhaust_pressure"],
                        alarm_code,
                    )
                )

            average_sensor = {
                key: float(
                    np.mean(
                        [
                            sensor[key]
                            for sensor in sensor_values_collection
                        ]
                    )
                )
                for key in sensor_values_collection[0]
            }

            quality_values = create_quality_values(
                rng,
                process_values,
                average_sensor,
                degradation_factor,
            )

            connection.execute(
                """
                INSERT INTO lot_history (
                    lot_id,
                    product_type,
                    process_step,
                    recipe_id,
                    equipment_id,
                    chemical_lot_id,
                    start_time,
                    end_time,
                    lot_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    lot_id,
                    product_type,
                    process_step,
                    recipe_id,
                    equipment_id,
                    chemical_lot_id,
                    lot_start.isoformat(
                        sep=" ",
                        timespec="seconds",
                    ),
                    lot_end.isoformat(
                        sep=" ",
                        timespec="seconds",
                    ),
                    quality_values["lot_status"],
                ),
            )

            wafer_rows = []

            failed_wafer_count = int(
                round(
                    WAFERS_PER_LOT
                    * (
                        100.0
                        - float(
                            quality_values["yield_percent"]
                        )
                    )
                    / 100.0
                )
            )

            failed_wafer_numbers = set(
                rng.choice(
                    np.arange(1, WAFERS_PER_LOT + 1),
                    size=min(
                        failed_wafer_count,
                        WAFERS_PER_LOT,
                    ),
                    replace=False,
                ).tolist()
            )

            for wafer_number in range(
                1,
                WAFERS_PER_LOT + 1,
            ):
                wafer_status = (
                    "FAIL"
                    if wafer_number in failed_wafer_numbers
                    else "PASS"
                )

                wafer_rows.append(
                    (
                        f"{lot_id}-W{wafer_number:02d}",
                        lot_id,
                        wafer_number,
                        wafer_status,
                    )
                )

            connection.executemany(
                """
                INSERT INTO wafer_history (
                    wafer_id,
                    lot_id,
                    wafer_number,
                    wafer_status
                )
                VALUES (?, ?, ?, ?);
                """,
                wafer_rows,
            )

            connection.execute(
                """
                INSERT INTO process_condition (
                    lot_id,
                    chemical_concentration,
                    bath_temperature,
                    cleaning_time,
                    rinse_time,
                    drying_time,
                    megasonic_power,
                    spin_speed,
                    diw_resistivity,
                    bath_age_hours
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    lot_id,
                    process_values[
                        "chemical_concentration"
                    ],
                    process_values["bath_temperature"],
                    process_values["cleaning_time"],
                    process_values["rinse_time"],
                    process_values["drying_time"],
                    process_values["megasonic_power"],
                    process_values["spin_speed"],
                    process_values["diw_resistivity"],
                    process_values["bath_age_hours"],
                ),
            )

            connection.executemany(
                """
                INSERT INTO sensor_history (
                    lot_id,
                    equipment_id,
                    measured_at,
                    pump_pressure,
                    flow_rate,
                    filter_differential_pressure,
                    nozzle_pressure,
                    motor_current,
                    vibration,
                    exhaust_pressure,
                    alarm_code
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                sensor_rows,
            )

            connection.execute(
                """
                INSERT INTO quality_result (
                    lot_id,
                    particle_count,
                    metal_contamination_ppb,
                    water_mark_count,
                    organic_residue_count,
                    pattern_collapse_count,
                    yield_percent,
                    inspection_result,
                    inspected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    lot_id,
                    quality_values["particle_count"],
                    quality_values[
                        "metal_contamination_ppb"
                    ],
                    quality_values["water_mark_count"],
                    quality_values[
                        "organic_residue_count"
                    ],
                    quality_values[
                        "pattern_collapse_count"
                    ],
                    quality_values["yield_percent"],
                    quality_values["inspection_result"],
                    (
                        lot_end + timedelta(minutes=30)
                    ).isoformat(
                        sep=" ",
                        timespec="seconds",
                    ),
                ),
            )

        insert_maintenance_history(connection)
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_generated_counts() -> dict[str, int]:
    """생성된 데이터 건수를 조회합니다."""

    connection = connect_database()

    try:
        table_names = [
            "lot_history",
            "wafer_history",
            "process_condition",
            "sensor_history",
            "quality_result",
            "maintenance_history",
        ]

        return {
            table_name: connection.execute(
                f"SELECT COUNT(*) FROM {table_name};"
            ).fetchone()[0]
            for table_name in table_names
        }

    finally:
        connection.close()


def print_quality_summary() -> None:
    """설비별 품질 요약을 출력합니다."""

    connection = connect_database()

    try:
        query = """
        SELECT
            l.equipment_id,
            COUNT(*) AS lot_count,
            ROUND(AVG(q.yield_percent), 2) AS avg_yield,
            ROUND(AVG(q.particle_count), 2) AS avg_particle,
            SUM(
                CASE
                    WHEN l.lot_status = 'FAIL' THEN 1
                    ELSE 0
                END
            ) AS fail_lots
        FROM lot_history AS l
        INNER JOIN quality_result AS q
            ON l.lot_id = q.lot_id
        GROUP BY l.equipment_id
        ORDER BY l.equipment_id;
        """

        rows = connection.execute(query).fetchall()

        print("\n" + "=" * 72)
        print("설비별 품질 요약")
        print("=" * 72)

        for row in rows:
            print(
                f"{row[0]} | "
                f"LOT {row[1]:3d}개 | "
                f"평균 수율 {row[2]:5.2f}% | "
                f"평균 Particle {row[3]:5.2f} | "
                f"FAIL {row[4]}개"
            )

    finally:
        connection.close()


def main() -> None:
    try:
        generate_production_data()
        counts = get_generated_counts()

        print("=" * 72)
        print("Wet Cleaning 가상 생산 데이터 생성 완료")
        print("=" * 72)
        print(f"LOT: {counts['lot_history']}건")
        print(f"Wafer: {counts['wafer_history']}건")
        print(
            f"공정조건: "
            f"{counts['process_condition']}건"
        )
        print(
            f"센서 데이터: "
            f"{counts['sensor_history']}건"
        )
        print(
            f"품질 결과: "
            f"{counts['quality_result']}건"
        )
        print(
            f"유지보수 이력: "
            f"{counts['maintenance_history']}건"
        )

        print_quality_summary()

    except (FileNotFoundError, RuntimeError) as error:
        print("=" * 72)
        print("데이터 생성 실패")
        print(error)
        print("=" * 72)

    except sqlite3.Error as error:
        print("=" * 72)
        print("SQLite 처리 중 오류 발생")
        print(error)
        print("=" * 72)


if __name__ == "__main__":
    main()