from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)


EQUIPMENT_DATA = [
    (
        "WC-01",
        "Wet Cleaner 01",
        "WET_CLEANER",
        "FAB1-CLEAN-01",
        "2022-03-15",
        "RUN",
    ),
    (
        "WC-02",
        "Wet Cleaner 02",
        "WET_CLEANER",
        "FAB1-CLEAN-02",
        "2022-05-20",
        "RUN",
    ),
    (
        "WC-03",
        "Wet Cleaner 03",
        "WET_CLEANER",
        "FAB1-CLEAN-03",
        "2023-01-10",
        "RUN",
    ),
]


RECIPE_DATA = [
    (
        "RCP-SC1-01",
        "SC1 Standard Clean",
        "SC1",
        4.50,    # 목표 농도 (%)
        65.0,    # Bath 온도 (℃)
        600.0,   # Cleaning 시간 (초)
        300.0,   # Rinse 시간 (초)
        180.0,   # Drying 시간 (초)
        60.0,    # Megasonic 출력 (%)
        1500.0,  # Spin 속도 (rpm)
        18.0,    # DIW 비저항 (MΩ·cm)
    ),
    (
        "RCP-SC2-01",
        "SC2 Metal Removal",
        "SC2",
        3.50,
        65.0,
        480.0,
        300.0,
        180.0,
        50.0,
        1500.0,
        18.0,
    ),
    (
        "RCP-DHF-01",
        "Dilute HF Native Oxide Removal",
        "DHF",
        0.50,
        25.0,
        90.0,
        240.0,
        180.0,
        20.0,
        1400.0,
        18.0,
    ),
    (
        "RCP-O3-01",
        "Ozone DI Water Clean",
        "O3_DIW",
        2.00,
        25.0,
        300.0,
        300.0,
        180.0,
        40.0,
        1500.0,
        18.0,
    ),
]


CHEMICAL_LOT_DATA = [
    (
        "CHM-SC1-260701",
        "SC1",
        "CleanChem Korea",
        "2026-07-01",
        "2026-08-01",
        99.99,  # 순도 (%)
        0.80,   # 금속 오염도 (ppb)
        12.0,   # Particle level
        11.20,  # pH
        145.0,  # 전도도
        "PASS",
    ),
    (
        "CHM-SC2-260701",
        "SC2",
        "CleanChem Korea",
        "2026-07-01",
        "2026-08-01",
        99.99,
        0.55,
        9.0,
        1.20,
        175.0,
        "PASS",
    ),
    (
        "CHM-DHF-260705",
        "DHF",
        "PureEtch Materials",
        "2026-07-05",
        "2026-09-05",
        99.999,
        0.30,
        7.0,
        2.40,
        85.0,
        "PASS",
    ),
    (
        "CHM-O3-260708",
        "O3_DIW",
        "Fab Utility System",
        "2026-07-08",
        "2026-07-29",
        99.999,
        0.20,
        5.0,
        6.80,
        8.0,
        "PASS",
    ),
]


def connect_database() -> sqlite3.Connection:
    """
    SQLite 데이터베이스에 연결합니다.

    Returns:
        sqlite3.Connection: 외래키 검사가 활성화된 DB 연결
    """
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "wet_cleaning.db가 존재하지 않습니다.\n"
            "먼저 아래 명령을 실행하세요.\n"
            "python src/database/create_database.py"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def seed_equipment(
    connection: sqlite3.Connection,
) -> None:
    """설비 마스터 데이터를 등록합니다."""

    sql = """
    INSERT INTO equipment_master (
        equipment_id,
        equipment_name,
        equipment_type,
        location,
        install_date,
        status
    )
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(equipment_id) DO UPDATE SET
        equipment_name = excluded.equipment_name,
        equipment_type = excluded.equipment_type,
        location = excluded.location,
        install_date = excluded.install_date,
        status = excluded.status;
    """

    connection.executemany(sql, EQUIPMENT_DATA)


def seed_recipes(
    connection: sqlite3.Connection,
) -> None:
    """Recipe 마스터 데이터를 등록합니다."""

    sql = """
    INSERT INTO recipe_master (
        recipe_id,
        recipe_name,
        chemical_type,
        target_concentration,
        target_bath_temperature,
        target_cleaning_time,
        target_rinse_time,
        target_drying_time,
        target_megasonic_power,
        target_spin_speed,
        target_diw_resistivity
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(recipe_id) DO UPDATE SET
        recipe_name = excluded.recipe_name,
        chemical_type = excluded.chemical_type,
        target_concentration = excluded.target_concentration,
        target_bath_temperature =
            excluded.target_bath_temperature,
        target_cleaning_time =
            excluded.target_cleaning_time,
        target_rinse_time =
            excluded.target_rinse_time,
        target_drying_time =
            excluded.target_drying_time,
        target_megasonic_power =
            excluded.target_megasonic_power,
        target_spin_speed =
            excluded.target_spin_speed,
        target_diw_resistivity =
            excluded.target_diw_resistivity;
    """

    connection.executemany(sql, RECIPE_DATA)


def seed_chemical_lots(
    connection: sqlite3.Connection,
) -> None:
    """약액 LOT 데이터를 등록합니다."""

    sql = """
    INSERT INTO chemical_lot (
        chemical_lot_id,
        chemical_type,
        supplier,
        manufacture_date,
        expiry_date,
        purity,
        metal_contamination_ppb,
        particle_level,
        ph,
        conductivity,
        coa_result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(chemical_lot_id) DO UPDATE SET
        chemical_type = excluded.chemical_type,
        supplier = excluded.supplier,
        manufacture_date = excluded.manufacture_date,
        expiry_date = excluded.expiry_date,
        purity = excluded.purity,
        metal_contamination_ppb =
            excluded.metal_contamination_ppb,
        particle_level = excluded.particle_level,
        ph = excluded.ph,
        conductivity = excluded.conductivity,
        coa_result = excluded.coa_result;
    """

    connection.executemany(sql, CHEMICAL_LOT_DATA)


def seed_master_data() -> None:
    """
    모든 마스터 데이터를 하나의 트랜잭션으로 등록합니다.
    """
    connection = connect_database()

    try:
        seed_equipment(connection)
        seed_recipes(connection)
        seed_chemical_lots(connection)

        connection.commit()

    except sqlite3.Error as error:
        connection.rollback()

        raise RuntimeError(
            f"마스터 데이터 등록 중 오류가 발생했습니다: {error}"
        ) from error

    finally:
        connection.close()


def get_record_counts() -> dict[str, int]:
    """
    등록된 마스터 데이터 건수를 조회합니다.

    Returns:
        dict[str, int]: 테이블별 데이터 건수
    """
    connection = connect_database()

    try:
        table_names = [
            "equipment_master",
            "recipe_master",
            "chemical_lot",
        ]

        counts = {}

        for table_name in table_names:
            query = f"SELECT COUNT(*) FROM {table_name};"
            count = connection.execute(query).fetchone()[0]
            counts[table_name] = count

        return counts

    finally:
        connection.close()


def main() -> None:
    try:
        seed_master_data()
        counts = get_record_counts()

        print("=" * 60)
        print("마스터 데이터 등록 완료")
        print("=" * 60)
        print(
            f"설비: {counts['equipment_master']}대"
        )
        print(
            f"Recipe: {counts['recipe_master']}개"
        )
        print(
            f"약액 LOT: {counts['chemical_lot']}개"
        )
        print("=" * 60)

    except (FileNotFoundError, RuntimeError) as error:
        print("=" * 60)
        print("마스터 데이터 등록 실패")
        print(error)
        print("=" * 60)


if __name__ == "__main__":
    main()