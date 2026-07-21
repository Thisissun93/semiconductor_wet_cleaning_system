from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "wet_cleaning.db"


@pytest.fixture
def connection() -> sqlite3.Connection:
    """
    각 테스트에서 사용할 SQLite 연결을 제공합니다.
    테스트 종료 후 자동으로 연결을 닫습니다.
    """
    if not DATABASE_PATH.exists():
        pytest.fail(
            "wet_cleaning.db가 존재하지 않습니다.\n"
            "다음 명령을 먼저 실행하세요.\n"
            "python src/database/create_database.py\n"
            "python src/database/seed_master_data.py\n"
            "python src/data_generation/generate_data.py"
        )

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        yield conn
    finally:
        conn.close()


def get_table_count(
    connection: sqlite3.Connection,
    table_name: str,
) -> int:
    """지정한 테이블의 데이터 건수를 반환합니다."""

    query = f"SELECT COUNT(*) FROM {table_name};"
    return int(connection.execute(query).fetchone()[0])


def test_master_data_counts(
    connection: sqlite3.Connection,
) -> None:
    """
    설비, Recipe, 약액 마스터 데이터가
    정해진 수만큼 존재하는지 확인합니다.
    """

    assert get_table_count(connection, "equipment_master") == 3
    assert get_table_count(connection, "recipe_master") == 4
    assert get_table_count(connection, "chemical_lot") == 4


def test_generated_data_counts(
    connection: sqlite3.Connection,
) -> None:
    """
    가상 생산 데이터 건수가 설계값과 일치하는지 확인합니다.
    """

    assert get_table_count(connection, "lot_history") == 120
    assert get_table_count(connection, "wafer_history") == 3000
    assert get_table_count(connection, "process_condition") == 120
    assert get_table_count(connection, "sensor_history") == 720
    assert get_table_count(connection, "quality_result") == 120
    assert get_table_count(connection, "maintenance_history") == 3


def test_every_lot_has_process_condition(
    connection: sqlite3.Connection,
) -> None:
    """
    모든 LOT에 공정조건 데이터가 연결되어 있는지 확인합니다.
    """

    query = """
    SELECT COUNT(*)
    FROM lot_history AS l
    LEFT JOIN process_condition AS p
        ON l.lot_id = p.lot_id
    WHERE p.lot_id IS NULL;
    """

    missing_count = connection.execute(query).fetchone()[0]

    assert missing_count == 0


def test_every_lot_has_quality_result(
    connection: sqlite3.Connection,
) -> None:
    """
    모든 LOT에 품질검사 결과가 존재하는지 확인합니다.
    """

    query = """
    SELECT COUNT(*)
    FROM lot_history AS l
    LEFT JOIN quality_result AS q
        ON l.lot_id = q.lot_id
    WHERE q.lot_id IS NULL;
    """

    missing_count = connection.execute(query).fetchone()[0]

    assert missing_count == 0


def test_every_lot_has_six_sensor_samples(
    connection: sqlite3.Connection,
) -> None:
    """
    각 LOT에 센서 데이터가 정확히 6건씩 존재하는지 확인합니다.
    """

    query = """
    SELECT
        lot_id,
        COUNT(*) AS sensor_count
    FROM sensor_history
    GROUP BY lot_id
    HAVING COUNT(*) != 6;
    """

    invalid_rows = connection.execute(query).fetchall()

    assert invalid_rows == []


def test_yield_range(
    connection: sqlite3.Connection,
) -> None:
    """
    수율이 현실적인 범위 안에 있는지 확인합니다.
    """

    query = """
    SELECT
        MIN(yield_percent),
        MAX(yield_percent)
    FROM quality_result;
    """

    minimum_yield, maximum_yield = connection.execute(
        query
    ).fetchone()

    assert minimum_yield >= 85.0
    assert maximum_yield <= 100.0


def test_non_negative_quality_values(
    connection: sqlite3.Connection,
) -> None:
    """
    Particle, Water Mark 등 품질값에
    음수가 존재하지 않는지 확인합니다.
    """

    query = """
    SELECT COUNT(*)
    FROM quality_result
    WHERE
        particle_count < 0
        OR metal_contamination_ppb < 0
        OR water_mark_count < 0
        OR organic_residue_count < 0
        OR pattern_collapse_count < 0;
    """

    invalid_count = connection.execute(query).fetchone()[0]

    assert invalid_count == 0


def test_valid_lot_status(
    connection: sqlite3.Connection,
) -> None:
    """
    LOT 상태값이 PASS, HOLD, FAIL 중 하나인지 확인합니다.
    """

    query = """
    SELECT DISTINCT lot_status
    FROM lot_history;
    """

    statuses = {
        row[0]
        for row in connection.execute(query).fetchall()
    }

    assert statuses.issubset({"PASS", "HOLD", "FAIL"})


def test_wc03_filter_degradation_scenario(
    connection: sqlite3.Connection,
) -> None:
    """
    WC-03 후반부 데이터에서 Filter 차압이
    전반부보다 높아지는지 확인합니다.

    이는 이번 프로젝트에 의도적으로 반영한
    설비 열화 시나리오입니다.
    """

    query = """
    SELECT
        l.start_time,
        AVG(s.filter_differential_pressure) AS avg_filter_dp
    FROM lot_history AS l
    INNER JOIN sensor_history AS s
        ON l.lot_id = s.lot_id
    WHERE l.equipment_id = 'WC-03'
    GROUP BY l.lot_id, l.start_time
    ORDER BY l.start_time;
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
    )

    assert len(dataframe) == 40

    early_average = dataframe.iloc[:20][
        "avg_filter_dp"
    ].mean()

    late_average = dataframe.iloc[-10:][
        "avg_filter_dp"
    ].mean()

    assert late_average > early_average + 0.20


def test_wc03_quality_is_worse_than_normal_equipment(
    connection: sqlite3.Connection,
) -> None:
    """
    열화 시나리오가 적용된 WC-03의 Particle 평균이
    WC-01과 WC-02 평균보다 높은지 확인합니다.
    """

    query = """
    SELECT
        l.equipment_id,
        AVG(q.particle_count) AS avg_particle,
        AVG(q.yield_percent) AS avg_yield
    FROM lot_history AS l
    INNER JOIN quality_result AS q
        ON l.lot_id = q.lot_id
    GROUP BY l.equipment_id
    ORDER BY l.equipment_id;
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
    ).set_index("equipment_id")

    wc03_particle = dataframe.loc[
        "WC-03",
        "avg_particle",
    ]

    normal_particle_average = dataframe.loc[
        ["WC-01", "WC-02"],
        "avg_particle",
    ].mean()

    assert wc03_particle > normal_particle_average