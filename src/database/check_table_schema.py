from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)

TABLE_NAMES: Final[list[str]] = [
    "anomaly_result",
    "root_cause_result",
    "action_recommendation",
]


def connect_database() -> sqlite3.Connection:
    """SQLite 데이터베이스에 연결합니다."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"데이터베이스 파일을 찾을 수 없습니다.\n"
            f"경로: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def get_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[tuple]:
    """
    지정한 테이블의 컬럼 구조를 조회합니다.

    PRAGMA table_info 결과:
    0: 컬럼 순번
    1: 컬럼명
    2: 데이터 타입
    3: NOT NULL 여부
    4: 기본값
    5: Primary Key 여부
    """

    query = f"PRAGMA table_info({table_name});"

    return connection.execute(query).fetchall()


def print_table_schema(
    connection: sqlite3.Connection,
    table_name: str,
) -> None:
    """테이블의 컬럼 정보를 보기 좋게 출력합니다."""

    columns = get_table_columns(
        connection,
        table_name,
    )

    print("\n" + "=" * 90)
    print(f"테이블: {table_name}")
    print("=" * 90)

    if not columns:
        print("테이블이 존재하지 않거나 컬럼을 찾을 수 없습니다.")
        return

    print(
        f"{'순번':<6}"
        f"{'컬럼명':<35}"
        f"{'타입':<15}"
        f"{'NOT NULL':<12}"
        f"{'PK':<6}"
        f"{'기본값'}"
    )

    print("-" * 90)

    for column in columns:
        column_id = column[0]
        column_name = column[1]
        data_type = column[2]
        not_null = "YES" if column[3] == 1 else "NO"
        default_value = column[4]
        primary_key = "YES" if column[5] == 1 else "NO"

        print(
            f"{column_id:<6}"
            f"{column_name:<35}"
            f"{data_type:<15}"
            f"{not_null:<12}"
            f"{primary_key:<6}"
            f"{default_value}"
        )


def main() -> None:
    try:
        connection = connect_database()

        try:
            print("=" * 90)
            print("분석 결과 테이블 구조 확인")
            print("=" * 90)

            for table_name in TABLE_NAMES:
                print_table_schema(
                    connection,
                    table_name,
                )

        finally:
            connection.close()

    except FileNotFoundError as error:
        print("=" * 90)
        print("테이블 구조 확인 실패")
        print(error)
        print("=" * 90)


if __name__ == "__main__":
    main()