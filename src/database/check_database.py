import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)


def print_table_names(
    connection: sqlite3.Connection,
) -> None:
    query = """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name;
    """

    rows = connection.execute(query).fetchall()

    print("=" * 60)
    print("생성된 테이블")
    print("=" * 60)

    for row in rows:
        print(row[0])


def print_equipment(
    connection: sqlite3.Connection,
) -> None:
    query = """
    SELECT
        equipment_id,
        equipment_name,
        location,
        status
    FROM equipment_master
    ORDER BY equipment_id;
    """

    rows = connection.execute(query).fetchall()

    print("\n" + "=" * 60)
    print("설비 마스터")
    print("=" * 60)

    for row in rows:
        print(
            f"{row[0]} | {row[1]} | "
            f"{row[2]} | {row[3]}"
        )


def print_recipes(
    connection: sqlite3.Connection,
) -> None:
    query = """
    SELECT
        recipe_id,
        recipe_name,
        chemical_type,
        target_bath_temperature
    FROM recipe_master
    ORDER BY recipe_id;
    """

    rows = connection.execute(query).fetchall()

    print("\n" + "=" * 60)
    print("Recipe 마스터")
    print("=" * 60)

    for row in rows:
        print(
            f"{row[0]} | {row[1]} | "
            f"{row[2]} | {row[3]} ℃"
        )


def main() -> None:
    if not DATABASE_PATH.exists():
        print("DB 파일이 존재하지 않습니다.")
        return

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        print_table_names(connection)
        print_equipment(connection)
        print_recipes(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    main()