from pathlib import Path
import sqlite3


def get_project_root() -> Path:
    """
    현재 파일 위치를 기준으로 프로젝트 최상위 폴더 경로를 반환합니다.

    현재 파일:
    semiconductor_wet_cleaning_system/src/database/create_database.py

    parents[0] -> src/database
    parents[1] -> src
    parents[2] -> 프로젝트 최상위 폴더
    """
    return Path(__file__).resolve().parents[2]


def create_database() -> Path:
    """
    database/schema.sql 파일을 읽어
    database/wet_cleaning.db 데이터베이스를 생성합니다.

    Returns:
        Path: 생성된 데이터베이스 파일 경로
    """
    project_root = get_project_root()

    schema_path = project_root / "database" / "schema.sql"
    database_path = project_root / "database" / "wet_cleaning.db"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"schema.sql 파일을 찾을 수 없습니다: {schema_path}"
        )

    schema_sql = schema_path.read_text(encoding="utf-8")

    connection = None

    try:
        connection = sqlite3.connect(database_path)

        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript(schema_sql)
        connection.commit()

    except sqlite3.Error as error:
        if connection is not None:
            connection.rollback()

        raise RuntimeError(
            f"데이터베이스 생성 중 오류가 발생했습니다: {error}"
        ) from error

    finally:
        if connection is not None:
            connection.close()

    return database_path


def main() -> None:
    """
    파일을 직접 실행했을 때 데이터베이스를 생성하고
    생성 결과를 터미널에 출력합니다.
    """
    try:
        database_path = create_database()

        print("=" * 60)
        print("Wet Cleaning 데이터베이스 생성 완료")
        print(f"생성 경로: {database_path}")
        print("=" * 60)

    except (FileNotFoundError, RuntimeError) as error:
        print("=" * 60)
        print("데이터베이스 생성 실패")
        print(error)
        print("=" * 60)


if __name__ == "__main__":
    main()