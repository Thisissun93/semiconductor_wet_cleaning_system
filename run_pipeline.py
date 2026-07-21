from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

DATABASE_PATH: Final[Path] = (
    PROJECT_ROOT / "database" / "wet_cleaning.db"
)


@dataclass(frozen=True)
class PipelineStep:
    """
    파이프라인 한 단계를 표현합니다.

    Attributes:
        name:
            터미널에 표시할 단계 이름

        script_path:
            실행할 Python 파일의 프로젝트 기준 상대 경로

        description:
            해당 단계가 수행하는 작업 설명
    """

    name: str
    script_path: Path
    description: str


PIPELINE_STEPS: Final[list[PipelineStep]] = [
    PipelineStep(
        name="01. Master Data 등록",
        script_path=Path(
            "src/database/seed_master_data.py"
        ),
        description=(
            "설비, Recipe, Chemical LOT 기준정보를 "
            "SQLite에 등록합니다."
        ),
    ),
    PipelineStep(
        name="02. 가상 생산 데이터 생성",
        script_path=Path(
            "src/data_generation/generate_data.py"
        ),
        description=(
            "LOT, Wafer, 공정조건, 센서, 품질 및 "
            "유지보수 데이터를 생성합니다."
        ),
    ),
    PipelineStep(
        name="03. 규칙 기반 이상 감지",
        script_path=Path(
            "src/detection/rule_detector.py"
        ),
        description=(
            "공정 기준과 실제값을 비교해 "
            "WATCH, WARNING, CRITICAL을 판정합니다."
        ),
    ),
    PipelineStep(
        name="04. Root Cause Analysis",
        script_path=Path(
            "src/root_cause/rule_based_rca.py"
        ),
        description=(
            "이상 항목 조합을 분석해 "
            "LOT별 원인 TOP 3를 생성합니다."
        ),
    ),
    PipelineStep(
        name="05. Action Recommendation",
        script_path=Path(
            "src/recommendation/action_engine.py"
        ),
        description=(
            "원인 분석 결과에 따라 "
            "LOT별 권장 조치 TOP 3를 생성합니다."
        ),
    ),
    PipelineStep(
        name="06. 통합 결과 검증",
        script_path=Path(
            "src/analysis/check_analysis_results.py"
        ),
        description=(
            "생산 데이터부터 조치 추천까지의 "
            "연결 상태를 검증합니다."
        ),
    ),
]


def print_header(
    title: str,
    character: str = "=",
    width: int = 100,
) -> None:
    """터미널 구분선을 출력합니다."""

    print("\n" + character * width)
    print(title)
    print(character * width)


def validate_project_files() -> None:
    """
    파이프라인 실행에 필요한 파일이 존재하는지 확인합니다.

    파일이 하나라도 없으면 실제 실행 전에 중단해
    일부 단계만 실행되는 상황을 방지합니다.
    """

    required_files = [
        PROJECT_ROOT / "database" / "schema.sql",
        PROJECT_ROOT / "config" / "process_limits.yaml",
    ]

    required_files.extend(
        PROJECT_ROOT / step.script_path
        for step in PIPELINE_STEPS
    )

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_file_text = "\n".join(
            f"- {path.relative_to(PROJECT_ROOT)}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "파이프라인 실행에 필요한 파일이 없습니다.\n"
            f"{missing_file_text}"
        )


def create_database_if_missing() -> None:
    """
    DB가 없을 때만 create_database.py를 실행합니다.

    기존 DB가 있다면 다시 생성하지 않으므로
    불필요한 초기화를 방지합니다.
    """

    if DATABASE_PATH.exists():
        print(
            f"DB 확인 완료: "
            f"{DATABASE_PATH.relative_to(PROJECT_ROOT)}"
        )
        return

    create_database_script = (
        PROJECT_ROOT
        / "src"
        / "database"
        / "create_database.py"
    )

    if not create_database_script.exists():
        raise FileNotFoundError(
            "DB 생성 스크립트를 찾을 수 없습니다.\n"
            f"경로: {create_database_script}"
        )

    print_header("SQLite DB가 없어 새로 생성합니다.")

    run_python_script(
        script_path=create_database_script,
        step_name="00. SQLite Database 생성",
    )

    if not DATABASE_PATH.exists():
        raise RuntimeError(
            "DB 생성 스크립트는 실행됐지만 "
            "wet_cleaning.db가 생성되지 않았습니다."
        )


def run_python_script(
    script_path: Path,
    step_name: str,
) -> float:
    """
    현재 가상환경의 Python으로 지정한 파일을 실행합니다.

    Returns:
        float: 해당 단계 실행 시간(초)
    """

    if not script_path.exists():
        raise FileNotFoundError(
            f"실행 파일을 찾을 수 없습니다: {script_path}"
        )

    start_time = time.perf_counter()

    process = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    if process.stdout:
        print(process.stdout.rstrip())

    if process.returncode != 0:
        if process.stderr:
            print("\n[오류 출력]")
            print(process.stderr.rstrip())

        raise RuntimeError(
            f"{step_name} 실행에 실패했습니다.\n"
            f"종료 코드: {process.returncode}"
        )

    if process.stderr:
        print("\n[경고 출력]")
        print(process.stderr.rstrip())

    return elapsed_seconds


def run_pipeline() -> dict[str, float]:
    """
    전체 Wet Cleaning 분석 파이프라인을 순서대로 실행합니다.

    한 단계라도 실패하면 다음 단계는 실행하지 않습니다.
    """

    validate_project_files()
    create_database_if_missing()

    execution_times: dict[str, float] = {}

    for step_number, step in enumerate(
        PIPELINE_STEPS,
        start=1,
    ):
        print_header(step.name)
        print(step.description)
        print(
            f"실행 파일: {step.script_path}"
        )
        print("-" * 100)

        absolute_script_path = (
            PROJECT_ROOT / step.script_path
        )

        elapsed_seconds = run_python_script(
            script_path=absolute_script_path,
            step_name=step.name,
        )

        execution_times[step.name] = elapsed_seconds

        print(
            f"\n완료 시간: {elapsed_seconds:.2f}초"
        )

        if step_number < len(PIPELINE_STEPS):
            print("다음 단계로 이동합니다.")

    return execution_times


def print_pipeline_summary(
    execution_times: dict[str, float],
    total_elapsed_seconds: float,
) -> None:
    """전체 파이프라인 실행 결과를 요약 출력합니다."""

    print_header("Wet Cleaning Pipeline 실행 완료")

    for step_name, elapsed_seconds in execution_times.items():
        print(
            f"{step_name:<35}"
            f"{elapsed_seconds:>8.2f}초"
        )

    print("-" * 100)
    print(
        f"{'전체 실행 시간':<35}"
        f"{total_elapsed_seconds:>8.2f}초"
    )

    print("\n최종 처리 흐름")

    print(
        "생산 데이터 → 이상 감지 → Root Cause → "
        "조치 추천 → 통합 검증"
    )

    print("\n파이프라인이 정상적으로 완료됐습니다.")


def main() -> None:
    """프로그램 시작 지점입니다."""

    total_start_time = time.perf_counter()

    try:
        execution_times = run_pipeline()

        total_elapsed_seconds = (
            time.perf_counter() - total_start_time
        )

        print_pipeline_summary(
            execution_times=execution_times,
            total_elapsed_seconds=total_elapsed_seconds,
        )

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as error:
        total_elapsed_seconds = (
            time.perf_counter() - total_start_time
        )

        print_header("Wet Cleaning Pipeline 실행 실패")

        print(error)

        print(
            f"\n실패 시점까지 실행 시간: "
            f"{total_elapsed_seconds:.2f}초"
        )

        print(
            "\n실패한 단계 이후의 작업은 실행하지 않았습니다."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()