from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "process_limits.yaml"


def main() -> None:
    if not CONFIG_PATH.exists():
        print(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
        return

    with CONFIG_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    print("=" * 60)
    print("공정 기준 설정 확인")
    print("=" * 60)

    print(
        "공정명:",
        config["project"]["process_name"],
    )

    print(
        "수율 정상 기준:",
        config["quality"]["yield_percent"]["normal_min"],
        "%",
    )

    print(
        "Particle 경고 상한:",
        config["quality"]["particle_count"]["warning_max"],
    )

    print(
        "Filter 차압 Critical 기준:",
        config["equipment"][
            "filter_differential_pressure"
        ]["critical_min"],
        "bar",
    )

    print(
        "Report 분석 LOT:",
        config["report"]["recent_lot_window"],
        "개",
    )

    print("=" * 60)
    print("process_limits.yaml 로드 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()