"""업무 시나리오 측정: 이상 LOT 판정 → 원인 후보 → 조치 추천.

DB 사본에서 규칙 판정·원인 분석·조치 추천을 5회 실행해 시간을 재고,
CRITICAL 항목이 가장 많은 LOT 한 건의 결과를 사례로 남깁니다.
원본 DB는 건드리지 않습니다.

실행: python scripts/benchmark_scenario.py
출력: docs/BENCHMARK.md, docs/benchmark.json
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import rule_detector  # noqa: E402
from src.recommendation import action_engine  # noqa: E402
from src.root_cause import rule_based_rca  # noqa: E402

REPEAT = 5
DOCS = PROJECT_ROOT / "docs"


def _run_once(database: Path) -> dict[str, float]:
    for module in (rule_detector, rule_based_rca, action_engine):
        module.DATABASE_PATH = database  # type: ignore[misc]
    timings: dict[str, float] = {}
    for name, step in (
        ("detect", rule_detector.run_rule_detection),
        ("root_cause", rule_based_rca.run_root_cause_analysis),
        ("action", action_engine.run_action_engine),
    ):
        start = time.perf_counter()
        step()
        timings[name] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    return timings


def _scenario(connection: sqlite3.Connection) -> dict[str, object]:
    lot_id, equipment_id = connection.execute(
        """
        SELECT lot_id, equipment_id FROM anomaly_result
        GROUP BY lot_id
        ORDER BY SUM(severity = 'CRITICAL') DESC, COUNT(*) DESC, lot_id
        LIMIT 1
        """
    ).fetchone()
    anomalies = connection.execute(
        """
        SELECT parameter_name, severity, ROUND(measured_value, 3)
        FROM anomaly_result WHERE lot_id = ?
        ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'WARNING' THEN 1
                 ELSE 2 END, parameter_name
        """,
        (lot_id,),
    ).fetchall()
    causes = connection.execute(
        """
        SELECT cause_rank, cause_name, ROUND(contribution_percent, 1),
               confidence_level
        FROM root_cause_result WHERE lot_id = ? ORDER BY cause_rank
        """,
        (lot_id,),
    ).fetchall()
    actions = connection.execute(
        """
        SELECT action_rank, action_name, priority, responsible_department
        FROM action_recommendation WHERE lot_id = ? ORDER BY action_rank
        """,
        (lot_id,),
    ).fetchall()
    return {
        "lot_id": lot_id,
        "equipment_id": equipment_id,
        "anomalies": anomalies,
        "causes": causes,
        "actions": actions,
    }


def main() -> None:
    source = PROJECT_ROOT / "database" / "wet_cleaning.db"
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "wet_cleaning.db"
        shutil.copy(source, database)
        runs = [_run_once(database) for _ in range(REPEAT)]
        connection = sqlite3.connect(database)
        try:
            lots = connection.execute("SELECT COUNT(*) FROM lot_history").fetchone()[0]
            severity = dict(
                connection.execute(
                    "SELECT severity, COUNT(*) FROM anomaly_result GROUP BY severity"
                ).fetchall()
            )
            flagged_lots = connection.execute(
                "SELECT COUNT(DISTINCT lot_id) FROM anomaly_result "
                "WHERE severity IN ('WARNING', 'CRITICAL')"
            ).fetchone()[0]
            cause_rows = connection.execute(
                "SELECT COUNT(*) FROM root_cause_result"
            ).fetchone()[0]
            action_rows = connection.execute(
                "SELECT COUNT(*) FROM action_recommendation"
            ).fetchone()[0]
            scenario = _scenario(connection)
        finally:
            connection.close()

    median = {key: statistics.median(run[key] for run in runs) for key in runs[0]}
    result = {
        "repeat": REPEAT,
        "lots": lots,
        "judgements": lots * 17,
        "severity": severity,
        "flagged_lots": flagged_lots,
        "cause_rows": cause_rows,
        "action_rows": action_rows,
        "median_seconds": median,
        "scenario": scenario,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS / "BENCHMARK.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(median, indent=2))


def _markdown(result: dict) -> str:
    median = result["median_seconds"]
    scenario = result["scenario"]
    lines = [
        "# 업무 시나리오 측정",
        "",
        "`python scripts/benchmark_scenario.py`로 다시 만들 수 있습니다. "
        "DB 사본에서 규칙 판정 → 원인 분석 → 조치 추천을 "
        f"{result['repeat']}회 실행한 중앙값입니다.",
        "",
        "## 처리량과 시간",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 대상 | LOT {result['lots']}개 × 판정 항목 17개 = {result['judgements']:,}건 |",
        "| 판정 결과 | "
        + ", ".join(
            f"{key} {result['severity'].get(key, 0)}"
            for key in ("CRITICAL", "WARNING", "WATCH")
        )
        + " |",
        f"| WARNING 이상 LOT | {result['flagged_lots']}개 |",
        f"| 원인 후보 / 조치 추천 | {result['cause_rows']}건 / {result['action_rows']}건 |",
        f"| 규칙 판정 | {median['detect'] * 1000:.0f} ms |",
        f"| 원인 분석 | {median['root_cause'] * 1000:.0f} ms |",
        f"| 조치 추천 | {median['action'] * 1000:.0f} ms |",
        f"| **합계** | **{median['total'] * 1000:.0f} ms** |",
        "",
        "시간은 실행 환경에 따라 달라집니다. 수작업 소요 시간과의 비교는 "
        "측정하지 않았습니다.",
        "",
        f"## 사례: {scenario['lot_id']} ({scenario['equipment_id']})",
        "",
        "CRITICAL 항목이 가장 많은 LOT입니다.",
        "",
        "| 항목 | 등급 | 측정값 |",
        "|---|---|---|",
    ]
    lines += [f"| {p} | {s} | {v} |" for p, s, v in scenario["anomalies"]]
    lines += ["", "| 순위 | 원인 후보 | 기여도(%) | 신뢰도 |", "|---|---|---|---|"]
    lines += [f"| {r} | {n} | {c} | {l} |" for r, n, c, l in scenario["causes"]]
    lines += ["", "| 순위 | 조치 | 우선순위 | 담당 |", "|---|---|---|---|"]
    lines += [f"| {r} | {n} | {p} | {d} |" for r, n, p, d in scenario["actions"]]
    lines += [
        "",
        "원인 기여도는 규칙 점수의 상대 비율입니다. 통계적 인과 추정이 아니므로 "
        "현장 확인 전 우선순위로만 씁니다.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
