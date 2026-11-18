"""독립 계산 대조 결과를 docs/stats_validation.md로 남깁니다.

실행: python scripts/validate_stats.py   (대조 실패 시 종료 코드 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.validation import run  # noqa: E402


def main() -> int:
    checks, model = run()
    lines = [
        "# 독립 계산 대조",
        "",
        "`python scripts/validate_stats.py`로 다시 만들 수 있습니다. "
        "앱 코드와 다른 방식(NumPy 직접 계산, 설정 구간표 재판정, "
        "scipy 회귀)으로 같은 값을 다시 계산해 비교했습니다.",
        "",
        "| 영역 | 대조 항목 | 비교 수 | 불일치 | 최대 오차 | 결과 |",
        "|---|---|---|---|---|---|",
    ]
    for check in checks:
        error = "-" if check.max_abs_error is None else f"{check.max_abs_error:.1e}"
        lines.append(
            f"| {check.group} | {check.name} | {check.compared:,} | "
            f"{check.mismatched} | {error} | {'통과' if check.passed else '실패'} |"
        )
    notes = [check.note for check in checks if check.note]
    if notes:
        lines += [""] + [f"- {note}" for note in notes]
    lines += [
        "",
        "## 예지보전 모델 평가의 한계 (공개 항목)",
        "",
        "계산 대조와 별개로, 현재 데이터로는 모델 성능을 주장하기 어렵습니다.",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 학습 데이터 | {model['rows']}행, 양성 비율 {model['positive_rate']:.1%} |",
        f"| 평가 데이터 | {model['test_rows']}행 중 음성 {model['test_negatives']}행 |",
        f"| 모델 정확도 | {model['model_accuracy']:.3f} |",
        f"| 전부 '이상'으로 찍는 기준선 정확도 | {model['majority_accuracy']:.3f} |",
        f"| 음성을 맞힌 건수 (TN) | {model['true_negatives']} |",
        f"| ROC-AUC | {model['roc_auc']:.3f} (음성 {model['test_negatives']}건 기준) |",
        "",
        "- 정확도가 기준선과 같고 음성을 하나도 맞히지 못했습니다.",
        "- 학습/평가를 무작위로 나눠, 겹치는 5 LOT 윈도가 양쪽에 섞입니다. "
        "시간 순서 분할이 필요합니다.",
        "- 라벨 0 중 일부는 이후 LOT가 10개 미만이라 실제 정상인지 알 수 없습니다(위 주석).",
        "- 정상 구간이 긴 데이터를 추가로 만들거나 실제 정비 이력이 있어야 "
        "의미 있는 평가가 됩니다.",
        "",
    ]
    output = PROJECT_ROOT / "docs" / "stats_validation.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
