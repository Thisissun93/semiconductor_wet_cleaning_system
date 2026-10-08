"""평가 웨이퍼 결과를 합성 데이터로 만듭니다.

반응 모델은 일반적인 SC1 세정 경향을 단순화한 시연용 가정입니다.

- 파티클 제거율(PRE): 온도·농도·메가소닉 출력이 높을수록 증가하지만 포화됩니다.
- 산화막 식각 손실: 온도에 지수적으로, 농도에 비례해 증가합니다.
- 패턴 손상 수: 메가소닉 출력이 높을수록 급격히 늘어납니다.

PRE는 0~100%에서 포화되므로 logit 변환, 손상 수는 개수 데이터이므로
제곱근 변환 후 회귀합니다.

세 반응이 서로 반대 방향으로 움직여, 판정 기준을 모두 만족하는 조건을
찾는 것이 평가의 핵심이 되도록 설계했습니다. 실제 장비·약액 조건이 아닙니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from src.evaluation.design import FACTORS


@dataclass(frozen=True)
class Response:
    """평가 반응값과 판정 기준."""

    key: str
    label: str
    unit: str
    goal: str  # "maximize" 또는 "minimize"
    spec: float  # maximize면 하한, minimize면 상한
    ideal: float  # 만족도 1이 되는 값
    decimals: int
    transform: str = "none"  # 회귀 전 변환: "none", "logit_pct", "sqrt"

    def forward(self, values: np.ndarray) -> np.ndarray:
        """회귀에 쓰는 변환 척도로 바꿉니다."""

        v = np.asarray(values, dtype=float)
        if self.transform == "logit_pct":
            frac = np.clip(v, 0.5, 99.5) / 100.0
            return np.log(frac / (1.0 - frac))
        if self.transform == "sqrt":
            return np.sqrt(np.clip(v, 0.0, None))
        return v

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """변환 척도의 값을 원래 단위로 되돌립니다."""

        v = np.asarray(values, dtype=float)
        if self.transform == "logit_pct":
            return 100.0 / (1.0 + np.exp(-v))
        if self.transform == "sqrt":
            return np.clip(v, 0.0, None) ** 2
        return v


RESPONSES: Final[tuple[Response, ...]] = (
    Response("pre_pct", "파티클 제거율(PRE)", "%", "maximize", 95.0, 99.0, 1, "logit_pct"),
    Response("oxide_loss_a", "산화막 식각 손실", "Å", "minimize", 10.0, 5.0, 2),
    Response("pattern_damage", "패턴 손상 수", "개/wafer", "minimize", 3.0, 0.0, 1, "sqrt"),
)

RESPONSE_KEYS: Final[tuple[str, ...]] = tuple(r.key for r in RESPONSES)


def true_responses(coded: np.ndarray) -> dict[str, np.ndarray]:
    """노이즈 없는 기대 반응값(시연용 가정 모델)을 계산합니다."""

    x = np.atleast_2d(np.asarray(coded, dtype=float))
    t, c, p = x[:, 0], x[:, 1], x[:, 2]

    logit = 1.2 + 0.9 * t + 0.7 * c + 1.3 * p + 0.3 * t * p - 0.4 * p**2
    pre = 100.0 / (1.0 + np.exp(-logit))

    oxide_loss = 2.6 * np.exp(0.95 * t) * (1.0 + 0.55 * c) + 0.3 * p

    damage_rate = np.exp(-0.2 + 0.25 * t + 1.3 * p + 0.6 * p**2)

    return {
        "pre_pct": pre,
        "oxide_loss_a": oxide_loss,
        "pattern_damage": damage_rate,
    }


def simulate_responses(
    design: pd.DataFrame,
    seed: int = 11,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """실험 계획표에 합성 평가 결과(측정 노이즈 포함)를 붙입니다."""

    coded_cols = [f"{factor.key}_coded" for factor in FACTORS]
    coded = design[coded_cols].to_numpy(dtype=float)
    expected = true_responses(coded)

    rng = np.random.default_rng(seed)
    result = design.copy()
    result["pre_pct"] = np.clip(
        expected["pre_pct"] + rng.normal(0.0, 0.8 * noise_scale, len(design)),
        0.0,
        100.0,
    ).round(2)
    result["oxide_loss_a"] = np.clip(
        expected["oxide_loss_a"]
        + rng.normal(0.0, 0.3 * noise_scale, len(design)),
        0.0,
        None,
    ).round(2)
    result["pattern_damage"] = rng.poisson(expected["pattern_damage"])
    return result
