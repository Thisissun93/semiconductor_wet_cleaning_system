"""평가용 실험 계획(Face-centered Central Composite Design)을 만듭니다."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Final

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Factor:
    """실험 인자와 실제 단위 범위."""

    key: str
    label: str
    unit: str
    low: float
    high: float

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2


FACTORS: Final[tuple[Factor, ...]] = (
    Factor("temperature_c", "SC1 온도", "°C", 25.0, 65.0),
    Factor("nh4oh_wt_pct", "NH4OH 농도", "wt%", 0.2, 1.0),
    Factor("megasonic_w", "메가소닉 출력", "W", 200.0, 800.0),
)

FACTOR_KEYS: Final[tuple[str, ...]] = tuple(f.key for f in FACTORS)


def coded_to_actual(coded: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    """코드값(-1~+1)을 실제 단위로 바꿉니다."""

    values = np.asarray(coded, dtype=float)
    columns = {
        factor.key: factor.center + values[:, i] * factor.half_range
        for i, factor in enumerate(FACTORS)
    }
    return pd.DataFrame(columns)


def actual_to_coded(actual: pd.DataFrame) -> np.ndarray:
    """실제 단위 값을 코드값(-1~+1)으로 바꿉니다."""

    return np.column_stack(
        [
            (actual[factor.key].to_numpy(dtype=float) - factor.center)
            / factor.half_range
            for factor in FACTORS
        ]
    )


def build_face_centered_ccd(
    center_points: int = 4,
    seed: int = 7,
) -> pd.DataFrame:
    """3인자 Face-centered CCD 실험 계획표를 만듭니다.

    요인점 8개 + 축점 6개 + 중심점 n개로 구성하며,
    시간에 따른 장비 변동이 특정 조건에 몰리지 않도록 실행 순서를 무작위화합니다.
    """

    if center_points < 1:
        raise ValueError("center_points는 1 이상이어야 합니다.")

    n_factors = len(FACTORS)
    factorial = [list(point) for point in product((-1.0, 1.0), repeat=n_factors)]
    axial: list[list[float]] = []
    for i in range(n_factors):
        for level in (-1.0, 1.0):
            point = [0.0] * n_factors
            point[i] = level
            axial.append(point)
    center = [[0.0] * n_factors for _ in range(center_points)]

    coded = np.array(factorial + axial + center, dtype=float)
    point_type = (
        ["요인점"] * len(factorial)
        + ["축점"] * len(axial)
        + ["중심점"] * len(center)
    )

    design = coded_to_actual(coded)
    for i, factor in enumerate(FACTORS):
        design[f"{factor.key}_coded"] = coded[:, i]
    design.insert(0, "point_type", point_type)
    design.insert(0, "std_order", np.arange(1, len(design) + 1))

    rng = np.random.default_rng(seed)
    run_order = rng.permutation(len(design)) + 1
    design.insert(0, "run_order", run_order)
    design.insert(0, "wafer_id", [f"EVAL-W{n:02d}" for n in run_order])

    return design.sort_values("run_order").reset_index(drop=True)
