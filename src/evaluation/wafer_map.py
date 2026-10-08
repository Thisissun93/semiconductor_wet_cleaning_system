"""300 mm 웨이퍼 49점 식각량 지도와 균일도를 계산합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

WAFER_RADIUS_MM = 150.0
RING_RADII_MM = (0.0, 49.0, 98.0, 145.0)
RING_COUNTS = (1, 8, 16, 24)


def forty_nine_points() -> pd.DataFrame:
    """중심 1점 + 동심원 8·16·24점의 표준 49점 측정 좌표."""

    rows = []
    for radius, count in zip(RING_RADII_MM, RING_COUNTS):
        for k in range(count):
            angle = 2 * np.pi * k / count
            rows.append(
                {
                    "x_mm": radius * np.cos(angle),
                    "y_mm": radius * np.sin(angle),
                    "r_mm": radius,
                }
            )
    points = pd.DataFrame(rows)
    points.insert(0, "site", np.arange(1, len(points) + 1))
    return points


def simulate_wafer_map(
    mean_loss_a: float,
    temperature_coded: float,
    seed: int = 3,
) -> pd.DataFrame:
    """매엽 스핀 세정의 중심-가장자리 식각 차이를 단순화해 49점 값을 만듭니다.

    중심에 공급된 약액이 바깥으로 퍼지며 식는다고 가정해,
    온도가 높을수록 중심이 더 많이 식각되는(center-fast) 형태가 강해집니다.
    """

    points = forty_nine_points()
    rng = np.random.default_rng(seed)
    radial = (points["r_mm"] / WAFER_RADIUS_MM) ** 2
    strength = 0.04 + 0.05 * (temperature_coded + 1.0) / 2.0
    angle = np.arctan2(points["y_mm"], points["x_mm"])
    azimuthal = 0.01 * np.cos(angle) * points["r_mm"] / WAFER_RADIUS_MM
    shape = 1.0 + strength * (0.5 - radial) + azimuthal
    values = mean_loss_a * shape + rng.normal(0.0, 0.05, len(points))
    points["oxide_loss_a"] = values.round(3)
    return points


def uniformity_summary(wafer: pd.DataFrame, column: str = "oxide_loss_a") -> dict[str, float]:
    """평균, 1σ 비균일도(%), 범위 기반 비균일도(%)."""

    values = wafer[column].to_numpy(dtype=float)
    mean = float(values.mean())
    return {
        "mean": mean,
        "min": float(values.min()),
        "max": float(values.max()),
        "nu_1sigma_pct": float(values.std(ddof=1) / mean * 100.0),
        "nu_range_pct": float((values.max() - values.min()) / (2.0 * mean) * 100.0),
    }
