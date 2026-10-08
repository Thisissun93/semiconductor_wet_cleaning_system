"""2차 반응표면 회귀, 권장 조건 탐색, 확인 실험 계획을 만듭니다."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.design import FACTORS, coded_to_actual
from src.evaluation.simulate import RESPONSES, Response

TERM_NAMES: tuple[str, ...] = (
    "절편",
    "T",
    "C",
    "P",
    "T×C",
    "T×P",
    "C×P",
    "T²",
    "C²",
    "P²",
)
"""T=SC1 온도, C=NH4OH 농도, P=메가소닉 출력 (모두 코드값)."""


def model_matrix(coded: np.ndarray) -> np.ndarray:
    """2차 모델 설계행렬(절편·주효과·교호작용·제곱항)을 만듭니다."""

    x = np.atleast_2d(np.asarray(coded, dtype=float))
    t, c, p = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack(
        [np.ones(len(x)), t, c, p, t * c, t * p, c * p, t**2, c**2, p**2]
    )


@dataclass
class QuadraticFit:
    """한 반응값에 대한 2차 회귀 결과."""

    response: Response
    coefficients: np.ndarray
    xtx_inv: np.ndarray
    sigma: float
    dof: int
    r_squared: float
    adj_r_squared: float

    def predict(self, coded: np.ndarray) -> np.ndarray:
        """원래 단위의 예측값."""

        return self.response.inverse(model_matrix(coded) @ self.coefficients)

    def prediction_interval(
        self, coded: np.ndarray, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """새 웨이퍼 1장에 대한 예측구간(원래 단위)을 계산합니다.

        변환 척도에서 구한 구간을 역변환하므로 0~100% 같은 범위를 벗어나지 않습니다.
        """

        x = model_matrix(coded)
        center = x @ self.coefficients
        leverage = np.einsum("ij,jk,ik->i", x, self.xtx_inv, x)
        half = stats.t.ppf(0.5 + confidence / 2, self.dof) * self.sigma * np.sqrt(
            1.0 + leverage
        )
        return (
            self.response.inverse(center - half),
            self.response.inverse(center + half),
        )

    def coefficient_table(self) -> pd.DataFrame:
        """계수표. 변환한 반응값은 변환 척도의 계수입니다."""

        se = self.sigma * np.sqrt(np.diag(self.xtx_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(se > 0, self.coefficients / se, np.nan)
        p_values = 2 * stats.t.sf(np.abs(t_values), self.dof)
        return pd.DataFrame(
            {
                "항": TERM_NAMES,
                "계수": self.coefficients,
                "표준오차": se,
                "t": t_values,
                "p": p_values,
                "유의(p<0.05)": p_values < 0.05,
            }
        )


def fit_quadratic(results: pd.DataFrame, response: Response) -> QuadraticFit:
    """실험 결과로 2차 반응표면 모델을 적합합니다."""

    coded = results[[f"{f.key}_coded" for f in FACTORS]].to_numpy(dtype=float)
    y = response.forward(results[response.key].to_numpy(dtype=float))
    x = model_matrix(coded)

    n, k = x.shape
    if n <= k:
        raise ValueError(f"실험 수({n})가 모델 항 수({k})보다 많아야 합니다.")

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    dof = n - k
    sse = float(residual @ residual)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof

    return QuadraticFit(
        response=response,
        coefficients=beta,
        xtx_inv=xtx_inv,
        sigma=float(np.sqrt(sse / dof)),
        dof=dof,
        r_squared=r2,
        adj_r_squared=adj,
    )


def coded_grid(steps: int = 21) -> np.ndarray:
    levels = np.linspace(-1.0, 1.0, steps)
    return np.array(list(product(levels, repeat=len(FACTORS))))


def _desirability(values: np.ndarray, response: Response) -> np.ndarray:
    """Derringer 만족도(0~1). 판정 기준 밖이면 0입니다."""

    if response.goal == "maximize":
        d = (values - response.spec) / (response.ideal - response.spec)
    else:
        d = (response.spec - values) / (response.spec - response.ideal)
    return np.clip(d, 0.0, 1.0)


def predict_grid(fits: dict[str, QuadraticFit], steps: int = 21) -> pd.DataFrame:
    """조건 격자 전체에 대한 예측값, 판정, 종합 만족도를 계산합니다."""

    grid = coded_grid(steps)
    table = coded_to_actual(grid)
    for i, factor in enumerate(FACTORS):
        table[f"{factor.key}_coded"] = grid[:, i]

    overall = np.ones(len(grid))
    feasible = np.ones(len(grid), dtype=bool)
    for response in RESPONSES:
        predicted = fits[response.key].predict(grid)
        table[response.key] = predicted
        if response.goal == "maximize":
            feasible &= predicted >= response.spec
        else:
            feasible &= predicted <= response.spec
        overall *= _desirability(predicted, response)

    table["feasible"] = feasible
    table["desirability"] = overall ** (1.0 / len(RESPONSES))
    return table


def recommend_condition(grid_table: pd.DataFrame) -> pd.Series | None:
    """판정 기준을 모두 만족하는 조건 중 종합 만족도가 가장 높은 조건."""

    feasible = grid_table[grid_table["feasible"]]
    if feasible.empty:
        return None
    return feasible.sort_values("desirability", ascending=False).iloc[0]


def confirmation_plan(
    fits: dict[str, QuadraticFit],
    condition: pd.Series,
    wafers: int = 3,
) -> pd.DataFrame:
    """권장 조건의 확인 실험 계획표(예측값·95% 예측구간·판정 기준)."""

    coded = np.array([[condition[f"{f.key}_coded"] for f in FACTORS]])
    rows = []
    for response in RESPONSES:
        fit = fits[response.key]
        low, high = fit.prediction_interval(coded)
        criterion = (
            f"≥ {response.spec:g}" if response.goal == "maximize" else f"≤ {response.spec:g}"
        )
        risk_side = low[0] if response.goal == "maximize" else high[0]
        within = (
            risk_side >= response.spec
            if response.goal == "maximize"
            else risk_side <= response.spec
        )
        rows.append(
            {
                "평가 항목": response.label,
                "단위": response.unit,
                "판정 기준": criterion,
                "예측값": round(float(fit.predict(coded)[0]), response.decimals),
                "95% 예측구간": (
                    f"{low[0]:.{response.decimals}f} ~ {high[0]:.{response.decimals}f}"
                ),
                "예측구간이 기준 안": "예" if within else "아니오(여유 부족)",
                "확인 웨이퍼 수": wafers if within else wafers + 2,
                "조치 제안": (
                    "권장 조건으로 확인"
                    if within
                    else "확인 웨이퍼를 늘리고, 미달 시 조건을 기준 안쪽으로 후퇴"
                ),
            }
        )
    return pd.DataFrame(rows)
